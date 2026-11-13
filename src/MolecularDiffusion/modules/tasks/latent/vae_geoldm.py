"""GeoLDM stage 1: E(n) hierarchical VAE task.

Thin subclass of `MolecularDiffusion.modules.tasks.diffusion.GeomMolecularGenerative`
wrapping the ported `EnHierarchicalVAE` (`modules/models/geoldm/diffusion.py`)
as `self.model`. See docs/model_integrations/geoldm/INTEGRATION_PLAN.md for
the full integration plan (revision 2).

Everything except `density_estimation`/`sample`/`predict_and_target`/
`evaluate` is inherited unmodified from `GeomMolecularGenerative` --
`__init__`, `preprocess`. `density_estimation` is overridden because the
base class's version subtracts a `-log_pN` node-count correction that
converts a density into an NLL over (x, h, N) jointly; that correction
doesn't apply to a VAE ELBO over (x, h) alone (see plan). `forward` needs no
separate override -- the base class's `forward` just calls
`self.density_estimation(batch)`, so overriding `density_estimation` alone
is enough (polymorphic dispatch). `sample` raises `NotImplementedError`: a
VAE alone has no cardinality-varying unconditional prior over (x, h),
matching vanilla GeoLDM where a VAE-only run never calls `sample()`.

`predict_and_target`/`evaluate` are overridden (attempt 2 fix, see
INTEGRATION_PLAN.md Tester attempt_1 diagnosis and Executor attempt 2 log):
`task_type='vae_geoldm'` prefix-matches `TASK_REGISTRY['vae']` in
`runmodes/train/eval.py`, which hardcodes `metric_key='match_rate'`. The
base class's inherited `evaluate()` only returns
`{'val_negative_log_likelihood': ...}`, so `val_metric_dict['match_rate']`
KeyErrors after validation. Fix: `predict_and_target` additionally computes
a real per-atom reconstruction match rate (encode to the latent mode, decode
back, compare argmax categorical atom types against the input, masked by
`node_mask`) using `EnHierarchicalVAE.encode`/`decode` -- both already
implemented on the ported model, so this needs no new model-side plumbing --
and packs it as a second column alongside the NLL so `evaluate()` can emit
both `val_negative_log_likelihood` and `match_rate` from one aggregated
tensor (matching the `Engine.evaluate` pred/target cat-then-evaluate
contract in `core/engine.py:530-567`).
"""

from typing import List, Optional

import torch

from MolecularDiffusion import core
from MolecularDiffusion.utils import (
    assert_correctly_masked,
    assert_mean_zero_with_mask,
    check_mask_correct,
    random_rotation,
    remove_mean_with_mask,
    sample_center_gravity_zero_gaussian_with_mask,
)

from MolecularDiffusion.modules.tasks.diffusion import GeomMolecularGenerative
from MolecularDiffusion.modules.models.geoldm.diffusion import EnHierarchicalVAE
from MolecularDiffusion.modules.models.geoldm.networks import EGNN_decoder_QM9, EGNN_encoder_QM9


@core.Registry.register("GeoLDMVAETask")
class GeoLDMVAETask(GeomMolecularGenerative):
    """GeoLDM's `EnHierarchicalVAE` wrapped in the platform's `Task` contract."""

    def density_estimation(self, batch):
        """"""
        node_mask = batch["node_mask"].unsqueeze(2)
        edge_mask = batch["edge_mask"]
        x = batch["coords"]
        h = batch["node_feature"]
        charges = batch["charges"].unsqueeze(2)

        x = remove_mean_with_mask(x, node_mask)
        if self.augment_noise > 0:
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * self.augment_noise
            x = remove_mean_with_mask(x, node_mask)
        if self.data_augmentation:
            x = random_rotation(x).detach()

        check_mask_correct([x, h], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        bs, n_nodes, n_dims = x.size()
        assert_correctly_masked(x, node_mask)
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
        h = {"categorical": h, "integer": charges}

        # No conditioning this pass (see plan's "explicitly out of scope").
        context = None

        # No -log_pN correction here -- see module docstring.
        nll = self.model(x, h, node_mask, edge_mask, context)
        loss = nll.mean(0)
        metric = {"train_negative_log_likelihood": loss}
        all_loss = loss

        return all_loss, metric

    def sample(self, *args, **kwargs):
        raise NotImplementedError(
            "GeoLDMVAETask has no cardinality-varying unconditional prior over "
            "(x, h) -- standalone generation from the VAE stage is not supported "
            "(matches vanilla GeoLDM, where a VAE-only run never calls sample())."
        )

    def _reconstruction_match_rate(self, batch) -> torch.Tensor:
        """Per-atom categorical reconstruction accuracy for this batch.

        Encodes to the latent mode (no sampling noise) and decodes back via
        the ported `EnHierarchicalVAE.encode`/`decode` (both already exist on
        the model -- no new model-side plumbing needed), then compares
        argmax atom types against the input, masked by `node_mask`. This is
        the `match_rate` metric `TASK_REGISTRY['vae']` (`runmodes/train/
        eval.py`) expects to find in the validation metric dict.
        """
        node_mask = batch["node_mask"].unsqueeze(2)
        edge_mask = batch["edge_mask"]
        x = batch["coords"]
        h = batch["node_feature"]
        charges = batch["charges"].unsqueeze(2)

        x = remove_mean_with_mask(x, node_mask)
        bs, n_nodes, _ = x.size()
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
        h_in = {"categorical": h, "integer": charges}

        with torch.no_grad():
            z_x_mu, _, z_h_mu, _ = self.model.encode(
                x, h_in, node_mask, edge_mask, context=None
            )
            z_xh = torch.cat([z_x_mu, z_h_mu], dim=2)
            _, h_recon = self.model.decode(
                z_xh, node_mask, edge_mask, context=None
            )

        pred_types = h_recon["categorical"].argmax(dim=-1)
        true_types = h.argmax(dim=-1)
        mask = node_mask.squeeze(2).bool()
        correct = ((pred_types == true_types) & mask).sum().float()
        total = mask.sum().float().clamp(min=1.0)
        return correct / total

    def predict_and_target(self, batch):
        """See module docstring -- packs `match_rate` as a second column
        alongside the base class's NLL so `evaluate()` can emit both."""
        all_loss, _ = super().predict_and_target(batch)
        match_rate = self._reconstruction_match_rate(batch)
        pred = torch.stack(
            [all_loss.reshape(()), match_rate.reshape(())], dim=0
        ).unsqueeze(0)
        target = torch.zeros_like(pred)
        return pred, target

    def evaluate(self, pred, target):
        """Emits `val_negative_log_likelihood` (unchanged) plus `match_rate`
        (attempt 2 fix -- see module docstring)."""
        return {
            "val_negative_log_likelihood": pred[:, 0].mean(),
            "match_rate": pred[:, 1].mean(),
        }


class GeoLDMVAETaskFactory:
    """Factory matching `train.py`'s `task_module.build()` /
    `task_module.task` instantiation pattern (see
    `diffusion_difflinker.py::DiffLinkerTaskFactory` for the precedent)."""

    def __init__(
        self,
        task_type: str,
        n_dims: int = 3,
        hidden_nf: int = 64,
        latent_nf: int = 1,
        n_layers: int = 4,
        attention: bool = False,
        tanh: bool = False,
        model: str = "egnn_dynamics",
        norm_constant: float = 0,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
        include_charges: bool = True,
        kl_weight: float = 0.01,
        normalize_factors: tuple = (1.0, 4.0, 10.0),
        augment_noise: float = 0.0,
        data_augmentation: bool = False,
        atom_vocab: Optional[List] = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.n_dims = n_dims
        self.hidden_nf = hidden_nf
        self.latent_nf = latent_nf
        self.n_layers = n_layers
        self.attention = attention
        self.tanh = tanh
        self.model_mode = model
        self.norm_constant = norm_constant
        self.inv_sublayers = inv_sublayers
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.include_charges = include_charges
        self.kl_weight = kl_weight
        self.normalize_factors = tuple(normalize_factors)
        self.augment_noise = augment_noise
        self.data_augmentation = data_augmentation
        self.atom_vocab = atom_vocab or kwargs.get("atom_vocab")
        self.kwargs = kwargs

    def build(self) -> "GeoLDMVAETask":
        if not self.atom_vocab:
            raise ValueError(
                "GeoLDMVAETaskFactory requires atom_vocab (injected from "
                "data.atom_vocab by cli/train.py) to size in_node_nf."
            )
        num_atom_types = len(self.atom_vocab)
        in_node_nf = num_atom_types + int(self.include_charges)

        encoder = EGNN_encoder_QM9(
            in_node_nf=in_node_nf,
            context_node_nf=0,
            out_node_nf=self.latent_nf,
            n_dims=self.n_dims,
            hidden_nf=self.hidden_nf,
            act_fn=torch.nn.SiLU(),
            n_layers=1,
            attention=self.attention,
            tanh=self.tanh,
            mode=self.model_mode,
            norm_constant=self.norm_constant,
            inv_sublayers=self.inv_sublayers,
            sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
            include_charges=self.include_charges,
        )
        decoder = EGNN_decoder_QM9(
            in_node_nf=self.latent_nf,
            context_node_nf=0,
            out_node_nf=in_node_nf,
            n_dims=self.n_dims,
            hidden_nf=self.hidden_nf,
            act_fn=torch.nn.SiLU(),
            n_layers=self.n_layers,
            attention=self.attention,
            tanh=self.tanh,
            mode=self.model_mode,
            norm_constant=self.norm_constant,
            inv_sublayers=self.inv_sublayers,
            sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
            include_charges=self.include_charges,
        )
        vae = EnHierarchicalVAE(
            encoder=encoder,
            decoder=decoder,
            in_node_nf=in_node_nf,
            n_dims=self.n_dims,
            latent_node_nf=self.latent_nf,
            kl_weight=self.kl_weight,
            norm_values=self.normalize_factors,
            include_charges=self.include_charges,
        )

        self.task = GeoLDMVAETask(
            diffusion_model=vae,
            n_node_dist={},
            augment_noise=self.augment_noise,
            data_augmentation=self.data_augmentation,
        )
        self.task.task_type = self.task_type
        self.task.atom_vocab = self.atom_vocab
        return self.task
