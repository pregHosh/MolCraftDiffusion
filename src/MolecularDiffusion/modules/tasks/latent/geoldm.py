"""GeoLDM stage 2: E(n) latent diffusion task.

Thin subclass of `MolecularDiffusion.modules.tasks.diffusion.GeomMolecularGenerative`
wrapping the ported `EnLatentDiffusion` (`modules/models/geoldm/diffusion.py`)
as `self.model`. See docs/model_integrations/geoldm/INTEGRATION_PLAN.md for
the full integration plan (revision 2).

`__init__`/`preprocess`/`forward`/`density_estimation`/`predict_and_target`/
`evaluate` are all inherited unmodified -- once the ported `EnLatentDiffusion`
accepts (and ignores) `reference_indices`/`reference_freeze_mode` (see
`modules/models/geoldm/diffusion.py`'s shim), the base class's non-graph
`density_estimation` branch works as-is, including the `-log_pN` node-count
correction (semantically correct here: this is a real NLL over p(x,h,N)).

`sample` is overridden: vanilla `EnLatentDiffusion.sample`'s signature
(`n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False`) and
2-tuple `(x, h)` return do not match the base class's `sample()` call site
(which expects `condition_tensor`/`condition_mode`/`n_frames`/`n_retrys`/
`t_retry`/`use_noised_conditioning` and a 3-tuple `(x, h, chain)` return) --
see plan's feasibility findings. `sample_chain`/`sample_guidance`/
`sample_conditonal`/etc. are left inherited but unsupported (out of scope).
"""

from typing import Dict, List, Optional

import torch

from MolecularDiffusion import core

from MolecularDiffusion.modules.tasks.diffusion import GeomMolecularGenerative
from MolecularDiffusion.modules.tasks.latent.vae_geoldm import GeoLDMVAETaskFactory
from MolecularDiffusion.modules.models.geoldm.diffusion import EnLatentDiffusion
from MolecularDiffusion.modules.models.geoldm.networks import EGNN_dynamics_QM9


@core.Registry.register("GeoLDMTask")
class GeoLDMTask(GeomMolecularGenerative):
    """GeoLDM's `EnLatentDiffusion` wrapped in the platform's `Task` contract."""

    def sample(
        self,
        nodesxsample: torch.Tensor = torch.tensor([10]),
        context: Optional[torch.Tensor] = None,
        condition_tensor: Optional[torch.Tensor] = None,
        condition_mode: Optional[str] = None,
        fix_noise: bool = False,
        n_frames: int = 0,
        n_retrys: int = 0,
        t_retry: int = 180,
        mode: str = "ddpm",
        use_noised_conditioning: bool = False,
        **kwargs,
    ):
        """Sample molecular structures from vanilla GeoLDM's 2-stage sampler.

        `condition_tensor`/`condition_mode`/`n_retrys`/`t_retry`/
        `use_noised_conditioning` have no vanilla-GeoLDM equivalent and raise
        `NotImplementedError` if set to a non-default value. `n_frames` is
        accepted and ignored (no trajectory export -- see plan's "explicitly
        out of scope"). `mode` must be "ddpm" (vanilla GeoLDM has no DDIM
        sampler).
        """
        if condition_tensor is not None:
            raise NotImplementedError(
                "GeoLDMTask.sample() has no property-conditioning support this pass "
                "(condition_tensor must be None)."
            )
        if condition_mode is not None:
            raise NotImplementedError(
                "GeoLDMTask.sample() has no condition_mode support this pass."
            )
        if n_retrys:
            raise NotImplementedError(
                "GeoLDMTask.sample() has no retry-loop support this pass (n_retrys must be 0)."
            )
        if use_noised_conditioning:
            raise NotImplementedError(
                "GeoLDMTask.sample() has no noised-conditioning support this pass."
            )
        if mode != "ddpm":
            raise NotImplementedError(
                f"GeoLDMTask.sample() only supports mode='ddpm' (vanilla GeoLDM has no "
                f"DDIM sampler), got mode={mode!r}."
            )
        del n_frames, t_retry, kwargs

        batch_size = nodesxsample.size(0)
        nnode = int(torch.max(nodesxsample).item())
        node_mask = torch.zeros(batch_size, nnode)

        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * nnode * nnode, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)

        x, h = self.model.sample(batch_size, nnode, node_mask, edge_mask, context, fix_noise=fix_noise)

        one_hot = h.get("categorical", torch.zeros_like(x))
        charges = h["integer"]

        return one_hot, charges, x, node_mask


class GeoLDMTaskFactory:
    """Factory matching `train.py`'s `task_module.build()` /
    `task_module.task` instantiation pattern. Loads a frozen VAE checkpoint
    (trained via `GeoLDMVAETaskFactory`/`configs/tasks/vae_geoldm.yaml`) the
    same way `diffusion_ldm.py::LDMTaskFactory.build()` loads its
    `autoencoder_ckpt` -- see plan's Core-Change-Requirement note."""

    def __init__(
        self,
        task_type: str,
        vae_ckpt: str,
        vae_config: Optional[Dict] = None,
        n_dims: int = 3,
        hidden_nf: int = 64,
        n_layers: int = 4,
        attention: bool = False,
        tanh: bool = False,
        model: str = "egnn_dynamics",
        norm_constant: float = 0,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
        condition_time: bool = True,
        include_charges: bool = True,
        diffusion_steps: int = 1000,
        diffusion_noise_schedule: str = "polynomial_2",
        diffusion_noise_precision: float = 1e-5,
        diffusion_loss_type: str = "l2",
        normalize_factors: tuple = (1.0, 4.0, 10.0),
        trainable_ae: bool = False,
        augment_noise: float = 0.0,
        data_augmentation: bool = False,
        atom_vocab: Optional[List] = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.vae_ckpt = vae_ckpt
        self.vae_config = dict(vae_config) if vae_config else None
        self.n_dims = n_dims
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.attention = attention
        self.tanh = tanh
        self.model_mode = model
        self.norm_constant = norm_constant
        self.inv_sublayers = inv_sublayers
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.condition_time = condition_time
        self.include_charges = include_charges
        self.diffusion_steps = diffusion_steps
        self.diffusion_noise_schedule = diffusion_noise_schedule
        self.diffusion_noise_precision = diffusion_noise_precision
        self.diffusion_loss_type = diffusion_loss_type
        self.normalize_factors = tuple(normalize_factors)
        self.trainable_ae = trainable_ae
        self.augment_noise = augment_noise
        self.data_augmentation = data_augmentation
        self.atom_vocab = atom_vocab or kwargs.get("atom_vocab")
        self.kwargs = kwargs

    def _load_frozen_vae(self):
        """Load the frozen VAE from `vae_ckpt`, inferring its architecture
        from the checkpoint itself rather than requiring the caller to
        hand-duplicate `vae_config` (error-prone -- a mismatch silently
        produces wrong-shaped weights or a confusing size-mismatch error).

        The two engines this platform supports store checkpoints
        differently, and conveniently use different file extensions too
        (`.pkl` for the legacy `original` engine, `.ckpt` for `lightning`):
          - `original` engine (`core/engine.py::Engine.save()`): pickles the
            fully-built `GeoLDMVAETask` instance directly under
            `checkpoint["hyperparameters"]["task"]["diffusion_model"]` -- no
            architecture needs to be rebuilt at all, just use it as-is.
          - `lightning` engine: stores only tensors under `state_dict`, but
            also stores the exact factory kwargs used to build the task
            under `checkpoint["hyper_parameters"]["model_config"]` --
            rebuild the architecture from that, then load `state_dict`.

        `self.vae_config` (if the caller supplied one) is only used as a
        last-resort fallback, for checkpoints predating this inference
        (or any other format this doesn't recognize).
        """
        checkpoint = torch.load(self.vae_ckpt, map_location="cpu", weights_only=False)

        task_hp = checkpoint.get("hyperparameters", {}).get("task")
        if isinstance(task_hp, dict) and isinstance(task_hp.get("diffusion_model"), torch.nn.Module):
            vae = task_hp["diffusion_model"]
            vae.eval()
            vae.requires_grad_(False)
            return vae

        model_config = checkpoint.get("hyper_parameters", {}).get("model_config")
        if model_config:
            vae_config = dict(model_config)
        elif self.vae_config:
            vae_config = dict(self.vae_config)
        else:
            raise ValueError(
                f"Could not infer the VAE's architecture from checkpoint "
                f"{self.vae_ckpt!r} (unrecognized format, no "
                f"hyperparameters.task.diffusion_model or "
                f"hyper_parameters.model_config found) and no vae_config "
                f"override was provided as a fallback."
            )
        for key in ("_target_", "atom_vocab", "train_set", "valid_set", "test_set"):
            vae_config.pop(key, None)
        vae_config.setdefault("task_type", "vae_geoldm")
        vae_factory = GeoLDMVAETaskFactory(atom_vocab=self.atom_vocab, **vae_config)
        vae_task = vae_factory.build()

        # Legacy Engine checkpoints store the task's state dict under "model";
        # Lightning checkpoints use "state_dict" with a "task." prefix.
        if "state_dict" in checkpoint:
            raw_state = checkpoint["state_dict"]
            raw_state = {
                (k[len("task.") :] if k.startswith("task.") else k): v for k, v in raw_state.items()
            }
        else:
            raw_state = checkpoint.get("model", checkpoint)

        missing, unexpected = vae_task.load_state_dict(raw_state, strict=False)
        if missing or unexpected:
            print(
                f"[GeoLDMTaskFactory] Loaded VAE checkpoint with "
                f"{len(missing)} missing / {len(unexpected)} unexpected keys."
            )

        vae_task.eval()
        vae_task.requires_grad_(False)
        return vae_task.model  # the ported EnHierarchicalVAE instance

    def build(self) -> "GeoLDMTask":
        if not self.atom_vocab:
            raise ValueError(
                "GeoLDMTaskFactory requires atom_vocab (injected from data.atom_vocab "
                "by cli/train.py) to size the frozen VAE's atom vocabulary."
            )

        vae = self._load_frozen_vae()
        in_node_nf = vae.latent_node_nf

        dynamics_in_node_nf = in_node_nf + 1 if self.condition_time else in_node_nf

        net_dynamics = EGNN_dynamics_QM9(
            in_node_nf=dynamics_in_node_nf,
            context_node_nf=0,
            n_dims=self.n_dims,
            hidden_nf=self.hidden_nf,
            act_fn=torch.nn.SiLU(),
            n_layers=self.n_layers,
            attention=self.attention,
            tanh=self.tanh,
            condition_time=self.condition_time,
            mode=self.model_mode,
            norm_constant=self.norm_constant,
            inv_sublayers=self.inv_sublayers,
            sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )

        diffusion_model = EnLatentDiffusion(
            vae=vae,
            trainable_ae=self.trainable_ae,
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=self.n_dims,
            timesteps=self.diffusion_steps,
            noise_schedule=self.diffusion_noise_schedule,
            noise_precision=self.diffusion_noise_precision,
            loss_type=self.diffusion_loss_type,
            norm_values=self.normalize_factors,
            include_charges=self.include_charges,
        )

        self.task = GeoLDMTask(
            diffusion_model=diffusion_model,
            n_node_dist={},
            augment_noise=self.augment_noise,
            data_augmentation=self.data_augmentation,
        )
        self.task.task_type = self.task_type
        self.task.atom_vocab = self.atom_vocab
        return self.task
