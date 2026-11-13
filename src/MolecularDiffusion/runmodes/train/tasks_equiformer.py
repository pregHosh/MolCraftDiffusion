import torch
import logging

from MolecularDiffusion.modules.tasks import SSL3D, ProperyPrediction
from MolecularDiffusion.modules.tasks.diffusion import GeomMolecularGenerative
from MolecularDiffusion.runmodes.train.tasks_egcl import _build_ssl3d_objectives
from MolecularDiffusion.modules.models.edm.en_diffusion import (
    EnVariationalDiffusion,
)
from MolecularDiffusion.modules.models.shepherd_arch.equiformer_v2_encoder import EquiformerV2
from MolecularDiffusion.modules.models.dynamics.equiformer_v2_dynamics import EquiformerV2_dynamics
from MolecularDiffusion.modules.models.dynamics.equiformer_v2_backbone import EquiformerV2Backbone

logger = logging.getLogger(__name__)


class ModelTaskFactory:
    """
    Factory to construct EquiformerV2-based models and task handlers.

    Supported task_type values:
        - "ssl3d_equiformer": SSL3D pretraining with an EquiformerV2 backbone.
        - "regression": Scalar property prediction with an EquiformerV2 backbone.
        - "diffusion": EnVariationalDiffusion generative training with
          EquiformerV2_dynamics as the denoising network -- a backbone swap,
          same diffusion math as every other diffusion_*.yaml (see
          diffusion_painn.yaml / tasks_painn.py for the identical pattern).

    Args:
        task_type (str): One of the supported task types above.
        atom_vocab (list[str]): Atom vocabulary used for feature encoding.
        condition_names (list[str]): Names of conditioning properties.
        chkpt_path (str | None): Optional checkpoint path.

        EquiformerV2 architecture kwargs:
            sphere_channels (int): Node embedding channels (default 128).
            input_sphere_channels (int): Must equal sphere_channels (default 128).
            num_layers (int): Number of transformer layers (default 8).
            lmax_list (list[int]): Max spherical harmonic degree per resolution (default [6]).
            mmax_list (list[int]): Max order per resolution (default [2]).
            grid_resolution (int | None): SO3 grid resolution (default None -> auto).
            num_sphere_samples (int): Sphere-sampling points for the S2 grid (default 128).
            attn_hidden_channels (int): Attention hidden size (default 128).
            attn_alpha_channels (int): Alpha vector channels per head (default 32).
            attn_value_channels (int): Value vector channels per head (default 16).
            ffn_hidden_channels (int): FFN hidden size (default 512).
            num_heads (int): Number of attention heads (default 8).
            norm_type (str): Normalisation type (default "rms_norm_sh").
            edge_channels (int): Edge invariant feature channels (default 128).
            use_atom_edge_embedding (bool): Use atom embedding in edge features (default True).
            share_atom_edge_embedding (bool): Share atom edge embedding (default False).
            use_m_share_rad (bool): Share the radial function across m (default False).
            distance_function (str): Radial basis function (default "gaussian").
            num_distance_basis (int): Number of basis functions (default 512).
            attn_activation (str): Attention activation (default "scaled_silu").
            use_s2_act_attn (bool): Use S2 activation in attention (default False).
            use_attn_renorm (bool): Renormalise attention weights (default True).
            ffn_activation (str): FFN activation (default "scaled_silu").
            use_gate_act (bool): Use gate activation (default False).
            use_grid_mlp (bool): Use grid MLP in FFN (default False).
            use_sep_s2_act (bool): Use separable S2 activation (default True).
            alpha_drop (float): Attention dropout (default 0.0).
            drop_path_rate (float): Drop path rate (default 0.0).
            proj_drop (float): Projection dropout (default 0.0).
            cutoff (float): Radial cutoff in Angstrom (default 5.0).
            weight_init (str): Weight initialisation scheme (default "normal").

        SSL3D kwargs:
            ssl3d_denoise_weight, ssl3d_sigma_min, ssl3d_sigma_max, ssl3d_sigma_schedule,
            ssl3d_mtype_weight, ssl3d_mask_rate, ssl3d_atom_vocab_size,
            ssl3d_dist_weight, ssl3d_dist_k_pairs

        Diffusion kwargs (see configs/tasks/diffusion_equiformer.yaml):
            diffusion_steps, diffusion_noise_schedule,
            diffusion_noise_precision, diffusion_loss_type,
            normalize_factors, extra_norm_values,
            augment_noise, data_augmentation, context_mask_rate, mask_value,
            normalize_condition, reference_indices

        Regression kwargs:
            task_learn (list[str]): Property names to predict.
            criterion (str): Loss function (default "mse").
            metric (list[str]): Evaluation metric(s) (default ["mae"]).
            num_mlp_layer (int): Number of MLP head layers (default 2).
            mlp_dropout (float): MLP head dropout (default 0.0).
            mlp_hidden_dim (int | None): MLP head hidden dim (default: backbone d_model).
            mlp_batch_norm (str | None): None, "layernorm", or "batchnorm".
            target_normalization (bool): Normalize targets by mean/std (default True).
            prediction_mlp_type (str): "pernode" or "padded" (default "pernode").
            prediction_activation (str): "relu" or "silu" (default "relu").

        Context routing kwargs:
            adapter_conditions (list[str]): condition_names routed through adapter MLPs.
            use_adapter_module (bool): DEPRECATED — routes all conditions through adapter.
    """

    def __init__(
        self,
        task_type: str,
        atom_vocab: list,
        condition_names: list = [],
        chkpt_path: str = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.atom_vocab = atom_vocab
        self.condition_names = condition_names
        self.chkpt_path = chkpt_path
        self.kwargs = kwargs

        # ── Feature dimensions ────────────────────────────────────────────────
        n_dim_extra = len(kwargs.get("extra_norm_values", []))
        self.in_node_nf = len(atom_vocab) + n_dim_extra + 1  # +1 for atomic number

        # Account for unknown atom category if fallback is enabled
        # during training allow_unknown appends Suisei to the vocab, so we don't need to add it twice
        if kwargs.get("use_unknown_fallback", False) and "Suisei" not in atom_vocab:
            self.in_node_nf += 1

        self.context_node_nf = len(self.condition_names)

        # ── Context routing ───────────────────────────────────────────────────
        adapter_conditions = self.kwargs.get("adapter_conditions", None)
        use_adapter_module = self.kwargs.get("use_adapter_module", False)

        if adapter_conditions:
            for ac in adapter_conditions:
                if ac not in self.condition_names:
                    raise ValueError(
                        f"adapter_conditions entry '{ac}' not found in "
                        f"condition_names {self.condition_names}"
                    )
            self.adapter_indices = [self.condition_names.index(ac) for ac in adapter_conditions]
            self.concat_indices = [
                i for i in range(len(self.condition_names)) if i not in self.adapter_indices
            ]
        elif use_adapter_module:
            self.adapter_indices = list(range(len(self.condition_names)))
            self.concat_indices = []
        else:
            self.adapter_indices = []
            self.concat_indices = list(range(len(self.condition_names)))

        self.n_adapter_context = len(self.adapter_indices)
        self.n_concat_context = len(self.concat_indices)

        # task is built lazily in build()
        self.task = None

    def build(self):
        """Build and return the task. Called by the training harness."""
        if self.task_type == "ssl3d_equiformer":
            self.task = self._build_ssl3d()
        elif self.task_type == "regression":
            self.task = self._build_regression()
        elif self.task_type == "diffusion":
            self.task = self._build_diffusion()
        else:
            raise ValueError(
                f"Unknown task_type '{self.task_type}'. "
                f"Expected 'ssl3d_equiformer', 'regression' or 'diffusion'."
            )

        if self.chkpt_path is not None:
            self._load_checkpoint(self.chkpt_path)

        return self.task

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_equiformer(self) -> EquiformerV2:
        k = self.kwargs
        return EquiformerV2(
            num_layers=k.get("num_layers", 8),
            input_sphere_channels=k.get("input_sphere_channels", 128),
            sphere_channels=k.get("sphere_channels", 128),
            attn_hidden_channels=k.get("attn_hidden_channels", 128),
            num_heads=k.get("num_heads", 8),
            attn_alpha_channels=k.get("attn_alpha_channels", 32),
            attn_value_channels=k.get("attn_value_channels", 16),
            ffn_hidden_channels=k.get("ffn_hidden_channels", 512),
            norm_type=k.get("norm_type", "rms_norm_sh"),
            lmax_list=k.get("lmax_list", [6]),
            mmax_list=k.get("mmax_list", [2]),
            grid_resolution=k.get("grid_resolution", None),
            num_sphere_samples=k.get("num_sphere_samples", 128),
            edge_channels=k.get("edge_channels", 128),
            use_atom_edge_embedding=k.get("use_atom_edge_embedding", True),
            share_atom_edge_embedding=k.get("share_atom_edge_embedding", False),
            use_m_share_rad=k.get("use_m_share_rad", False),
            distance_function=k.get("distance_function", "gaussian"),
            num_distance_basis=k.get("num_distance_basis", 512),
            attn_activation=k.get("attn_activation", "scaled_silu"),
            use_s2_act_attn=k.get("use_s2_act_attn", False),
            use_attn_renorm=k.get("use_attn_renorm", True),
            ffn_activation=k.get("ffn_activation", "scaled_silu"),
            use_gate_act=k.get("use_gate_act", False),
            use_grid_mlp=k.get("use_grid_mlp", False),
            use_sep_s2_act=k.get("use_sep_s2_act", True),
            alpha_drop=k.get("alpha_drop", 0.0),
            drop_path_rate=k.get("drop_path_rate", 0.0),
            proj_drop=k.get("proj_drop", 0.0),
            cutoff=k.get("cutoff", 5.0),
            weight_init=k.get("weight_init", "normal"),
        )

    def _build_dynamics(self, in_node_nf_override: int = None) -> EquiformerV2_dynamics:
        k = self.kwargs
        in_node_nf = in_node_nf_override if in_node_nf_override is not None else self.in_node_nf
        return EquiformerV2_dynamics(
            equiformer=self._build_equiformer(),
            in_node_nf=in_node_nf,
            n_dims=3,
            condition_time=True,
            context_node_nf=self.context_node_nf,
            adapter_indices=self.adapter_indices if self.n_adapter_context > 0 else None,
            concat_indices=self.concat_indices if self.n_concat_context > 0 else None,
            sphere_channels=k.get("sphere_channels", 128),
            lmax_list=k.get("lmax_list", [6]),
        )

    def _build_regression(self):
        k = self.kwargs
        model = EquiformerV2Backbone(
            equiformer=self._build_equiformer(),
            in_node_channels=self.in_node_nf,
            sphere_channels=k.get("sphere_channels", 128),
            lmax_list=k.get("lmax_list", [6]),
        )
        model.hidden_nf = model.d_model
        return ProperyPrediction(
            model,
            task=k.get("task_learn", ""),
            include_charge=True,
            criterion=k.get("criterion", "mse"),
            metric=k.get("metric", "mae"),
            num_mlp_layer=k.get("num_mlp_layer", 2),
            mlp_batch_norm=k.get("mlp_batch_norm", None),
            mlp_dropout=k.get("mlp_dropout", 0.0),
            mlp_hidden_dim=k.get("mlp_hidden_dim", None),
            readout=k.get("aggregation_method", "sum"),
            normalization=k.get("target_normalization", True),
            num_class=len(k.get("task_learn", "")),
            prediction_mlp_type=k.get("prediction_mlp_type", "pernode"),
            prediction_activation=k.get("prediction_activation", "relu"),
        )

    def _build_diffusion(self):
        k = self.kwargs
        model = EnVariationalDiffusion(
            dynamics=self._build_dynamics(),
            in_node_nf=self.in_node_nf,
            n_dims=3,
            timesteps=k["diffusion_steps"],
            noise_schedule=k.get("diffusion_noise_schedule", "polynomial_2"),
            noise_precision=k.get("diffusion_noise_precision", 1e-5),
            loss_type=k.get("diffusion_loss_type", "l2"),
            norm_values=k.get("normalize_factors", [1, 4, 10]),
            include_charges=True,
            extra_norm_values=k.get("extra_norm_values", []),
            context_mask_rate=k.get("context_mask_rate", 0.15),
            mask_value=k.get("mask_value", None),
        )
        task = GeomMolecularGenerative(
            model,
            augment_noise=k.get("augment_noise", False),
            data_augmentation=k.get("data_augmentation", False),
            condition=self.condition_names,
            sp_regularizer=None,
            normalize_condition=k.get("normalize_condition", None),
            reference_indices=k.get("reference_indices", None),
        )
        task.atom_vocab = self.atom_vocab
        task.task_type = self.task_type
        return task

    def _build_ssl3d(self):
        # Equiformer handles time internally via condition_time=True;
        # SSL3D passes sigma as the "t" input rather than embedding in node feats.
        dynamics = self._build_dynamics()
        # hidden_nf = in_node_nf (head_h maps sphere_channels → in_node_nf)
        dynamics.hidden_nf = self.in_node_nf
        objectives = _build_ssl3d_objectives(self.kwargs)
        return SSL3D(dynamics, objectives, include_charge=True, t_embedding="sinusoidal")

    def _load_checkpoint(self, path: str):
        import os
        if not os.path.exists(path):
            logger.warning(f"Checkpoint path not found, skipping: {path}")
            return
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        self.task.model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint from {path}")
