"""Factory wiring the ported GCDM/GCPNet backbone into the existing,
unmodified diffusion task.

Structured like ``runmodes/train/tasks_gfmdiff.py`` -- a new dynamics network
dropped into ``EnVariationalDiffusion`` + ``GeomMolecularGenerative``, with
zero edits to either.

Per the approved integration plan (revision 2) this is a §1 backbone swap:
GCDM's own ~1800-line ``variational_diffusion.py`` is a line-for-line EDM
re-implementation of maths the platform already has, so only
``GCPNetDynamics`` is ported.

The one thing this factory does that its four siblings
(``tasks_{egcl,egt,gfmdiff,painn}.py``) do not is read ``include_charges``
from config instead of hardcoding ``True``: GCDM's GEOM checkpoint and its six
per-property conditional QM9 checkpoints were all trained with
``include_charges: false`` (atom one-hot only, no atomic-number scalar).
Defaults to ``True``, so nothing about the other backbones changes.
"""

import logging

import torch

from MolecularDiffusion.core.checkpoint import engine_weights
from MolecularDiffusion.modules.models.edm.en_diffusion import EnVariationalDiffusion
from MolecularDiffusion.modules.models.gcdm.dynamics import GCDMDynamics
from MolecularDiffusion.modules.models.gcdm.gcp_layers import (
    GCPLayerConfig,
    GCPModuleConfig,
)
from MolecularDiffusion.modules.tasks.diffusion import GeomMolecularGenerative

logger = logging.getLogger(__name__)


class ModelTaskFactory:
    """Build the GCDM-backbone diffusion model + task.

    Parameters:
        task_type (str): must be ``"diffusion"``.
        train_set: unused; kept for interface parity with the other
            factories (GCPNet needs no dataset statistics at build time).
        atom_vocab (list): atom vocabulary used for one-hot encoding.
        task_names (list): conditional labels (context columns).
        condition_names (list): condition names for conditional generation.
        include_charges (bool): whether the diffusion latent carries the
            atomic-number scalar channel. ``True`` matches GCDM's
            unconditional QM9 preset; ``False`` matches its GEOM and
            conditional-QM9 presets.
        num_encoder_layers (int): number of ``GCPInteractions`` blocks.
        hidden_dims (dict): ``h_hidden_dim`` / ``chi_hidden_dim`` /
            ``e_hidden_dim`` / ``xi_hidden_dim``.
        module_cfg (dict): overrides for :class:`GCPModuleConfig`.
        layer_cfg (dict): overrides for :class:`GCPLayerConfig`.
        chkpt_path (str): optional path to a model checkpoint.
        **kwargs: diffusion keyword arguments (see ``diffusion_gcdm.yaml``).
    """

    def __init__(
        self,
        task_type: str,
        train_set=None,
        atom_vocab=None,
        task_names: list = [],
        condition_names: list = [],
        include_charges: bool = True,
        num_encoder_layers: int = 9,
        hidden_dims: dict = {},
        module_cfg: dict = {},
        layer_cfg: dict = {},
        chkpt_path: str = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.train_set = train_set
        self.atom_vocab = atom_vocab
        self.task_names = task_names
        self.condition_names = condition_names
        self.include_charges = bool(include_charges)
        self.num_encoder_layers = num_encoder_layers
        self.hidden_dims = dict(hidden_dims or {})
        self.module_cfg = dict(module_cfg or {})
        self.layer_cfg = dict(layer_cfg or {})

        n_dim_extra = len(kwargs.get("extra_norm_values", []))
        self.in_node_nf = (
            len(atom_vocab) + n_dim_extra + int(self.include_charges)
        )
        self.context_node_nf = len(self.task_names)

        self.chkpt_path = chkpt_path
        self.kwargs = kwargs

    def build(self):
        """Build and return the ``GeomMolecularGenerative`` task."""
        is_main_process = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        if self.task_type != "diffusion":
            raise ValueError(
                f"Unknown task_type '{self.task_type}' for GCDM "
                "ModelTaskFactory. Only 'diffusion' is supported."
            )

        dynamics_model = GCDMDynamics(
            in_node_nf=self.in_node_nf,
            context_node_nf=self.context_node_nf,
            n_dims=3,
            num_encoder_layers=self.num_encoder_layers,
            h_hidden_dim=self.hidden_dims.get("h_hidden_dim", 256),
            chi_hidden_dim=self.hidden_dims.get("chi_hidden_dim", 32),
            e_hidden_dim=self.hidden_dims.get("e_hidden_dim", 64),
            xi_hidden_dim=self.hidden_dims.get("xi_hidden_dim", 16),
            chi_input_dim=self.hidden_dims.get("chi_input_dim", 2),
            e_input_dim=self.hidden_dims.get("e_input_dim", 1),
            xi_input_dim=self.hidden_dims.get("xi_input_dim", 1),
            dropout=self.kwargs.get("dropout", 0.0),
            condition_on_time=self.kwargs.get("condition_on_time", True),
            self_condition=self.kwargs.get("self_condition", False),
            module_cfg=GCPModuleConfig(**self.module_cfg),
            layer_cfg=GCPLayerConfig(**self.layer_cfg),
        )

        model = EnVariationalDiffusion(
            dynamics=dynamics_model,
            in_node_nf=self.in_node_nf,
            n_dims=3,
            timesteps=self.kwargs["diffusion_steps"],
            noise_schedule=self.kwargs.get(
                "diffusion_noise_schedule", "polynomial_2"
            ),
            noise_precision=self.kwargs.get("diffusion_noise_precision", 1e-5),
            loss_type=self.kwargs.get("diffusion_loss_type", "l2"),
            norm_values=self.kwargs.get("normalize_factors", [1, 4, 10]),
            include_charges=self.include_charges,
            extra_norm_values=self.kwargs.get("extra_norm_values", []),
            context_mask_rate=self.kwargs.get("context_mask_rate", 0.15),
            mask_value=self.kwargs.get("mask_value", None),
            # needed by the include_charges=False decode path: with no
            # atomic-number channel, element identity is read back off the
            # one-hot through this vocabulary.
            atom_vocab=self.atom_vocab,
        )

        self.task = GeomMolecularGenerative(
            model,
            augment_noise=self.kwargs.get("augment_noise", False),
            data_augmentation=self.kwargs.get("data_augmentation", False),
            condition=self.task_names,
            sp_regularizer=None,
            normalize_condition=self.kwargs.get("normalize_condition", None),
            reference_indices=self.kwargs.get("reference_indices", None),
        )

        n_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        if is_main_process:
            logger.info(f"Number of parameters: {n_params}")

        if self.chkpt_path:
            try:
                ckpt = torch.load(self.chkpt_path, weights_only=False)

                ckpt_task_type = ckpt.get("hyperparameters", {}).get(
                    "task_type"
                ) or ckpt.get("task_type")
                if (
                    ckpt_task_type is not None
                    and ckpt_task_type != self.task_type
                ):
                    raise ValueError(
                        f"Task type mismatch: checkpoint was trained as "
                        f"'{ckpt_task_type}' but current config specifies "
                        f"'{self.task_type}'. Update your config to use "
                        f"tasks: {ckpt_task_type} or point to the correct "
                        f"checkpoint."
                    )

                chk_point = engine_weights(ckpt)
                if chk_point is None:
                    raise KeyError(
                        "Checkpoint missing both 'ema_model' and 'model'"
                    )

                if is_main_process:
                    logger.info(
                        f"Loading checkpoint from {self.chkpt_path}"
                    )

                load_result = self.task.load_state_dict(
                    chk_point, strict=False
                )
                if is_main_process and (
                    load_result.missing_keys or load_result.unexpected_keys
                ):
                    logger.warning(
                        "\033[93mCheckpoint loaded with mismatched keys."
                        "\033[0m"
                    )
                    if load_result.missing_keys:
                        logger.warning(
                            f"\033[93mMissing keys "
                            f"({len(load_result.missing_keys)}): "
                            f"{load_result.missing_keys}\033[0m"
                        )
                    if load_result.unexpected_keys:
                        logger.warning(
                            f"\033[93mUnexpected keys "
                            f"({len(load_result.unexpected_keys)}): "
                            f"{load_result.unexpected_keys}\033[0m"
                        )

                if "mean" in chk_point and "std" in chk_point:
                    self.task.mean = chk_point["mean"]
                    self.task.std = chk_point["std"]
            except FileNotFoundError:
                if is_main_process:
                    logger.warning(
                        f"Checkpoint not found at {self.chkpt_path}. "
                        "Initializing model without loading."
                    )
                raise FileNotFoundError(
                    f"Checkpoint not found at {self.chkpt_path}."
                )

        self.task.atom_vocab = self.atom_vocab
        self.task.task_type = self.task_type

        return self.task
