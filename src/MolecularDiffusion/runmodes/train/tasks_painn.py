"""Factory wiring the ported PaiNN/OM-Diff ``EquivNet`` backbone into the
existing, unmodified diffusion task -- structured exactly like
``runmodes/train/tasks_gfmdiff.py::ModelTaskFactory`` (a new dynamics
network dropped into ``EnVariationalDiffusion`` +
``GeomMolecularGenerative``, zero edits to either).

Per the approved integration plan
(``docs/model_integrations/omdiff/INTEGRATION_PLAN.md``) this is a §1
backbone swap: OM-Diff's plain-L2 loss, metal-centre masking, conditional
size prior and regressor guidance are out of scope -- only the network is
ported, so it is named after its architecture family (``painn``) rather
than after the paper.
"""

import logging

import torch

from MolecularDiffusion.modules.models.en_diffusion import (
    EnVariationalDiffusion,
)
from MolecularDiffusion.modules.models.painn_dynamics import PaiNNDynamics
from MolecularDiffusion.modules.tasks.diffusion import (
    GeomMolecularGenerative,
)

logger = logging.getLogger(__name__)

# Shared verbatim with LDMTaskFactory (ADiT): neither has an adapter route.
ADAPTER_UNSUPPORTED_MSG = (
    "adapter_conditions is not supported by this model (no adapter path); "
    "use condition_names only."
)


class ModelTaskFactory:
    """Build the PaiNN-backbone diffusion model + task.

    Parameters:
        task_type (str): must be ``"diffusion"``.
        train_set: unused; kept for interface parity with the other
            factories (the atom-count histogram is computed lazily by
            ``GeomMolecularGenerative.preprocess``).
        atom_vocab (list): atom vocabulary used for encoding.
        task_names (list): conditional labels (context columns).
        condition_names (list): condition names for conditional generation.
        num_interactions (int): PaiNN interaction/update blocks.
        hidden_dims (dict): keys ``node_size``, ``edge_size``,
            ``embedding_dim`` for the backbone widths.
        chkpt_path (str): optional path to a model checkpoint.
        **kwargs: diffusion + backbone keyword arguments, see
            ``configs/tasks/diffusion_painn.yaml``.
    """

    def __init__(
        self,
        task_type: str,
        train_set=None,
        atom_vocab=None,
        task_names: list = [],
        condition_names: list = [],
        num_interactions: int = 5,
        hidden_dims: dict = {},
        chkpt_path: str = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.train_set = train_set
        self.atom_vocab = atom_vocab
        self.task_names = task_names
        self.condition_names = condition_names
        self.num_interactions = num_interactions
        self.hidden_dims = hidden_dims

        n_dim_extra = len(kwargs.get("extra_norm_values", []))
        self.in_node_nf = len(atom_vocab) + n_dim_extra + 1  # +1 atomic num
        self.context_node_nf = len(self.task_names)

        self.chkpt_path = chkpt_path
        self.kwargs = kwargs
        # Uniform conditioning signature (same 5 keys/defaults in every model's
        # factory). These fallbacks only matter when condition_names is
        # non-empty: unconditional runs never read mask/normalisation values.
        if kwargs.get("adapter_conditions"):
            raise ValueError(ADAPTER_UNSUPPORTED_MSG)
        # Hydra ignores unknown keys silently: make the effective values visible.
        names = list(condition_names or task_names or [])
        print(
            f"[PaiNNTaskFactory] condition_names={names} "
            f"context_mask_rate={kwargs.get('context_mask_rate', 0.2)} "
            f"mask_value={kwargs.get('mask_value', 5)} "
            f"normalize_condition={kwargs.get('normalize_condition', 'value_10')}"
            + ("" if names else " (unconditional)")
        )

    def build(self):
        """Build and return the ``GeomMolecularGenerative`` task."""
        is_main_process = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        if self.task_type != "diffusion":
            raise ValueError(
                f"Unknown task_type '{self.task_type}' for the PaiNN "
                "ModelTaskFactory. Only 'diffusion' is supported."
            )

        dynamics_model = PaiNNDynamics(
            in_node_nf=self.in_node_nf,
            context_node_nf=self.context_node_nf,
            n_dims=3,
            num_interactions=self.num_interactions,
            node_size=self.hidden_dims.get("node_size", 256),
            edge_size=self.hidden_dims.get("edge_size", 64),
            embedding_dim=self.hidden_dims.get("embedding_dim", 256),
            rbf_features=self.kwargs.get("rbf_features", 64),
            rbf_max_distance=self.kwargs.get("rbf_max_distance", 5.0),
            time_features=self.kwargs.get("time_features", 16),
            cutoff=self.kwargs.get("cutoff", None),
            envelop_p=self.kwargs.get("envelop_p", 6),
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
            include_charges=True,
            extra_norm_values=self.kwargs.get("extra_norm_values", []),
            context_mask_rate=self.kwargs.get("context_mask_rate", 0.2),
            mask_value=self.kwargs.get("mask_value", 5),
        )

        self.task = GeomMolecularGenerative(
            model,
            augment_noise=self.kwargs.get("augment_noise", False),
            data_augmentation=self.kwargs.get("data_augmentation", False),
            condition=self.task_names,
            sp_regularizer=None,
            normalize_condition=self.kwargs.get("normalize_condition", "value_10"),
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
                if ckpt_task_type is not None and (
                    ckpt_task_type != self.task_type
                ):
                    raise ValueError(
                        f"Task type mismatch: checkpoint was trained as "
                        f"'{ckpt_task_type}' but current config specifies "
                        f"'{self.task_type}'. Update your config to use "
                        f"tasks: {ckpt_task_type} or point to the correct "
                        f"checkpoint."
                    )

                chk_point = ckpt.get("ema_model") or ckpt.get("model")
                if chk_point is None:
                    raise KeyError(
                        "Checkpoint missing both 'ema_model' and 'model'"
                    )

                if is_main_process:
                    logger.info(f"Loading checkpoint from {self.chkpt_path}")

                load_result = self.task.load_state_dict(
                    chk_point, strict=False
                )
                if is_main_process and (
                    load_result.missing_keys or load_result.unexpected_keys
                ):
                    logger.warning(
                        "\033[93mCheckpoint loaded with mismatched keys.\033[0m"
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
