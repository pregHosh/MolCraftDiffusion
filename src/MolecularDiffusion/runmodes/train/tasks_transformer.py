"""Factory wiring ``TransformerDynamics`` -- a plain (non-equivariant)
multi-head self-attention backbone -- into the existing, unmodified
diffusion task, structured exactly like
``runmodes/train/tasks_painn.py::ModelTaskFactory`` (a new dynamics
network dropped into ``EnVariationalDiffusion`` + ``GeomMolecularGenerative``,
zero edits to either).

Per the approved ledger
(``docs/model_novel/diffusion_transformer/INTEGRATION_PLAN.md``) this is a
novel-model ablation against the EGCL parent (``tasks_egcl.py``): only the
backbone (and the platform's own ``data_augmentation`` flag) differ --
every diffusion-math hyperparameter is copied verbatim from
``configs/tasks/diffusion.yaml`` so the ablation isolates the backbone
swap and nothing else. Follows the *parent's* simpler constructor
convention (a plain ``condition_names`` parameter) rather than
``tasks_painn.ModelTaskFactory``'s ``train_set``/``task_names``
legacy-alias pair, which this backbone does not need.
"""

import logging

import torch

from MolecularDiffusion.modules.models.en_diffusion import (
    EnVariationalDiffusion,
)
from MolecularDiffusion.modules.models.transformer_dynamics import (
    TransformerDynamics,
)
from MolecularDiffusion.modules.tasks.diffusion import (
    GeomMolecularGenerative,
)

logger = logging.getLogger(__name__)


class ModelTaskFactory:
    """Build the Transformer-backbone diffusion model + task.

    Parameters:
        task_type (str): must be ``"diffusion"``.
        atom_vocab (list): atom vocabulary used for encoding.
        condition_names (list): condition names for conditional
            generation.
        hidden_dim (int): transformer token width.
        num_layers (int): transformer block depth
            (``Transformer``'s ``depth``).
        num_heads (int): multi-head self-attention heads.
        mlp_dim (int): feed-forward hidden width; ``None`` defaults to
            ``4 * hidden_dim`` inside ``Transformer``.
        dropout (float): dropout probability.
        activation_type (str): ``Transformer``'s feed-forward activation
            string knob (e.g. ``"gelu"``).
        add_sinusoid_posenc (bool): ablation-only knob, off by default.
        chkpt_path (str): optional path to a model checkpoint.
        **kwargs: diffusion keyword arguments, see
            ``configs/tasks/diffusion_transformer.yaml``.
    """

    def __init__(
        self,
        task_type: str,
        atom_vocab: list = None,
        condition_names: list = [],
        hidden_dim: int = 192,
        num_layers: int = 9,
        num_heads: int = 8,
        mlp_dim: int = None,
        dropout: float = 0.0,
        activation_type: str = "gelu",
        add_sinusoid_posenc: bool = False,
        chkpt_path: str = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.atom_vocab = atom_vocab
        self.condition_names = condition_names
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.activation_type = activation_type
        self.add_sinusoid_posenc = add_sinusoid_posenc

        n_dim_extra = len(kwargs.get("extra_norm_values", []))
        self.in_node_nf = len(atom_vocab) + n_dim_extra + 1  # +1 atomic num
        self.context_node_nf = len(self.condition_names)

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
                f"Unknown task_type '{self.task_type}' for the "
                "Transformer ModelTaskFactory. Only 'diffusion' is "
                "supported."
            )

        dynamics_model = TransformerDynamics(
            in_node_nf=self.in_node_nf,
            context_node_nf=self.context_node_nf,
            n_dims=3,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            activation_type=self.activation_type,
            add_sinusoid_posenc=self.add_sinusoid_posenc,
        )

        model = EnVariationalDiffusion(
            dynamics=dynamics_model,
            in_node_nf=self.in_node_nf,
            n_dims=3,
            timesteps=self.kwargs["diffusion_steps"],
            noise_schedule=self.kwargs.get(
                "diffusion_noise_schedule", "polynomial_2"
            ),
            noise_precision=self.kwargs.get(
                "diffusion_noise_precision", 1e-5
            ),
            loss_type=self.kwargs.get("diffusion_loss_type", "l2"),
            norm_values=self.kwargs.get("normalize_factors", [1, 4, 10]),
            include_charges=True,
            extra_norm_values=self.kwargs.get("extra_norm_values", []),
            context_mask_rate=self.kwargs.get("context_mask_rate", 0.15),
            mask_value=self.kwargs.get("mask_value", None),
        )

        self.task = GeomMolecularGenerative(
            model,
            augment_noise=self.kwargs.get("augment_noise", False),
            data_augmentation=self.kwargs.get(
                "data_augmentation", False
            ),
            num_random_augmentations=self.kwargs.get(
                "num_random_augmentations", 0
            ),
            condition=self.condition_names,
            sp_regularizer=None,
            normalize_condition=self.kwargs.get(
                "normalize_condition", None
            ),
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
