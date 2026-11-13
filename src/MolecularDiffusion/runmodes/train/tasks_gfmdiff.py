"""Factory wiring the ported GFMDiff Dual-Track Transformer backbone into
the existing, unmodified diffusion task -- structured like
``runmodes/train/tasks_egt.py::ModelTaskFactory`` (a new dynamics network
dropped into ``EnVariationalDiffusion`` + ``GeomMolecularGenerative``, zero
edits to either).

Per the approved integration plan (revision 2), this is a §1 backbone swap:
GFMDiff's own noise schedule/VLB loss/sampler and the Geometric-Facilitated
Loss (+ diffused "degree" channel) are out of scope -- only the Dual-Track
Transformer architecture (``GFMDiffDynamics`` / ``EquiGNN``) is ported, and
``modules.models.en_diffusion.EnVariationalDiffusion`` /
``modules.tasks.diffusion.GeomMolecularGenerative`` are reused unmodified.
"""

import logging

import torch

from MolecularDiffusion.core.checkpoint import engine_weights
from MolecularDiffusion.modules.models.gfmdiff.dynamics import GFMDiffDynamics
from MolecularDiffusion.modules.models.edm.en_diffusion import EnVariationalDiffusion
from MolecularDiffusion.modules.tasks.diffusion import GeomMolecularGenerative

logger = logging.getLogger(__name__)


class ModelTaskFactory:
    """
    Factory to build the GFMDiff-backbone diffusion model + task.

    Parameters:
        task_type (str): must be "diffusion".
        train_set: Training dataset, used to infer input node feature
            dimensions (kept for interface parity with other factories;
            unused directly here).
        atom_vocab (list): List of atom vocabulary used for encoding.
        task_names (list): List of conditional labels (context columns).
        condition_names (list): List of condition names for conditional
            generation.
        num_layers (int): Number of Dual-Track Transformer blocks.
        hidden_dims (dict): expects keys "emb_dim", "hidden_dim",
            "num_heads" for the GFMDiff backbone.
        chkpt_path (str): Optional path to model checkpoint.
        **kwargs: diffusion + backbone-specific keyword arguments (see
            ``diffusion_gfmdiff.yaml``).
    """

    def __init__(
        self,
        task_type: str,
        train_set=None,
        atom_vocab=None,
        task_names: list = [],
        condition_names: list = [],
        num_layers: int = 5,
        hidden_dims: dict = {},
        chkpt_path: str = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.train_set = train_set
        self.atom_vocab = atom_vocab
        self.task_names = task_names
        self.condition_names = condition_names
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        n_dim_extra = len(kwargs.get("extra_norm_values", []))
        self.in_node_nf = len(atom_vocab) + n_dim_extra + 1  # +1 for atomic number
        self.context_node_nf = len(self.task_names)

        self.chkpt_path = chkpt_path
        self.kwargs = kwargs

    def build(self):
        """Build and return the GeomMolecularGenerative task."""
        is_main_process = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        if self.task_type != "diffusion":
            raise ValueError(
                f"Unknown task_type '{self.task_type}' for GFMDiff ModelTaskFactory. "
                "Only 'diffusion' is supported."
            )

        dynamics_model = GFMDiffDynamics(
            in_node_nf=self.in_node_nf,
            context_node_nf=self.context_node_nf,
            n_dims=3,
            num_layers=self.num_layers,
            emb_dim=self.hidden_dims.get("emb_dim", 256),
            hidden_dim=self.hidden_dims.get("hidden_dim", 512),
            num_heads=self.hidden_dims.get("num_heads", 8),
            dropout=self.kwargs.get("dropout", 0.1),
            pair_loss_scale=self.kwargs.get("pair_loss_scale", 0.0),
            add_time=self.kwargs.get("add_time", False),
            block_calc=self.kwargs.get("block_calc", True),
            dataset_name=self.kwargs.get("dataset_name", "qm9"),
        )

        model = EnVariationalDiffusion(
            dynamics=dynamics_model,
            in_node_nf=self.in_node_nf,
            n_dims=3,
            timesteps=self.kwargs["diffusion_steps"],
            noise_schedule=self.kwargs.get("diffusion_noise_schedule", "polynomial_2"),
            noise_precision=self.kwargs.get("diffusion_noise_precision", 1e-5),
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
            data_augmentation=self.kwargs.get("data_augmentation", False),
            condition=self.task_names,
            sp_regularizer=None,
            normalize_condition=self.kwargs.get("normalize_condition", None),
            reference_indices=self.kwargs.get("reference_indices", None),
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_main_process:
            logger.info(f"Number of parameters: {n_params}")

        if self.chkpt_path:
            try:
                ckpt = torch.load(self.chkpt_path, weights_only=False)

                ckpt_task_type = ckpt.get("hyperparameters", {}).get("task_type") or ckpt.get(
                    "task_type"
                )
                if ckpt_task_type is not None and ckpt_task_type != self.task_type:
                    raise ValueError(
                        f"Task type mismatch: checkpoint was trained as '{ckpt_task_type}' "
                        f"but current config specifies '{self.task_type}'. "
                        f"Update your config to use tasks: {ckpt_task_type} or point to the correct checkpoint."
                    )

                chk_point = engine_weights(ckpt)
                if chk_point is None:
                    raise KeyError("Checkpoint missing both 'ema_model' and 'model'")

                if is_main_process:
                    logger.info(f"Loading checkpoint from {self.chkpt_path}")

                load_result = self.task.load_state_dict(chk_point, strict=False)
                if is_main_process and (load_result.missing_keys or load_result.unexpected_keys):
                    logger.warning("\033[93mCheckpoint loaded with mismatched keys.\033[0m")
                    if load_result.missing_keys:
                        logger.warning(
                            f"\033[93mMissing keys ({len(load_result.missing_keys)}): {load_result.missing_keys}\033[0m"
                        )
                    if load_result.unexpected_keys:
                        logger.warning(
                            f"\033[93mUnexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys}\033[0m"
                        )

                if "mean" in chk_point and "std" in chk_point:
                    self.task.mean = chk_point["mean"]
                    self.task.std = chk_point["std"]
            except FileNotFoundError:
                if is_main_process:
                    logger.warning(
                        f"Checkpoint not found at {self.chkpt_path}. Initializing model without loading."
                    )
                raise FileNotFoundError(f"Checkpoint not found at {self.chkpt_path}.")

        self.task.atom_vocab = self.atom_vocab
        self.task.task_type = self.task_type

        return self.task
