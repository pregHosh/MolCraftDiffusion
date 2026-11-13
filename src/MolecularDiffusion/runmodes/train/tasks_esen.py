from MolecularDiffusion.core.checkpoint import engine_weights
from MolecularDiffusion.modules.tasks import ProperyPrediction, GuidanceModelPrediction
from MolecularDiffusion.modules.tasks import SSL3D
from MolecularDiffusion.modules.layers.common import SinusoidsEmbeddingNew
from MolecularDiffusion.runmodes.train.tasks_egcl import _build_ssl3d_objectives
from MolecularDiffusion.modules.models.esen import eSEN_Backbone, eSEN_dynamics
from MolecularDiffusion.modules.models import NoiseModel

import torch
import logging

logger = logging.getLogger(__name__)


class ModelTaskFactory:
    """
    Factory to construct eSEN models and task handlers for regression, guidance,
    and SSL3D tasks.

    Constructor Parameters:
        task_type (str): "regression", "guidance", or "ssl3d".
        atom_vocab (list): List of atom vocabulary used for encoding.
        condition_names (list): List of conditional labels.
        hidden_size (int): Hidden dimension size.
        num_layers (int): Number of layers.

        # eSEN specific
        lmax (int): Maximum degree of spherical harmonics.
        mmax (int): Maximum order of spherical harmonics.
        grid_resolution (int): Resolution of the grid.
        cutoff (float): Cutoff radius.
        edge_channels (int): Number of edge channels.
        distance_function (str): Distance function type.
        num_distance_basis (int): Number of distance basis functions.
        norm_type (str): Normalization type.
        act_type (str): Activation type.
        mlp_type (str): MLP type.
        use_envelope (bool): Whether to use envelope function.
        activation_checkpointing (bool): Whether to use activation checkpointing.
        global_attributes (bool): Whether to use global attributes.
        sphere_embedding_type (str): Type of sphere embedding.

        Property-prediction kwargs:
            task_learn (List)
            criterion (str)
            metric (List)
            num_mlp_layer (int)
            mlp_dropout (float)
            normalization (bool)
            chkpt_path (str)
    """
    def __init__(
        self,
        task_type: str,
        atom_vocab: list,
        condition_names: list = [],
        # Common model arguments
        hidden_size: int = 128,
        num_layers: int = 2,
        # eSEN specific
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        hidden_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        norm_type: str = "rms_norm_sh",
        act_type: str = "s2",
        mlp_type: str = "grid",
        use_envelope: bool = False,
        activation_checkpointing: bool = False,
        global_attributes: bool = False,
        sphere_embedding_type: str = "mixed",
        aggregation_method: str = "sum",

        chkpt_path: str = None,
        **kwargs
    ):
        self.task_type = task_type
        self.atom_vocab = atom_vocab
        self.condition_names = condition_names

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lmax = lmax
        self.mmax = mmax
        self.grid_resolution = grid_resolution
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.hidden_channels = hidden_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis
        self.norm_type = norm_type
        self.act_type = act_type
        self.mlp_type = mlp_type
        self.use_envelope = use_envelope
        self.activation_checkpointing = activation_checkpointing
        self.global_attributes = global_attributes
        self.sphere_embedding_type = sphere_embedding_type
        self.aggregation_method = aggregation_method

        self.chkpt_path = chkpt_path
        self.kwargs = kwargs

        # Compute feature dimensions
        n_dim_extra = len(kwargs.get("extra_norm_values", []))
        self.in_node_nf = len(atom_vocab) + n_dim_extra + 1  # +1 for atomic number

        # Account for unknown atom category if fallback is enabled
        # during training allow_unknown appends Suisei to the vocab, so we don't need to add it twice
        if kwargs.get("use_unknown_fallback", False) and "Suisei" not in atom_vocab:
            self.in_node_nf += 1

        self.context_node_nf = len(self.condition_names)

    def build(self):
        """
        Build and return (model, task) based on task_type.

        Parameters:
            task_type (str): "regression", "guidance", or "ssl3d".
        """
        is_main_process = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        if self.task_type == "regression":

            # Instantiate eSEN_Backbone
            model = eSEN_Backbone(
                max_num_elements=100, # Assuming max atomic number < 100
                sphere_channels=self.hidden_size,
                lmax=self.lmax,
                mmax=self.mmax,
                grid_resolution=self.grid_resolution,
                otf_graph=True, # Always compute graph on the fly for now
                max_neighbors=300, # Default max neighbors
                use_pbc=False, # Assuming molecular data
                use_pbc_single=False,
                cutoff=self.cutoff,
                edge_channels=self.edge_channels,
                distance_function=self.distance_function,
                num_distance_basis=self.num_distance_basis,
                direct_forces=False, # Not needed for property prediction usually
                regress_forces=False,
                regress_stress=False,
                num_layers=self.num_layers,
                hidden_channels=self.hidden_channels,
                norm_type=self.norm_type,
                act_type=self.act_type,
                mlp_type=self.mlp_type,
                use_envelope=self.use_envelope,
                activation_checkpointing=self.activation_checkpointing,
                global_attributes=self.global_attributes,
                sphere_embedding_type=self.sphere_embedding_type,
                in_node_channels=self.in_node_nf
            )
            # eSEN model needs hidden_nf attribute for task wrapper
            model.hidden_nf = model.d_model

            self.task = ProperyPrediction(
                model,
                task=self.kwargs.get("task_learn", ""),
                include_charge=True,
                criterion=self.kwargs.get("criterion", "mse"),
                metric=self.kwargs.get("metric", "mae"),
                num_mlp_layer=self.kwargs.get("num_mlp_layer", 2),
                mlp_batch_norm=self.kwargs.get("mlp_batch_norm", None),  # Options: None, 'layernorm', 'batchnorm'
                mlp_dropout=self.kwargs.get("mlp_dropout", 0.0),
                mlp_hidden_dim=self.kwargs.get("mlp_hidden_dim", None),
                readout=self.aggregation_method,
                normalization=self.kwargs.get("target_normalization", True),  # Normalize targets by mean/std
                num_class=len(self.kwargs.get("task_learn", "")),
                prediction_mlp_type=self.kwargs.get("prediction_mlp_type", "pernode"),
                prediction_activation=self.kwargs.get("prediction_activation", "relu"),
            )

        elif self.task_type == "guidance":
            model = eSEN_Backbone(
                max_num_elements=100,
                sphere_channels=self.hidden_size,
                lmax=self.lmax,
                mmax=self.mmax,
                grid_resolution=self.grid_resolution,
                otf_graph=True,
                max_neighbors=300,
                use_pbc=False,
                use_pbc_single=False,
                cutoff=self.cutoff,
                edge_channels=self.edge_channels,
                distance_function=self.distance_function,
                num_distance_basis=self.num_distance_basis,
                direct_forces=False,
                regress_forces=False,
                regress_stress=False,
                num_layers=self.num_layers,
                hidden_channels=self.hidden_size,
                norm_type=self.norm_type,
                act_type=self.act_type,
                mlp_type=self.mlp_type,
                use_envelope=self.use_envelope,
                activation_checkpointing=self.activation_checkpointing,
                global_attributes=self.global_attributes,
                sphere_embedding_type=self.sphere_embedding_type,
                in_node_channels=self.in_node_nf + 1 # +1 for time
            )
            model.hidden_nf = model.d_model

            noise_model = NoiseModel(
                timestep=self.kwargs.get("diffusion_steps"),
                noise_precision=self.kwargs.get("diffusion_noise_precision"),
                nu_arr=self.kwargs.get("nu_arr"),
                mapping=self.kwargs.get("mapping"),
            )

            self.task = GuidanceModelPrediction(
                model,
                noise_model,
                task=self.kwargs.get("task_learn", ""),
                include_charge=True,
                metric=self.kwargs.get("metric", "mae"),
                num_mlp_layer=self.kwargs.get("num_mlp_layer", 2),
                mlp_batch_norm=True,
                mlp_dropout=self.kwargs.get("mlp_dropout", 0.0),
                readout=self.aggregation_method,
                normalization=True,
                weight_classes=self.kwargs.get("weight_classes"),
                norm_values=self.kwargs.get("norm_values"),
                t_max=self.kwargs.get("t_max"),
                num_class=len(self.kwargs.get("task_learn", "")),
                loss_weighting=self.kwargs.get("loss_weighting", "none")
            )

        elif self.task_type == "ssl3d":
            t_dim = SinusoidsEmbeddingNew().dim
            model = eSEN_Backbone(
                max_num_elements=100,
                sphere_channels=self.hidden_size,
                lmax=self.lmax,
                mmax=self.mmax,
                grid_resolution=self.grid_resolution,
                otf_graph=True,
                max_neighbors=300,
                use_pbc=False,
                use_pbc_single=False,
                cutoff=self.cutoff,
                edge_channels=self.edge_channels,
                distance_function=self.distance_function,
                num_distance_basis=self.num_distance_basis,
                direct_forces=False,
                regress_forces=False,
                regress_stress=False,
                num_layers=self.num_layers,
                hidden_channels=self.hidden_channels,
                norm_type=self.norm_type,
                act_type=self.act_type,
                mlp_type=self.mlp_type,
                use_envelope=self.use_envelope,
                activation_checkpointing=self.activation_checkpointing,
                global_attributes=self.global_attributes,
                sphere_embedding_type=self.sphere_embedding_type,
                in_node_channels=self.in_node_nf + t_dim,
            )
            model.hidden_nf = model.d_model
            objectives = _build_ssl3d_objectives(self.kwargs)
            self.task = SSL3D(model, objectives, include_charge=True)

        else:
            raise ValueError(
                f"Unknown task_type '{self.task_type}'. "
                f"Choose from: regression, guidance, ssl3d."
            )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_main_process:
            logger.info(f"Number of parameters: {n_params}")

        if self.chkpt_path:
            try:
                ckpt = torch.load(self.chkpt_path, weights_only=False)

                # Validate task_type matches the checkpoint
                ckpt_task_type = ckpt.get("hyperparameters", {}).get("task_type") or ckpt.get("task_type")
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

                try:
                    load_result = self.task.load_state_dict(chk_point, strict=False)
                    if load_result.missing_keys or load_result.unexpected_keys:
                        if is_main_process:
                            logger.warning(f"\033[93mCheckpoint loaded with mismatched keys.\033[0m")
                            if load_result.missing_keys:
                                logger.warning(f"\033[93mMissing keys ({len(load_result.missing_keys)}): {load_result.missing_keys}\033[0m")
                            if load_result.unexpected_keys:
                                logger.warning(f"\033[93mUnexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys}\033[0m")
                except RuntimeError as e:
                     if is_main_process:
                        logger.error(f"Failed to load checkpoint: {e}")
                        raise e

                if "mean" in chk_point and "std" in chk_point:
                    self.task.mean = chk_point["mean"]
                    self.task.std = chk_point["std"]
            except FileNotFoundError:
                if is_main_process:
                    logger.warning(f"Checkpoint not found at {self.chkpt_path}. Initializing model without loading.")
                raise FileNotFoundError(f"Checkpoint not found at {self.chkpt_path}.")

        self.task.atom_vocab = self.atom_vocab
        self.task.task_type = self.task_type

        return self.task
