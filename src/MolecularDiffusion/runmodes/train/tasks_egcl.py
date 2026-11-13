
from MolecularDiffusion.core.checkpoint import engine_weights
from MolecularDiffusion.callbacks.train_helper import SP_regularizer
from MolecularDiffusion.modules.tasks import ProperyPrediction, GuidanceModelPrediction,  GeomMolecularGenerative
from MolecularDiffusion.modules.models import EGNN, EGNN_dynamics, NoiseModel, EnVariationalDiffusion
from MolecularDiffusion.utils import adjust_weights, adjust_bias
from MolecularDiffusion.modules.tasks import SSL3D, CoordDenoiseObjective, MaskedAtomTypeObjective, PairwiseDistObjective
from MolecularDiffusion.modules.layers.common import SinusoidsEmbeddingNew
import torch
import logging

logger = logging.getLogger(__name__)



def _build_ssl3d_objectives(kwargs: dict) -> list:
    """Build the list of SSL3DObjective instances from factory kwargs."""
    objectives = []
    objectives.append(CoordDenoiseObjective(
        weight=kwargs.get("ssl3d_denoise_weight", 1.0),
        sigma_min=kwargs.get("ssl3d_sigma_min", 0.01),
        sigma_max=kwargs.get("ssl3d_sigma_max", 1.0),
        sigma_schedule=kwargs.get("ssl3d_sigma_schedule", "uniform"),
    ))
    objectives.append(MaskedAtomTypeObjective(
        weight=kwargs.get("ssl3d_mtype_weight", 0.5),
        mask_rate=kwargs.get("ssl3d_mask_rate", 0.15),
        atom_vocab_size=kwargs.get("ssl3d_atom_vocab_size", 5),
    ))
    dist_weight = kwargs.get("ssl3d_dist_weight", 0.0)
    if dist_weight > 0.0:
        objectives.append(PairwiseDistObjective(
            weight=dist_weight,
            k_pairs=kwargs.get("ssl3d_dist_k_pairs", 16),
        ))
    return objectives


class ModelTaskFactory:
    """
    Factory to construct models and task handlers for different learning paradigms:
    - Molecular diffusion
    - Property prediction
    - Guidance-conditioned generation

    Constructor Parameters:
        task_type (str): One of "diffusion", "regression", or "guidance".
        atom_vocab (list): List of atom vocabulary used for encoding.
        condition_names (list): List of conditional labels.
        hidden_size (int): Hidden dimension size.
        act_fn (str): Activation function name.
        num_layers (int): Number of layers.
        attention (bool): Use attention mechanism or not.
        tanh (bool): Use tanh activation or not.
        num_sublayers (int): Number of sublayers in EGNN.
        sin_embedding (bool): Use sinusoidal embedding.
        aggregation_method (str): Aggregation method (e.g., sum, mean).
        dropout (float): Dropout probability.
        normalization (bool): Apply normalization.
        include_cosine (bool): Include cosine features.
        norm_constant (float): Normalization constant.
        normalization_factor (float): Scaling for norm.
        chkpt_path (str): Optional path to model checkpoint.

        Diffusion kwargs:
            diffusion_steps (int): Number of timesteps.
            diffusion_noise_schedule (str)
            diffusion_noise_precision (float)
            diffusion_loss_type (str)
            normalize_factors (List)
            extra_norm_values (List)
            augment_noise (bool)
            data_augmentation (bool)
            context_mask_rate (float)
            mask_value (float)
            normalize_condition (str)
            sp_regularizer_regularizer (str)
            sp_regularizer_lambda_ (float)
            sp_regularizer_lambda_2 (float)
            sp_regularizer_lambda_update_value (float)
            sp_regularizer_lambda_update_step (int)
            sp_regularizer_polynomial_p (float)
            sp_regularizer_warm_up_steps (int)

        Property-prediction kwargs:
            task_learn (List)
            criterion (str)
            metric (List)
            num_mlp_layer (int)
            mlp_dropout (float)

        Guidance kwargs:
            diffusion_steps (int)
            diffusion_noise_precision (float)
            nu_arr (List)
            mapping (List)
            task_learn (List)
            metric (List)
            num_mlp_layer (int)
            mlp_dropout (float)
            weight_classes (list)
            norm_values (list)
            t_max (float)
    """
    def __init__(
        self,
        task_type: str,
        atom_vocab: list,
        condition_names: list = [],
        # Common model arguments
        hidden_size: int = 64,
        act_fn: torch.nn.Module = torch.nn.SiLU(),
        num_layers: int = 1,
        attention: bool = True,
        tanh: bool = True,
        num_sublayers: int = 9,
        sin_embedding: bool = True,
        aggregation_method: str = "sum",
        dropout: float = 0.0,
        normalization: bool = False,
        include_cosine: bool = True,
        norm_constant: float = 1.0,
        normalization_factor: float = 1.0,
        chkpt_path: str = None,
        **kwargs
    ):
        self.task_type = task_type
        self.atom_vocab = atom_vocab
        self.condition_names = condition_names
        # Common model hyperparameters
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.num_layers = num_layers
        self.attention = attention
        self.tanh = tanh
        self.num_sublayers = num_sublayers
        self.sin_embedding = sin_embedding
        self.aggregation_method = aggregation_method
        self.dropout = dropout
        self.normalization = normalization
        self.include_cosine = include_cosine
        self.norm_constant = norm_constant
        self.normalization_factor = normalization_factor

        # Compute feature dimensions
        n_dim_extra = len(kwargs.get("extra_norm_values", []))
        self.in_node_nf = len(atom_vocab) + n_dim_extra + 1 # +1 for atomic number
        self.node_feature = kwargs.get("node_feature", kwargs.get("node_feature_choice", None))
        self.node_feature_choice = kwargs.get("node_feature_choice", self.node_feature)
        self.node_feature_dim = kwargs.get("node_feature_dim", n_dim_extra)
        
        self.dynamics_in_node_nf = self.in_node_nf + 1 # +1 for time (always include time in dynamics)
        self.context_node_nf = len(self.condition_names)

        self.chkpt_path = chkpt_path
        self.kwargs = kwargs
        
        # Resolve hybrid adapter/concat configuration
        adapter_conditions = self.kwargs.get("adapter_conditions", None)
        use_adapter_module = self.kwargs.get("use_adapter_module", False)
        
        if adapter_conditions:
            # New hybrid config: map condition names to indices
            for ac in adapter_conditions:
                if ac not in self.condition_names:
                    raise ValueError(
                        f"adapter_conditions entry '{ac}' not found in condition_names {self.condition_names}"
                    )
            self.adapter_indices = [self.condition_names.index(ac) for ac in adapter_conditions]
            self.concat_indices = [i for i in range(len(self.condition_names)) if i not in self.adapter_indices]
        elif use_adapter_module:
            # DEPRECATED: use_adapter_module=True routes ALL conditions through adapter.
            # Prefer adapter_conditions: [...] for fine-grained control. Will be removed in a future version.
            import warnings
            warnings.warn(
                "use_adapter_module=True is deprecated. Use 'adapter_conditions: [...]' instead "
                "to specify which conditions use the adapter. Setting all conditions to adapter.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.adapter_indices = list(range(len(self.condition_names)))
            self.concat_indices = []
        else:
            # Default: all conditions through concatenation
            self.adapter_indices = []
            self.concat_indices = list(range(len(self.condition_names)))
        
        self.n_adapter_context = len(self.adapter_indices)
        self.n_concat_context = len(self.concat_indices)
        
        # Validation
        if self.n_adapter_context > 0 and self.context_node_nf < 1:
            raise ValueError("Must specify conditions to use the adapter module.")

    def build(self):
        """
        Build and return (model, task) based on task_type.

        Parameters:
            task_type (str): "diffusion", "property", or "guidance".
            **kwargs: task-specific keyword arguments.


        """
        is_main_process = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        
        # For diffusion_hybrid, use scalar atomic numbers (1 dim) instead of one-hot
        if self.task_type == "diffusion_hybrid":
            dynamics_in_node_nf = 1 + len(self.kwargs.get("extra_norm_values", [])) + 1  # atomic_num(1) + extra + time(1)
        else:
            dynamics_in_node_nf = self.dynamics_in_node_nf
        
        # Construct shared EGNN dynamics
        dynamics_model = EGNN_dynamics(
            in_node_nf=dynamics_in_node_nf,
            context_node_nf=self.context_node_nf,
            n_dims=3,
            hidden_nf=self.hidden_size,
            act_fn=self.act_fn,
            n_layers=self.num_layers,
            attention=self.attention,
            tanh=self.tanh,
            norm_constant=self.norm_constant,
            inv_sublayers=self.num_sublayers,
            sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
            condition_time=True,
            dropout=self.dropout,
            normalization=self.normalization,
            include_cosine=self.include_cosine,
            adapter_indices=self.adapter_indices,
            concat_indices=self.concat_indices,
        )

        if self.task_type == "diffusion":
            model = EnVariationalDiffusion(
                dynamics=dynamics_model,
                in_node_nf=self.in_node_nf,
                n_dims=3,
                timesteps=self.kwargs["diffusion_steps"],
                noise_schedule=self.kwargs.get("diffusion_noise_schedule", "polynomial_2"),
                noise_precision=self.kwargs.get("diffusion_noise_precision", 1e-5),
                loss_type=self.kwargs.get("diffusion_loss_type", "l2"),
                norm_values=self.kwargs.get("normalize_factors", [1,4,10]),
                include_charges=True,
                extra_norm_values=self.kwargs.get("extra_norm_values", []),
                context_mask_rate=self.kwargs.get("context_mask_rate", 0.15),
                mask_value=self.kwargs.get("mask_value", None), # CFG
            )
            
            if self.kwargs.get("sp_regularizer_deploy", False):
                if is_main_process:
                    logging.info("SP regularizer is enabled for diffusion task.")
                sp_reg = SP_regularizer(
                    regularizer=self.kwargs.get("sp_regularizer_regularizer", "hard"),
                    lambda_=self.kwargs.get("sp_regularizer_lambda_", 0),
                    lambda_2=self.kwargs.get("sp_regularizer_lambda_2", 1000),
                    lambda_update_value=self.kwargs.get("sp_regularizer_lambda_update_value", 1),
                    lambda_update_step=self.kwargs.get("sp_regularizer_lambda_update_step", 100),
                    polynomial_p=self.kwargs.get("sp_regularizer_polynomial_p", 1.1),
                    warm_up_steps=self.kwargs.get("sp_regularizer_warm_up_steps", 100),
                )
            else:
                if is_main_process:
                    logging.info("SP regularizer is disabled for diffusion task.")
                sp_reg = None
            self.task = GeomMolecularGenerative(
                model,
                augment_noise=self.kwargs.get("augment_noise", False),
                data_augmentation=self.kwargs.get("data_augmentation", False),
                condition=self.condition_names, # CFG and conditional 
                sp_regularizer=sp_reg,
                normalize_condition=self.kwargs.get("normalize_condition", None),
                reference_indices=self.kwargs.get("reference_indices", None), # outpaint task
            )


        elif self.task_type == "regression":
            model = EGNN(
                in_node_nf=self.in_node_nf,
                hidden_nf=self.hidden_size,
                act_fn=self.act_fn,
                n_layers=self.num_layers,
                attention=self.attention,
                tanh=self.tanh,
                norm_constant=self.norm_constant,
                inv_sublayers=self.num_sublayers,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
                dropout=self.dropout,
                normalization=False,
                include_cosine=self.include_cosine,
            )
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
                normalization=self.kwargs.get("target_normalization", True),  # Normalize targets by mean/std
                num_class=len(self.kwargs.get("task_learn", "")),
                prediction_mlp_type=self.kwargs.get("prediction_mlp_type", "pernode"),
                prediction_activation=self.kwargs.get("prediction_activation", "relu"),
            )

        elif self.task_type == "ssl3d":
            t_dim = SinusoidsEmbeddingNew().dim
            model = EGNN(
                in_node_nf=self.in_node_nf + t_dim,
                hidden_nf=self.hidden_size,
                act_fn=self.act_fn,
                n_layers=self.num_layers,
                attention=self.attention,
                tanh=self.tanh,
                norm_constant=self.norm_constant,
                inv_sublayers=self.num_sublayers,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
                dropout=self.dropout,
                normalization=False,
                include_cosine=self.include_cosine,
            )
            objectives = _build_ssl3d_objectives(self.kwargs)
            self.task = SSL3D(model, objectives, include_charge=True)

        elif self.task_type == "guidance":
            model = EGNN(
                in_node_nf=self.dynamics_in_node_nf,
                hidden_nf=self.hidden_size,
                act_fn=self.act_fn,
                n_layers=self.num_layers,
                attention=self.attention,
                tanh=self.tanh,
                norm_constant=self.norm_constant,
                inv_sublayers=self.num_sublayers,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
                dropout=self.dropout,
                normalization=self.normalization,
                include_cosine=self.include_cosine,
            )
            noise_model = NoiseModel(
                timestep=self.kwargs.get("diffusion_steps"),
                noise_precision=self.kwargs.get("diffusion_noise_precision"),
                nu_arr=self.kwargs.get("nu_arr"),
                mapping=self.kwargs.get("mapping"),
            )
            # Select task class based on dense_mode flag
            task_kwargs = dict(
                task=self.kwargs.get("task_learn", ""),
                include_charge=True,
                metric=self.kwargs.get("metric", "mae"),
                num_mlp_layer=self.kwargs.get("num_mlp_layer", 2),
                mlp_batch_norm=True,
                mlp_dropout=self.kwargs.get("mlp_dropout", 0.0),
                normalization=True,
                weight_classes=self.kwargs.get("weight_classes"),
                norm_values=self.kwargs.get("norm_values"),
                t_max=self.kwargs.get("t_max"),
                num_class=len(self.kwargs.get("task_learn", "")),
                loss_weighting=self.kwargs.get("loss_weighting", "none")
            )
            


            self.task = GuidanceModelPrediction(model, noise_model, **task_kwargs)

        else:   
            raise ValueError(f"Unknown task_type '{self.task_type}'. Choose 'diffusion', 'regression', 'guidance', or 'ssl3d'.")
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # type: ignore
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
                    made_adjustment = False

                    # Suffix-based key search: handles both direct (egnn.embedding...)
                    # and PyGAdapter-wrapped (egnn_dynamics.egnn.embedding...) paths
                    emb_in_keys = [k for k in chk_point.keys() if k.endswith("egnn.embedding.layers.0.weight")]
                    emb_out_w_keys = [k for k in chk_point.keys() if k.endswith("egnn.embedding_out.layers.2.weight")]
                    emb_out_b_keys = [k for k in chk_point.keys() if k.endswith("egnn.embedding_out.layers.2.bias")]

                    if emb_in_keys and emb_out_w_keys and emb_out_b_keys:
                        k_emb_in = emb_in_keys[0]
                        k_emb_out_w = emb_out_w_keys[0]
                        k_emb_out_b = emb_out_b_keys[0]

                        n_dim_pretrain = chk_point[k_emb_in].shape[1]
                        n_extra_dim = self.dynamics_in_node_nf + self.n_concat_context - n_dim_pretrain

                        if n_extra_dim > 0:
                            if is_main_process:
                                logger.info("Adding dimensions to the EGNN input embedding...")
                            chk_point[k_emb_in] = adjust_weights(
                                chk_point[k_emb_in],
                                (self.hidden_size, n_dim_pretrain + n_extra_dim),
                            )

                            chk_point[k_emb_out_w] = adjust_weights(
                                chk_point[k_emb_out_w],
                                (n_dim_pretrain + n_extra_dim, self.hidden_size),
                            )

                            chk_point[k_emb_out_b] = adjust_bias(
                                chk_point[k_emb_out_b],
                                (n_dim_pretrain + n_extra_dim,),
                            )
                            made_adjustment = True

                    if self.n_adapter_context > 0:
                        emb_c_in_w_keys = [
                            k for k in chk_point.keys()
                            if k.endswith("egnn.emb_c_in.layers.0.weight")
                        ]
                        if emb_c_in_w_keys:
                            emb_c_in_w_key = emb_c_in_w_keys[0]
                            n_context_pretrain = chk_point[emb_c_in_w_key].shape[1]
                            n_context_extra = self.n_adapter_context - n_context_pretrain
                            if n_context_extra > 0:
                                if is_main_process:
                                    logger.info("Adding dimensions to the adapter context embedding...")
                                chk_point[emb_c_in_w_key] = adjust_weights(
                                    chk_point[emb_c_in_w_key],
                                    (self.hidden_size, n_context_pretrain + n_context_extra),
                                )
                                made_adjustment = True

                    if made_adjustment:
                        res = self.task.load_state_dict(chk_point, strict=False)
                        if res.missing_keys or res.unexpected_keys:
                            if is_main_process:
                                logger.warning(f"\033[93mCheckpoint loaded with mismatched keys after adjustment.\033[0m")
                                if res.missing_keys:
                                    logger.warning(f"\033[93mMissing keys ({len(res.missing_keys)}): {res.missing_keys}\033[0m")
                                if res.unexpected_keys:
                                    logger.warning(f"\033[93mUnexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys}\033[0m")
                    else:
                        raise RuntimeError(f"The specified model configuration does not match with the checkpoint. Original error: {e}")
                                
    
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

    def adjust_state_dict(self, state_dict, task):
        """Adjust checkpoint state dict dimensions to match the current model.
        
        Handles size mismatches from added conditions by resizing EGCL
        embedding layers and adapter context embeddings. Uses suffix-based
        key matching on BOTH checkpoint and model state dicts to handle
        cross-prefix scenarios (e.g. PyGAdapter wrapping).
        
        Args:
            state_dict: The checkpoint state dict to adjust.
            task: The task model (used to get expected dimensions).
            
        Returns:
            The adjusted state dict.
        """
        model_sd = task.state_dict()
        
        def _find_key(sd, suffix):
            """Find a key in state dict ending with the given suffix."""
            matches = [k for k in sd if k.endswith(suffix)]
            return matches[0] if matches else None
        
        # --- EGCL embedding layers ---
        suffix_in = "egnn.embedding.layers.0.weight"
        suffix_out_w = "egnn.embedding_out.layers.2.weight"
        suffix_out_b = "egnn.embedding_out.layers.2.bias"
        
        ckpt_in = _find_key(state_dict, suffix_in)
        ckpt_out_w = _find_key(state_dict, suffix_out_w)
        ckpt_out_b = _find_key(state_dict, suffix_out_b)
        model_in = _find_key(model_sd, suffix_in)
        
        if ckpt_in and ckpt_out_w and ckpt_out_b and model_in:
            expected_dim = model_sd[model_in].shape[1]
            ckpt_dim = state_dict[ckpt_in].shape[1]
            hidden = state_dict[ckpt_in].shape[0]
            
            if expected_dim != ckpt_dim:
                logger.info(f"Adjusting EGCL embedding: {ckpt_dim} -> {expected_dim}")
                state_dict[ckpt_in] = adjust_weights(state_dict[ckpt_in], (hidden, expected_dim))
                state_dict[ckpt_out_w] = adjust_weights(state_dict[ckpt_out_w], (expected_dim, hidden))
                state_dict[ckpt_out_b] = adjust_bias(state_dict[ckpt_out_b], (expected_dim,))
        
        # --- Adapter context embedding ---
        suffix_adapter = "egnn.emb_c_in.layers.0.weight"
        ckpt_adapter = _find_key(state_dict, suffix_adapter)
        model_adapter = _find_key(model_sd, suffix_adapter)
        
        if ckpt_adapter and model_adapter:
            expected_ctx = model_sd[model_adapter].shape[1]
            ckpt_ctx = state_dict[ckpt_adapter].shape[1]
            hidden_ctx = state_dict[ckpt_adapter].shape[0]
            if expected_ctx != ckpt_ctx:
                logger.info(f"Adjusting adapter context: {ckpt_ctx} -> {expected_ctx}")
                state_dict[ckpt_adapter] = adjust_weights(
                    state_dict[ckpt_adapter], (hidden_ctx, expected_ctx))
        
        return state_dict
