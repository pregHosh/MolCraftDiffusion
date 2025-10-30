from MolecularDiffusion.callbacks.train_helper import SP_regularizer
from MolecularDiffusion.modules.tasks import ProperyPrediction, GuidanceModelPrediction, GeomMolecularGenerative
from MolecularDiffusion.modules.models import EGNN, EGNN_dynamics, NoiseModel, EnVariationalDiffusion
from MolecularDiffusion.utils import adjust_weights, adjust_bias
import torch
import logging

logger = logging.getLogger(__name__)


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
       
        self.in_node_nf = len(atom_vocab) + len(kwargs.get("extra_norm_values", [])) + 1 # +1 for atomic number
        self.dynamics_in_node_nf = self.in_node_nf + 1 # +1 for time (always include time in dynamics)
        self.context_node_nf = len(self.condition_names)
        
        # checkpoint path
        self.chkpt_path = chkpt_path
        
        self.kwargs = kwargs

    def build(self):
        """
        Build and return (model, task) based on task_type.

        Parameters:
            task_type (str): "diffusion", "property", or "guidance".
            **kwargs: task-specific keyword arguments.


        """
        # Construct shared EGNN dynamics
        dynamics_model = EGNN_dynamics(
            in_node_nf=self.dynamics_in_node_nf,
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
            use_adapter_module=self.kwargs.get("use_adapter_module", False),
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
                mlp_batch_norm=self.normalization,
                mlp_dropout=self.kwargs.get("mlp_dropout", 0.0),
                normalization=self.normalization,
                num_class=len(self.kwargs.get("task_learn", ""))
            )

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
            self.task = GuidanceModelPrediction(
                model,
                noise_model,
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
                num_class=len(self.kwargs.get("task_learn", ""))
            )

        else:
            raise ValueError(f"Unknown task_type '{self.task_type}'. Choose 'diffusion', 'regression', or 'guidance'.")

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # type: ignore
        logger.info(f"Number of parameters: {n_params}")

        if self.chkpt_path:    
            try:
                ckpt = torch.load(self.chkpt_path)
                chk_point = getattr(ckpt, "ema_model", ckpt["model"])
                logger.info(f"Loading checkpoint from {self.chkpt_path}")
                
                try:
                    load_result = self.task.load_state_dict(chk_point, strict=False)
                    if load_result.missing_keys or load_result.unexpected_keys:
                        logger.warning(f"\033[93mCheckpoint loaded with mismatched keys.\033[0m")
                        if load_result.missing_keys:
                            logger.warning(f"\033[93mMissing keys ({len(load_result.missing_keys)}): {load_result.missing_keys}\033[0m")
                        if load_result.unexpected_keys:
                            logger.warning(f"\033[93mUnexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys}\033[0m")
                except RuntimeError as e:
                    n_dim_pretrain = chk_point["model.dynamics.egnn.embedding.layers.0.weight"].shape[1] 
                    n_extra_dim = self.dynamics_in_node_nf - n_dim_pretrain + len(self.condition_names)

                    if n_extra_dim > 0:
                        logger.info("Adding dimensions to the EGNN...")
                        chk_point["model.dynamics.egnn.embedding.layers.0.weight"] = adjust_weights(
                            chk_point["model.dynamics.egnn.embedding.layers.0.weight"], (self.hidden_size, 
                                                                                        n_dim_pretrain + n_extra_dim)
                        )

                        chk_point["model.dynamics.egnn.embedding_out.layers.2.weight"] = adjust_weights(
                            chk_point["model.dynamics.egnn.embedding_out.layers.2.weight"], (n_dim_pretrain + n_extra_dim, 
                                                                                            self.hidden_size)
                        )
    
                        chk_point["model.dynamics.egnn.embedding_out.layers.2.bias"] = adjust_bias(
                        chk_point["model.dynamics.egnn.embedding_out.layers.2.bias"], (n_dim_pretrain + n_extra_dim,)
                        )      
                        res = self.task.load_state_dict(chk_point, strict=False) 
                        if res.missing_keys or res.unexpected_keys:
                            logger.warning(f"\033[93mCheckpoint loaded with mismatched keys after adjustment.\033[0m")
                            if res.missing_keys:
                                logger.warning(f"\033[93mMissing keys ({len(res.missing_keys)}): {res.missing_keys}\033[0m")
                            if res.unexpected_keys:
                                logger.warning(f"\033[93mUnexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys}\033[0m")
                    else:
                        raise RuntimeError("The specified model configuration does not match with the checkpoint.")
                                
    
                if "mean" in chk_point and "std" in chk_point:
                    self.task.mean = chk_point["mean"]
                    self.task.std = chk_point["std"]                           
            except FileNotFoundError:
                logger.warning(f"Checkpoint not found at {self.chkpt_path}. Initializing model without loading.")      
                
        self.task.atom_vocab = self.atom_vocab
            
        return self.task

