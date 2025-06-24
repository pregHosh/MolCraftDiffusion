from MolecularDiffusion.callbacks.train_helper import SP_regularizer
from MolecularDiffusion.modules.tasks import ProperyPrediction, GuidanceModelPrediction, GeomMolecularGenerative
from MolecularDiffusion.modules.models import EGNN, EGNN_dynamics, NoiseModel, EnVariationalDiffusion
import torch

class ModelTaskFactory:
    """
    Factory to build models and tasks for diffusion, property prediction, or guidance.

    Usage:
        factory = ModelTaskFactory(
            train_set=train_data,
            atom_vocab=atom_vocab_list,
            task_names=task_name_list,
            include_charges=False,av
            hidden_size=64,
            act_fn="relu",
            num_layers=5,
            attention=False,
            tanh=False,
            num_sublayers=2,
            sin_embedding=True,
            aggregation_method="sum",
            dropout=0.0,
            normalization=False,
            include_cosine=False,
            norm_constant=1.0,
            normalization_factor=1.0
        )
        # Diffusion-specific kwargs:
        diffusion_kwargs = {
            "diffusion_steps": 1000,
            "diffusion_noise_schedule": "linear",
            "diffusion_noise_precision": None,
            "diffusion_loss_type": "l2",
            "normalize_factors": None,
            "extra_norm_values": None,
            "augment_noise": False,
            "data_augmentation": False,
            "context_mask_rate": 0.15,
            "mask_value": -4,
            "normalize_condition": "value_2",
            # SP_regularizer args:
            "sp_regularizer_regularizer": "hard",
            "sp_regularizer_lambda_": 0,
            "sp_regularizer_lambda_2": 1000,
            "sp_regularizer_lambda_update_value": 1,
            "sp_regularizer_lambda_update_step": 100,
            "sp_regularizer_polynomial_p": 1.1,
            "sp_regularizer_warm_up_steps": 100,
        }
        model, task = factory.build("diffusion", **diffusion_kwargs)
    """
    def __init__(
        self,
        train_set,
        atom_vocab,
        task_names,
        # Common model arguments
        hidden_size: int = 64,
        act_fn: str = "relu",
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
    ):
        self.train_set = train_set
        self.atom_vocab = atom_vocab
        self.task_names = task_names
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
       
        self.in_node_nf = self.train_set[0]["node_feature"].size(1) + 1 # +1 for atomic number
        self.dynamics_in_node_nf = self.in_node_nf + 1 # +1 for time 
        self.context_node_nf = len(self.task_names)
        
        # checkpoint path
        self.chkpt_path = chkpt_path

    def build(self, task_type: str, **kwargs):
        """
        Build and return (model, task) based on task_type.

        Parameters:
            task_type (str): "diffusion", "property", or "guidance".
            **kwargs: task-specific keyword arguments.

        Diffusion kwargs:
            diffusion_steps (int): Number of timesteps.
            diffusion_noise_schedule (str)
            diffusion_noise_precision
            diffusion_loss_type (str)
            normalize_factors
            extra_norm_values
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
            task_learn (str)
            criterion (str)
            metric (str)
            num_mlp_layer (int)
            mlp_dropout (float)

        Guidance kwargs:
            diffusion_steps (int)
            diffusion_noise_precision
            nu_arr
            mapping
            task_learn (str)
            metric (str)
            num_mlp_layer (int)
            mlp_dropout (float)
            weight_classes
            norm_values
            t_max
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
            use_adapter_module=kwargs.get("use_adapter_module", False),
        )

        if task_type == "diffusion":
            model = EnVariationalDiffusion(
                dynamics=dynamics_model,
                in_node_nf=self.in_node_nf,
                n_dims=3,
                timesteps=kwargs["diffusion_steps"],
                noise_schedule=kwargs.get("diffusion_noise_schedule", "polynomial_2"),
                noise_precision=kwargs.get("diffusion_noise_precision", 1e-5),
                loss_type=kwargs.get("diffusion_loss_type", "l2"),
                norm_values=kwargs.get("normalize_factors", [1,4,10]),
                include_charges=True,
                extra_norm_values=kwargs.get("extra_norm_values", []),
                context_mask_rate=kwargs.get("context_mask_rate", 0.15),
                mask_value=kwargs.get("mask_value", None), # CFG
            )
            
            if kwargs.get("sp_regularizer_deploy", False):
                sp_reg = SP_regularizer(
                    regularizer=kwargs.get("sp_regularizer_regularizer", "hard"),
                    lambda_=kwargs.get("sp_regularizer_lambda_", 0),
                    lambda_2=kwargs.get("sp_regularizer_lambda_2", 1000),
                    lambda_update_value=kwargs.get("sp_regularizer_lambda_update_value", 1),
                    lambda_update_step=kwargs.get("sp_regularizer_lambda_update_step", 100),
                    polynomial_p=kwargs.get("sp_regularizer_polynomial_p", 1.1),
                    warm_up_steps=kwargs.get("sp_regularizer_warm_up_steps", 100),
                )
            else:
                sp_reg = None
            self.task = GeomMolecularGenerative(
                model,
                augment_noise=kwargs.get("augment_noise", False),
                data_augmentation=kwargs.get("data_augmentation", False),
                condition=self.task_names, # CFG and conditional 
                sp_regularizer=sp_reg,
                normalize_condition=kwargs.get("normalize_condition", None),
                reference_indices=kwargs.get("reference_indices", None), # outpaint task
            )

        elif task_type == "property":
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
                normalization=self.normalization,
                include_cosine=self.include_cosine,
            )
            self.task = ProperyPrediction(
                model,
                task=kwargs.get("task_learn", ""),
                include_charge=True,
                criterion=kwargs.get("criterion", "mse"),
                metric=kwargs.get("metric", "mae"),
                num_mlp_layer=kwargs.get("num_mlp_layer", 2),
                mlp_batch_norm=self.normalization,
                mlp_dropout=kwargs.get("mlp_dropout", 0.0),
                normalization=self.normalization
            )

        elif task_type == "guidance":
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
                normalization=self.normalization,
                include_cosine=self.include_cosine,
            )
            noise_model = NoiseModel(
                timestep=kwargs.get("diffusion_steps"),
                noise_precision=kwargs.get("diffusion_noise_precision"),
                nu_arr=kwargs.get("nu_arr"),
                mapping=kwargs.get("mapping"),
            )
            self.task = GuidanceModelPrediction(
                model,
                noise_model,
                task=kwargs.get("task_learn", ""),
                include_charge=True,
                metric=kwargs.get("metric", "mae"),
                num_mlp_layer=kwargs.get("num_mlp_layer", 2),
                mlp_batch_norm=True,
                mlp_dropout=kwargs.get("mlp_dropout", 0.0),
                normalization=True,
                weight_classes=kwargs.get("weight_classes"),
                norm_values=kwargs.get("normalize_factors"),
                t_max=kwargs.get("t_max"),
            )

        else:
            raise ValueError(f"Unknown task_type '{task_type}'. Choose 'diffusion', 'property', or 'guidance'.")

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # type: ignore
        print(f"\n{'='*50}\nNumber of parameters: {n_params}\n{'='*50}\n")
        
        if self.chkpt_path:    
            chk_point = torch.load(self.chkpt_path)["model"]
    
        self.task.load_state_dict(chk_point, strict=False)
        return self.task
