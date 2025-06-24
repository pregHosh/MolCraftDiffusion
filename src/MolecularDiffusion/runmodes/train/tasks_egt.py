from MolecularDiffusion.callbacks.train_helper import SP_regularizer
from MolecularDiffusion.modules.tasks import ProperyPrediction, GuidanceModelPrediction, GeomMolecularGenerative
from MolecularDiffusion.modules.models import EGT_dynamics, GraphTransformer, NoiseModel, EnVariationalDiffusion
import torch

from tqdm import tqdm
import shutil
import glob
import pandas as pd
import wandb
import os


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
        hidden_mlp_dims: dict = {},
        hidden_dims: dict = {},
        act_fn_in: torch.nn.Module = torch.nn.SiLU(),
        act_fn_out: torch.nn.Module = torch.nn.SiLU(),
        chkpt_path: str = None,
    ):
        self.train_set = train_set
        self.atom_vocab = atom_vocab
        self.task_names = task_names
        # Common model hyperparameters
        self.hidden_mlp_dims = hidden_mlp_dims
        self.hidden_dims = hidden_dims
        self.act_fn_in = act_fn_in
        self.act_fn_out = act_fn_out


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
        dynamics_model = EGT_dynamics(
            in_edge_nf=1,
            in_global_nf=1,
            n_layers=self.num_layers,
            hidden_mlp_dims=self.hidden_mlp_dims,
            hidden_dims=self.hidden_dims,
            context_node_nf=self.context_node_nf,
            n_dims=3,
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
            model = GraphTransformer(
                in_node_nf=self.in_node_nf,
                in_edge_nf=1,
                in_global_nf=1,
                n_layers=self.num_layers,
                hidden_mlp_dims=self.hidden_mlp_dims,
                hidden_dims=self.hidden_dims,
                context_node_nf=self.context_node_nf,
                dropout=self.dropout,
                act_fn_in=self.act_fn_in,
                act_fn_out=self.act_fn_out,
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
            model = GraphTransformer(
                in_node_nf=self.in_node_nf,
                in_edge_nf=1,
                in_global_nf=1,
                n_layers=self.num_layers,
                hidden_mlp_dims=self.hidden_mlp_dims,
                hidden_dims=self.hidden_dims,
                context_node_nf=self.context_node_nf,
                dropout=self.dropout,
                act_fn_in=self.act_fn_in,
                act_fn_out=self.act_fn_out,
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




def analyze_and_save(
    model,
    epoch,
    n_samples=1000,
    batch_size=100,
    logger="logging",
    path_save="samples",
):
    DIST_THRESHOLD = 3
    DIST_RELAX_BOND = 0.25
    print(f"Analyzing molecule stability at epoch {epoch}...")
    batch_size = min(batch_size, n_samples)
    # assert n_samples % batch_size == 0
    molecules = {"one_hot": [], "x": [], "node_mask": []}
    model.max_n_nodes = 150
    for i in tqdm(range(int(n_samples / batch_size)), total=int(n_samples / batch_size)):
        nodesxsample = model.node_dist_model.sample(batch_size)
        try:
            one_hot, charges, x, node_mask = model.sample(nodesxsample=nodesxsample)
            keep = charges > 0
            keep = keep.squeeze()
            one_hot = one_hot[:, keep, :]
            charges = charges[:, keep]
            x = x[:, keep, :]
            molecules["one_hot"].append(one_hot.detach().cpu().squeeze(0))
            molecules["x"].append(x.detach().cpu().squeeze(0))
            molecules["node_mask"].append(node_mask.detach().cpu().squeeze(0))

            save_xyz_file(
                path_save,
                one_hot,
                x,
                atom_decoder=atom_vocab,
            )
            path_xyz = os.path.join(path_save, f"molecule_000.xyz")
            shutil.move(
                path_xyz,
                os.path.join(path_save, f"molecule_{str(i+1).zfill(4)}.xyz"),
            )

        except Exception as e:
            print(f"Fuck {e}")
        
    xyzs = glob.glob(f"{path_save}/*xyz")
    is_valid_strict = torch.zeros(len(xyzs), dtype=torch.float16)
    is_valid_relax = torch.zeros(len(xyzs), dtype=torch.float16)
    is_fully_connected = torch.zeros(len(xyzs), dtype=torch.float16)
    percent_atom_valids = torch.zeros(len(xyzs), dtype=torch.float16)
    for idx, xyz in enumerate(tqdm(xyzs, desc="Processing XYZ files", total=len(xyzs))):
        try:
            cartesian_coordinates_tensor, atomic_numbers_tensor = read_xyz_file(xyz)
            data = create_pyg_graph(cartesian_coordinates_tensor, atomic_numbers_tensor, r=DIST_THRESHOLD)
            data = correct_edges(data, d_relax=DIST_RELAX_BOND)
            #TODO change 
            (is_valid, percent_atom_valid, 
            num_components, bad_atoms, to_be_recheck_flag) = check_validity_v0(data, 
                                                                            angle_relax=20, 
                                                                            verbose=True)
        
            if is_valid:
                is_valid_strict[idx] = 1
                is_valid_relax[idx] = 1
            else:
                if to_be_recheck_flag:
                    is_valid_strict[idx] = 0
                    is_valid_relax[idx] = 1
                else:
                    is_valid_strict[idx] = 0
                    is_valid_relax[idx] = 0
            if num_components == 1:
                is_fully_connected[idx] = 1
            else:
                is_fully_connected[idx] = 0
            
            percent_atom_valids[idx] = percent_atom_valid
        except Exception as e:  
            print(f"Error: {e}")
            is_valid_strict[idx] = 0
            is_valid_relax[idx] = 0
            is_fully_connected[idx] = 0
            percent_atom_valids[idx] = 0
    validity_dict = {
        "Validity Strict": is_valid_strict.mean().item(),
        "Validity Relax": is_valid_relax.mean().item(),
        "Fully-connected": is_fully_connected.mean().item(),
        "Percent Atom Valid": percent_atom_valids.mean().item(),
    }

    validity_dict_save = {
        "Filename": xyzs,
        "Validity Strict": is_valid_strict,
        "Validity Relax": is_valid_relax,
        "Fully-connected": is_fully_connected,
        "Percent Atom Valid": percent_atom_valids,
    }
    
    
    df = pd.DataFrame(validity_dict_save)
    df.to_csv(f"{path_save}/validity.csv", index=False)
    

    if logger == "wandb":
        wandb.log(validity_dict)
        #    return validity_dict
    elif logger == "logging":
        for key, value in validity_dict.items():
            print(f"{key}: {value}")
    
    return validity_dict

