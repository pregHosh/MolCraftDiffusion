

from MolecularDiffusion.modules.tasks.task import Task
from MolecularDiffusion.utils.geom_utils import save_xyz_file

import logging
import os
import shutil
from typing import List
from tqdm import tqdm
import pandas as pd
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph
from MolecularDiffusion.data.component.pointcloud import PointCloud_Mol
from MolecularDiffusion.data.component.feature import onehot
from ase.data import atomic_numbers

logger = logging.getLogger(__name__)

class GenerativeFactory:
    def __init__(self,
                 task: Task,
                 task_type: str = "unconditional",
                 sampling_mode: str = "ddpm",
                 num_generate: int = 100,
                 mol_size: List[int] = [0,0],
                 target_values: List[float] = [],
                 property_names: List[str] = [],
                 negative_target_values: List[float] = [],
                 batch_size: int = 1,
                 seed: int = 86,
                 n_frames: int = 0,
                 output_path: str = "generated_mol",
                 condition_configs={},
    ):

        self.task = task
        self.task_type = task_type
        self.num_generate = num_generate
        try:
            self.max_atom = max(task.n_node_dist.keys())
        except ValueError:
            logger.warning("Node distribution model is not available, set max_atom to 86")
            self.max_atom = 86
            
        self.mol_size = mol_size

        if self.mol_size is not None and len(self.mol_size) > 0:
            if self.mol_size[-1] > self.max_atom:
                logger.info(
                    "Specified molecular size is larger than the largest molecules in the training data, reset...")
                self.mol_size[-1] = self.max_atom

        self.target_values = target_values
        self.property_names = property_names
        self.negative_target_values = negative_target_values
        
        
        if len(self.target_values) != len(self.property_names):
            logger.warning("Number of target values must match with number of property names")
            self.property_names = ["a"]*len(self.target_values)
        
        self.batch_size = batch_size
        self.seed = seed
        self.n_frames = n_frames
        
        if n_frames > 0:
            self.visualize_trajectory = True
            
            if condition_configs.get("denoising_strength", 0) > 0:
                t_start  = int(self.task.model.T * condition_configs.get("denoising_strength", 0.8))
            elif condition_configs.get("t_start", 1) < 1:
                t_start = int(self.task.model.T * condition_configs.get("t_start", 1))
            else:
                t_start = int(self.task.model.T)
                
            s_saves = torch.linspace(0, t_start, 
                                            steps=self.n_frames).long()
            self.s_saves = s_saves.flip(0)
            logger.warning(f"Frames will be saved at timesteps: {self.s_saves.tolist()}")
        else:
            self.visualize_trajectory = False
            
        self.output_path = output_path
        
        self.sampling_mode = sampling_mode # ddim not available for CFG and GG

        self.condition_configs = condition_configs

        if self.task.node_dist_model is None:
            logger.warning("Number of atoms distribution is not available, specify the size of molecules to generate")
            import random
            if len(self.mol_size) == 2:
                if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                    self.mol_size = random.randint(14, 100)
        
    def run(self):
        
        if self.task_type == "unconditional":
            self.unconditional_generation()
        elif self.task_type in ("conditional", "cfg"):
            self.conditional_generation()
        elif self.task_type in ("gradient_guidance", "cfggg"):
            self.property_guidance()
        elif self.task_type in ("inpaint", "outpaint", "outpaintft"):
            if self.batch_size > 1:
                self.batch_size = 1
                logger.warning("Structure-guided generation can be carried out in batch size of 1")
            self.structural_guidance()
        elif self.task_type in {"inpaint_cfg", "inpaint_gg", "inpaint_cfggg", "outpaint_cfg", "outpaint_gg", "outpaint_cfggg"}:
            self.hybrid_guidance()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
    def unconditional_generation(self):
            
        fail_count = 0
        
        num_round = self.num_generate // self.batch_size
        if self.num_generate % self.batch_size != 0:
            num_round += 1
        current_batch_size = self.batch_size
        
        progress_bar = tqdm(range(num_round), desc="Sampling molecules", leave=True)
        
        for i in progress_bar:
            if i == num_round-1 and self.num_generate % self.batch_size != 0:
                current_batch_size = self.num_generate % self.batch_size
            else:
                current_batch_size = self.batch_size
                
            try:
                if len(self.mol_size) == 1:
                    nodesxsample = torch.tensor(self.mol_size, dtype=torch.long)
                    nodesxsample = nodesxsample.repeat(current_batch_size) 
                elif len(self.mol_size) == 2:
                    if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                        nodesxsample = self.task.node_dist_model.sample(current_batch_size)
                    else:
                        mean = (self.mol_size[0] + self.mol_size[1]) / 2
                        std = (self.mol_size[1] - self.mol_size[0]) / 4
                        nodesxsample = torch.normal(mean=mean, std=std, size=(1,)).long()
                        nodesxsample = torch.clamp(nodesxsample, min=self.mol_size[0], max=self.mol_size[1])
                        nodesxsample = nodesxsample.repeat(current_batch_size) 
                if self.task.prop_dist_model and len(self.target_values) == 0:
                    size = nodesxsample.item()
                    target_value = self.task.prop_dist_model.sample(size)
                    one_hot, _, x, _ = self.task.sample_conditonal(
                        nodesxsample=nodesxsample, 
                        target_value=target_value,
                        mode=self.sampling_mode,
                        n_frames=self.n_frames
                    )
                else:
                    one_hot, _, x, _ = self.task.sample(
                        nodesxsample=nodesxsample,
                        mode=self.sampling_mode,
                        n_frames=self.n_frames
                    )
            
                if self.visualize_trajectory:
                    for j in range(current_batch_size):
                        mol_idx = i * self.batch_size + j
                        output_path_frame = os.path.join(self.output_path, f"mol_{mol_idx}")
                        save_xyz_file(
                            output_path_frame,
                            one_hot[:, j],
                            x[:, j],
                            atom_decoder=self.task.atom_vocab,
                            idxs=self.s_saves.tolist()
                        )            
                else:
                    save_xyz_file(
                        self.output_path,
                        one_hot,
                        x,
                        atom_decoder=self.task.atom_vocab,
                    )

                    for j in range(current_batch_size):
                        
                        path_xyz = os.path.join(self.output_path, f"molecule_{str(j).zfill(3)}.xyz")
                        idx = i * self.batch_size + j
                        shutil.move(
                            path_xyz,
                            os.path.join(self.output_path, f"molecule_{str(idx).zfill(4)}.xyz"),
                        )
            except Exception as e:
                fail_count += 1
                tqdm.write(f"[Batch {i}] Sampling failed: {e}")

            progress_bar.set_postfix({
                "completed": i + 1,
                "failed": fail_count,
                "success": (i + 1 - fail_count),
                "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
            })
    
    def conditional_generation(self):
        
        assert len(self.target_values) == len(self.task.condition); "Number of target values must match with number of conditions in the model"
        
        if hasattr(self.task, 'predictive_model'):
            property_eval = True
            df_dict = {
                "filename": [],
            }
            for prop_name in self.property_names:
                df_dict[prop_name] = []
            df_dict["size"] = []           
        else:
            logger.warning("Property model is not available, skip evaluation.")
            property_eval = False        
        
        fail_count = 0
        num_round = self.num_generate // self.batch_size
        if self.num_generate % self.batch_size != 0:
            num_round += 1
        current_batch_size = self.batch_size
        progress_bar = tqdm(range(num_round), desc="Sampling molecules", leave=True)
        
        for i in progress_bar:

            if i == num_round-1 and self.num_generate % self.batch_size != 0:
                current_batch_size = self.num_generate % self.batch_size
            else:
                current_batch_size = self.batch_size
                
            try:
                if len(self.mol_size) == 1:
                    nodesxsample = torch.tensor(self.mol_size, dtype=torch.long)
                    nodesxsample = nodesxsample.repeat(current_batch_size) 
                elif len(self.mol_size) == 2:
                    if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                        nodesxsample = self.task.node_dist_model.sample(current_batch_size)
                    else:
                        mean = (self.mol_size[0] + self.mol_size[1]) / 2
                        std = (self.mol_size[1] - self.mol_size[0]) / 4
                        nodesxsample = torch.normal(mean=mean, std=std, size=(1,)).long()
                        nodesxsample = torch.clamp(nodesxsample, min=self.mol_size[0], max=self.mol_size[1])
                        nodesxsample = nodesxsample.repeat(current_batch_size) 
                if self.task_type == "conditional":
                    one_hot, charges, x, _ = self.task.sample_conditonal(
                            nodesxsample=nodesxsample, 
                            target_value=self.target_values,
                            n_frames=self.n_frames,
                            mode=self.sampling_mode
                        )
                elif self.task_type == "cfg":
                    one_hot, charges, x, _ = self.task.sample_guidance_conitional(
                            target_function=None,
                            target_value=self.target_values,
                            negative_target_value=self.negative_target_values,
                            nodesxsample=nodesxsample, 
                            cfg_scale=self.condition_configs.get("cfg_scale",1),
                            guidance_ver="cfg",
                            n_frames=self.n_frames
                        )

                if self.visualize_trajectory:
                    for j in range(current_batch_size):
                        mol_idx = i * self.batch_size + j
                        output_path_frame = os.path.join(self.output_path, f"mol_{mol_idx}")
                        save_xyz_file(
                            output_path_frame,
                            one_hot[:, j],
                            x[:, j],
                            atom_decoder=self.task.atom_vocab,
                            idxs=self.s_saves.tolist()
                        )     
                    x = x[:, -1]
                    one_hot = one_hot[:, -1]   
                else:     
                    save_xyz_file(
                        self.output_path,
                        one_hot,
                        x,
                        atom_decoder=self.task.atom_vocab,
                    )

                    for j in range(current_batch_size):
                        
                        path_xyz = os.path.join(self.output_path, f"molecule_{str(j).zfill(3)}.xyz")
                        idx = i * self.batch_size + j
                        shutil.move(
                            path_xyz,
                            os.path.join(self.output_path, f"molecule_{str(idx).zfill(4)}.xyz"),
                        )
                
                #TODO to adapt this to work with batch of mols
                if property_eval:
                    xh = torch.cat([
                        x,
                        one_hot,
                        charges
                    ])
                    preds = self.property_prediction(xh, t=0)
                    for prop_name in self.property_names:
                        logger.info(f"{prop_name}: {preds[prop_name]}")
                        df_dict[prop_name].append(preds[prop_name])
                    df_dict["filename"].append(f"molecule_{str(i+1).zfill(4)}.xyz")
                    df_dict["size"].append(nodesxsample.item())
            except Exception as e:
                fail_count += 1
                tqdm.write(f"[Batch {i}] Sampling failed: {e}")

            progress_bar.set_postfix({
                "completed": i + 1,
                "failed": fail_count,
                "success": (i + 1 - fail_count),
                "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
            })
        if property_eval:    
            self.df = pd.DataFrame(df_dict)
    
    
    def property_guidance(self):
        
        target_function=self.condition_configs.get("target_function", None)
        target_function.atom_vocab = self.task.atom_vocab  

        target_function.norm_factor = self.task.model.norm_values
        target_function = target_function()
        scheduler = self.condition_configs.get("scheduler", None)
        if scheduler is not None:
            scheduler = scheduler()
            
        fail_count = 0
        progress_bar = tqdm(range(self.num_generate), desc="Sampling molecules", leave=True)
        
        if hasattr(self.task, 'predictive_model'):
            property_eval = True
            df_dict = {
                "filename": [],
            }
            for prop_name in self.property_names:
                df_dict[prop_name] = []
            df_dict["size"] = []           
        else:
            logger.warning("Property model is not available, skip evaluation.")
            property_eval = False
             
        for i in progress_bar:
            try:
                if len(self.mol_size) == 1:
                    nodesxsample = torch.tensor(self.mol_size, dtype=torch.long)
                elif len(self.mol_size) == 2:
                    if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                        nodesxsample = self.task.node_dist_model.sample(self.batch_size)
                    else:
                        mean = (self.mol_size[0] + self.mol_size[1]) / 2
                        std = (self.mol_size[1] - self.mol_size[0]) / 4
                        nodesxsample = torch.normal(mean=mean, std=std, size=(1,)).long()
                        nodesxsample = torch.clamp(nodesxsample, min=self.mol_size[0], max=self.mol_size[1])
                
                if len(self.target_values) == 0:
                    one_hot, charges, x, _  = self.task.sample_guidance(
                        target_function=target_function,
                        nodesxsample=nodesxsample,
                        scale=self.condition_configs.get("gg_scale",1),
                        max_norm=self.condition_configs.get("max_norm",1),
                        std=1,
                        scheduler=scheduler,
                        guidance_ver=self.condition_configs.get("guidance_ver",1),
                        guidance_at=self.condition_configs.get("guidance_at",1),
                        guidance_stop=self.condition_configs.get("guidance_stop",0),
                        n_backwards=self.condition_configs.get("n_backwards",1)
                    )              
                else:
                    
                    one_hot, charges, x, _  = self.task.sample_guidance_conitional(
                        target_function=target_function,
                        target_value=self.target_values,
                        negative_target_value=self.negative_target_values,
                        nodesxsample=nodesxsample,
                        gg_scale=self.condition_configs.get("gg_scale",1),
                        cfg_scale=self.condition_configs.get("cfg_scale",1),
                        max_norm=self.condition_configs.get("max_norm",1),
                        std=1,
                        scheduler=scheduler,
                        guidance_ver=self.condition_configs.get("guidance_ver",1),
                        guidance_at=self.condition_configs.get("guidance_at",1),
                        guidance_stop=self.condition_configs.get("guidance_stop",0),
                        n_backwards=self.condition_configs.get("n_backwards",1)
                    )   
                    
                save_xyz_file(
                    self.output_path,
                    one_hot,
                    x,
                    atom_decoder=self.task.atom_vocab,
                )

                path_xyz = os.path.join(self.output_path, f"molecule_000.xyz")
                shutil.move(
                    path_xyz,
                    os.path.join(self.output_path, f"molecule_{str(i+1).zfill(4)}.xyz"),
                )
                
                if property_eval:
                    xh = torch.cat([
                        x,
                        one_hot,
                        charges
                    ])
                    preds = self.property_prediction(xh, t=0)
                    for prop_name in self.property_names:
                        logger.info(f"{prop_name}: {preds[prop_name]}")
                        df_dict[prop_name].append(preds[prop_name])
                    df_dict["filename"].append(f"molecule_{str(i+1).zfill(4)}.xyz")
                    df_dict["size"].append(nodesxsample.item())    
            except Exception as e:
                fail_count += 1
                tqdm.write(f"[Batch {i}] Sampling failed: {e}")

            progress_bar.set_postfix({
                "completed": i + 1,
                "failed": fail_count,
                "success": (i + 1 - fail_count),
                "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
            })
            
        if property_eval:    
            self.df = pd.DataFrame(df_dict)
    
    def structural_guidance(self):
        
        # get condition structure
        xh_ref = self.preprocess_ref_structure(self.task.device)

        n_retrys = self.condition_configs.get("n_retrys")
        if n_retrys > 0 and self.n_frames:
            logger.info("No frames saved, set n_retrys = 0")
            n_retrys = 0
        
        # process condition values
        if len(self.target_values) > 0 and self.task.prop_dist_model is not None:
            
            context = []
            for i, key in enumerate(self.task.prop_dist_model.distributions):

                if self.task.normalize_condition == "mad":
                    mean, mad = (
                        self.task.prop_dist_model.normalizer[key]["mean"],
                        self.task.prop_dist_model.normalizer[key]["mad"],
                    )
                    val = (self.target_values[i] - mean) / (mad)
                elif self.task.normalize_condition == "maxmin":   
                    mean, min, max = (
                        self.task.prop_dist_model.normalizer[key]["mean"],
                        self.task.prop_dist_model.normalizer[key]["min"],
                        self.task.prop_dist_model.normalizer[key]["max"],
                    )
                    val = 2 * (self.target_values[i] - min) / (max - min) - 1           
                else:
                    val = self.target_values[i]
                context_row = torch.tensor(
                        [val]
                ).unsqueeze(1)
                context.append(context_row)
            context = torch.cat(context, dim=1).float().to(self.task.device)

        else:
            context = None
        
    
        fail_count = 0
        progress_bar = tqdm(range(self.num_generate), desc="Sampling molecules", leave=True)
        
        condition_mode = self.task_type + "_" + self.condition_configs.get("condition_component",  "xh")
        
        for i in progress_bar:
            try:
                if len(self.mol_size) == 1:
                    nodesxsample = torch.tensor(self.mol_size, dtype=torch.long)
                elif len(self.mol_size) == 2:
                    if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                        nodesxsample = self.task.node_dist_model.sample(self.batch_size)
                    else:
                        mean = (self.mol_size[0] + self.mol_size[1]) / 2
                        std = (self.mol_size[1] - self.mol_size[0]) / 4
                        nodesxsample = torch.normal(mean=mean, std=std, size=(1,)).long()
                        nodesxsample = torch.clamp(nodesxsample, min=self.mol_size[0], max=self.mol_size[1])                

                if self.task_type == "inpaint":
                    try:
                        mask_node_index = torch.tensor([self.condition_configs.get("mask_node_index", [])])
                    except RuntimeError:
                        mask_node_index = torch.tensor([[]])
                    if nodesxsample.item() < xh_ref.shape[1]:
                        nodesxsample = torch.tensor([xh_ref.shape[1]])
                        logging.warning("Specified molecular size is too small, set it as the same size as the reference structure")
                        
                    one_hot, charges, x, node_mask = self.task.sample(
                        nodesxsample,
                        condition_tensor=xh_ref,
                        condition_mode=condition_mode,
                        denoising_strength=self.condition_configs.get("denoising_strength", 0.8),
                        t_start=self.condition_configs.get("t_start", 1),
                        t_critical_1=self.condition_configs.get("t_critical_1"),
                        t_critical_2=self.condition_configs.get("t_critical_2"),
                        d_threshold_f=self.condition_configs.get("d_threshold_f"),
                        w_b=self.condition_configs.get("w_b"),
                        all_frozen=self.condition_configs.get("all_frozen"),
                        use_covalent_radii=self.condition_configs.get("use_covalent_radii"),
                        scale_factor=self.condition_configs.get("scale_factor"),
                        noise_initial_mask=self.condition_configs.get("noise_initial_mask"),
                        mask_node_index=mask_node_index,    
                        n_frames=self.n_frames,
                        n_retrys=self.condition_configs.get("n_retrys"),
                        t_retry=self.condition_configs.get("t_retry"),
                        context=context,
                    )
                
                elif self.task_type == "outpaint":
                    one_hot, charges, x, node_mask = self.task.sample(
                        nodesxsample,
                        condition_tensor=xh_ref,
                        condition_mode=condition_mode,
                        t_start=self.condition_configs.get("t_start", 1),
                        t_critical_1=self.condition_configs.get("t_critical_1"),
                        t_critical_2=self.condition_configs.get("t_critical_2"),
                        d_threshold_f=self.condition_configs.get("d_threshold_f"),
                        w_b=self.condition_configs.get("w_b"),
                        all_frozen=self.condition_configs.get("all_frozen"),
                        use_covalent_radii=self.condition_configs.get("use_covalent_radii"),
                        scale_factor=self.condition_configs.get("scale_factor"),
                        connector_dicts=self.condition_configs.get("connector_dicts"),
                        n_frames=self.n_frames,
                        n_retrys=self.condition_configs.get("n_retrys"),
                        t_retry=self.condition_configs.get("t_retry"),
                        context=context,
                    )
                elif self.task_type == "outpaintft":
                    one_hot, charges, x, node_mask = self.task.sample(
                        nodesxsample,
                        condition_tensor=xh_ref,
                        condition_mode=condition_mode,
                        t_start=self.condition_configs.get("t_start", 1),
                        n_frames=self.n_frames,
                        n_retrys=self.condition_configs.get("n_retrys"),
                        t_retry=self.condition_configs.get("t_retry"),
                        context=context,
                    )

                if self.visualize_trajectory:   
                    output_path_frame = os.path.join(self.output_path, f"mol_{i}")
                    # assume batch_size = 1
                    save_xyz_file(
                        output_path_frame,
                        one_hot[:, 0],
                        x[:, 0],
                        atom_decoder=self.task.atom_vocab,
                        idxs=self.s_saves.tolist()
                    )     
                    x = x[:, -1]
                    one_hot = one_hot[:, -1]   
                else:     
                    save_xyz_file(
                        self.output_path,
                        one_hot,
                        x,
                        atom_decoder=self.task.atom_vocab,
                    )
                    path_xyz = os.path.join(self.output_path, f"molecule_000.xyz")
                    shutil.move(
                        path_xyz,
                        os.path.join(self.output_path, f"molecule_{str(i).zfill(4)}.xyz"),
                    )             
                             
            except Exception as e:
                fail_count += 1
                tqdm.write(f"[Batch {i}] Sampling failed: {e}")

            progress_bar.set_postfix({
                "completed": i + 1,
                "failed": fail_count,
                "success": (i + 1 - fail_count),
                "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
            })   


    def hybrid_guidance(self):
        
        xh_ref = self.preprocess_ref_structure(self.task.device)
        condition_mode = self.task_type.split("_")[0] + "_" + self.condition_configs.get("condition_component",  "xh")
        
        guidance_ver = self.condition_configs.get("guidance_ver", "cfg")
        if guidance_ver in [0,1,2,"cfggg"]:

            target_function=self.condition_configs.get("target_function", None)            
            if target_function is None:
                raise ValueError("Target function must be provided for gradient guidance")
            else:
                target_function.atom_vocab = self.task.atom_vocab      
                target_function.norm_factor = self.task.model.norm_values
                target_function = target_function()
        else:
            target_function = None
                
        scheduler = self.condition_configs.get("scheduler", None)
        if scheduler is not None:
            scheduler = scheduler()
            
        fail_count = 0
        num_round = self.num_generate // self.batch_size
        if self.num_generate % self.batch_size != 0:
            num_round += 1
        current_batch_size = self.batch_size
        progress_bar = tqdm(range(num_round), desc="Sampling molecules", leave=True)
        
        
        if hasattr(self.task, 'predictive_model'):
            property_eval = True
            df_dict = {
                "filename": [],
            }
            for prop_name in self.property_names:
                df_dict[prop_name] = []
            df_dict["size"] = []           
        else:
            logger.warning("Property model is not available, skip evaluation.")
            property_eval = False
             
        for i in progress_bar:
            if i == num_round-1 and self.num_generate % self.batch_size != 0:
                current_batch_size = self.num_generate % self.batch_size
            else:
                current_batch_size = self.batch_size
            try:
                if len(self.mol_size) == 1:
                    nodesxsample = torch.tensor(self.mol_size, dtype=torch.long)
                    nodesxsample = nodesxsample.repeat(current_batch_size) 
                elif len(self.mol_size) == 2:
                    if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                        nodesxsample = self.task.node_dist_model.sample(current_batch_size)
                    else:
                        mean = (self.mol_size[0] + self.mol_size[1]) / 2
                        std = (self.mol_size[1] - self.mol_size[0]) / 4
                        nodesxsample = torch.normal(mean=mean, std=std, size=(1,)).long()
                        nodesxsample = torch.clamp(nodesxsample, min=self.mol_size[0], max=self.mol_size[1])


                if "inpaint" in self.task_type:
                    if nodesxsample[0].item() < xh_ref.shape[1]:
                        nodesxsample = torch.tensor([xh_ref.shape[1]]*current_batch_size)
                        logging.warning("Specified molecular size is too small, set it as the same size as the reference structure")
                            
                    try:
                        mask_node_index = torch.tensor([self.condition_configs.get("mask_node_index", [])])
                    except RuntimeError:
                        mask_node_index = torch.tensor([[]])
                else:
                    mask_node_index = torch.tensor([[]])

                xh_tensor = xh_ref.repeat(current_batch_size, 1, 1)
                one_hot, charges, x, _  = self.task.sample_hybrid_guidance(
                    target_function=target_function,
                    target_value=self.target_values,
                    negative_target_value=self.negative_target_values,
                    nodesxsample=nodesxsample,
                    gg_scale=self.condition_configs.get("gg_scale",1),
                    cfg_scale=self.condition_configs.get("cfg_scale",1),
                    max_norm=self.condition_configs.get("max_norm",1),
                    std=1,
                    scheduler=scheduler,
                    guidance_ver=guidance_ver,
                    guidance_at=self.condition_configs.get("guidance_at",1),
                    guidance_stop=self.condition_configs.get("guidance_stop",0),
                    n_backwards=self.condition_configs.get("n_backwards",1),
                    condition_tensor=xh_tensor,
                    condition_mode=condition_mode,
                    mask_node_index=mask_node_index, # For Inpainting
                    denoising_strength=self.condition_configs.get("denoising_strength", 0.8), # For Inpainting
                    noise_initial_mask=self.condition_configs.get("noise_initial_mask", False), # For Inpainting
                    t_start=self.condition_configs.get("t_start", 1),
                    t_critical=self.condition_configs.get("t_critical", 0),
                    n_frames=self.n_frames,
                )   
                
                if self.visualize_trajectory:
                    for j in range(current_batch_size):
                        mol_idx = i * self.batch_size + j
                        output_path_frame = os.path.join(self.output_path, f"mol_{mol_idx}")
                        save_xyz_file(
                            output_path_frame,
                            one_hot[:, j],
                            x[:, j],
                            atom_decoder=self.task.atom_vocab,
                            idxs=self.s_saves.tolist()
                        )     
                    x = x[:, -1]
                    one_hot = one_hot[:, -1]   
                else:     
                    save_xyz_file(
                        self.output_path,
                        one_hot,
                        x,
                        atom_decoder=self.task.atom_vocab,
                    )

                    for j in range(current_batch_size):
                        
                        path_xyz = os.path.join(self.output_path, f"molecule_{str(j).zfill(3)}.xyz")
                        idx = i * self.batch_size + j
                        shutil.move(
                            path_xyz,
                            os.path.join(self.output_path, f"molecule_{str(idx).zfill(4)}.xyz"),
                        )
            
                if property_eval:
                    xh = torch.cat([
                        x,
                        one_hot,
                        charges
                    ])
                    preds = self.property_prediction(xh, t=0)
                    for prop_name in self.property_names:
                        logger.info(f"{prop_name}: {preds[prop_name]}")
                        df_dict[prop_name].append(preds[prop_name])
                    df_dict["filename"].append(f"molecule_{str(i+1).zfill(4)}.xyz")
                    df_dict["size"].append(nodesxsample.item())    
            except Exception as e:
                fail_count += 1
                tqdm.write(f"[Batch {i}] Sampling failed: {e}")

            progress_bar.set_postfix({
                "completed": i + 1,
                "failed": fail_count,
                "success": (i + 1 - fail_count),
                "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
            })
            
        if property_eval:    
            self.df = pd.DataFrame(df_dict)
    
                   
    def property_prediction(self, 
                            xh: torch.Tensor, # pos, node_feature
                            t: int):
        DIM = 3; RADIUS = 4
        bs, n_nodes, _ = xh.shape
        
        mol = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mol = {}
        coords = xh[:, :, :DIM].view(n_nodes*bs, DIM).to(device)
        h = xh[:, :, DIM:-1].view(n_nodes*bs, -1).to(device)
        charge = xh[:, :, -1].view(n_nodes*bs).to(device)

        edge_index = radius_graph(coords, r=RADIUS)
        tags = torch.zeros(n_nodes, dtype=torch.long, device=device)
        
        times = torch.zeros(n_nodes, dtype=torch.float32, device=device) + t.item()
        times = times.view(n_nodes, 1)

        graph_data = Data(
                            x=h,
                            pos=coords,
                            atomic_numbers=charge,
                            natoms=n_nodes,
                            smiles=None,
                            xyz=None,
                            edge_index=edge_index,
                            tags=tags,
                            times=times,
                        )
        graph_data = Batch.from_data_list([graph_data])
        mol["graph"] = graph_data
        preds = self.task.predictive_model.predict(mol, evaluate=True)[0]
        return preds
    

    def preprocess_ref_structure(self, device):
        """
        Load and preprocess a reference molecular structure from an XYZ file.
        
        This function reads an XYZ file, encodes atomic features, normalizes
        coordinates and features, and returns a tensor combining positions
        and processed features.

        Returns:
            torch.Tensor: Tensor of shape (1, n_atoms, 3 + n_features + 1) containing:
                        [normalized_coords | normalized_onehot_features | normalized_charges],
                        or `None` if the file path does not exist.
        """
        if not os.path.exists(self.condition_configs.get("reference_structure_path", None)):
            return None

        # Load molecule with hydrogen atoms
        mol = PointCloud_Mol.from_xyz(
            self.condition_configs.get("reference_structure_path"), with_hydrogen=True, forbidden_atoms=[]
        )
        
        # Extract atomic coordinates and number of atoms
        coords = mol.get_coord()
        n_atoms = len(mol.atoms)

        # One-hot encode atomic types
        atom_vocab = self.task.atom_vocab
        node_features = [
            onehot(atom.element, atom_vocab, allow_unknown=False)
            for atom in mol.atoms
        ]

        # Atomic numbers (or model-specific charge encoding)
        charges = [atomic_numbers[atom.element]
                    for atom in mol.atoms
                    if atom.element in atomic_numbers]
        
        # Normalization
        normalize_coords, normalize_feats, normalize_charges = self.task.model.norm_values
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32).view(1, n_atoms, 3) / normalize_coords
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32).view(1, n_atoms, -1) / normalize_feats
        charges_tensor = torch.tensor(charges, dtype=torch.float32).view(1, n_atoms, 1) / normalize_charges

        # Concatenate features: [coords | onehot | charges]
        features = torch.cat([node_features_tensor, charges_tensor], dim=-1)
        xh_ref = torch.cat([coords_tensor, features], dim=-1).to(device)

        return xh_ref