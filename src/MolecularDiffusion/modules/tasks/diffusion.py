import logging
import math
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from ase.data import chemical_symbols
from rdkit import Chem

from MolecularDiffusion import core
from MolecularDiffusion.callbacks import SP_regularizer
from MolecularDiffusion.modules.layers import common, functional
from MolecularDiffusion.modules.models.en_diffusion import (
    DistributionNodes,
    DistributionProperty,
)
from MolecularDiffusion.utils import (
    assert_correctly_masked,
    assert_mean_zero_with_mask,
    check_mask_correct,
    check_stability,
    compute_mean_mad_from_dataloader,
    prepare_context,
    random_rotation,
    remove_mean_pyG,
    remove_mean_with_mask,
    sample_center_gravity_zero_gaussian_with_mask,
    sample_gaussian_with_mask,
)

from .task import Task, _get_criterion_name, _get_metric_name


logger = logging.getLogger(__name__)


@core.Registry.register("GeomMolecularGenerative")
class GeomMolecularGenerative(Task, core.Configurable):

    _option_members = {"task"}

    def __init__(
        self,
        diffusion_model,
        node_dist_model=None,
        prop_dist_model=None,
        n_node_dist: Dict = {},
        augment_noise: float = 0,
        data_augmentation: bool = False,
        condition: List = [],
        normalize_condition: str = None,
        sp_regularizer: SP_regularizer = None,
        reference_indices: List = None, # For OP task
    ):
        """
        Generative Diffusion model for molecular structures.
        Parameters:
        - diffusion_model: The dynamic functional model for diffusion.
        - node_dist_model (Optional[NodeDistributionModel]): The model for number of node distribution. Default is None.
        - prop_dist_model (Optional[PropertyDistributionModel]): The model for property distribution. Default is None.
        - n_node_dist (Dict): The distribution of number of nodes. Default is {}.
        - augment_noise (float): The amount of noise to add to the coordinates for data augmentation. Default is 0.
        - data_augmentation (bool): Whether to apply data augmentation by symmetry operations. Default is False.
        - condition (List): The list of conditions for the model. Default is [].
        - normalize_condition (str): The normalization method for the condition. Default is None. [None, "maxmin", "mad"]
        - sp_regularizer (SP_regularizer): The self-pace learning regularizer for the model. Default is None.
        """
        super(GeomMolecularGenerative, self).__init__()
        self.model = diffusion_model
        self.node_dist_model = node_dist_model
        self.prop_dist_model = prop_dist_model
        self.n_node_dist = n_node_dist
        self.augment_noise = augment_noise
        self.data_augmentation = data_augmentation
        self.condition = condition
        self.sp_regularizer = sp_regularizer
        self.reference_indices = reference_indices
        self.normalize_condition = normalize_condition
        
        
    def preprocess(
        self,
        train_set=None,
    ):
        if train_set is not None:
            self.atomic_numbers = train_set.atom_types()
            self.atom_decoder = [
                chemical_symbols[number]
                for number in self.atomic_numbers
                if number < len(chemical_symbols)
            ]

            self.atom_encoder = {symbol: i for i, symbol in enumerate(self.atom_decoder)}

        
            self.dataset_smiles_list = []
            if "graph" in train_set[0]:
                self.max_n_nodes = 0
                self.n_node_dist = {}
                for sample in train_set:
                    self.dataset_smiles_list.append(sample["graph"].smiles)
                    n_node = sample["graph"].natoms
                    if n_node > self.max_n_nodes:
                        self.max_n_nodes = n_node
                    if n_node in self.n_node_dist:
                        self.n_node_dist[n_node] += 1
                    else:
                        self.n_node_dist[n_node] = 1    
                        
            else:

                self.max_n_nodes = train_set[0]["coords"].size()[0]
                
                # train_set = tqdm(train_set, "Determining number of atoms distibution")
                if not self.n_node_dist:
                    self.n_node_dist = {}
                    for sample in train_set:
                        node_mask = sample["node_mask"].nonzero()
                        n_node = node_mask.size(0)
                        if n_node in self.n_node_dist:
                            self.n_node_dist[n_node] += 1
                        else:
                            self.n_node_dist[n_node] = 1

            if self.node_dist_model is None:
                print("---------------Creating node distribution model-----------------")
                self.node_dist_model = DistributionNodes(self.n_node_dist)

            if (self.prop_dist_model is None) and len(self.condition) > 0:
                print("---------------Creating property distribution model---------------")
                num_atoms = train_set.num_atoms
                props = []
                for task in self.condition:
                    props.append(train_set.get_property(task))
                props = torch.stack(props)
                self.prop_dist_model = DistributionProperty(
                    num_atoms, props, self.condition, num_bins=10
                )

                self.property_norms = compute_mean_mad_from_dataloader(
                    props, self.condition
                )
                self.prop_dist_model.set_normalizer(self.property_norms)


            if len(self.dataset_smiles_list) == 0:
                smiles_list = train_set.smiles_list
                canonical_smiles = []
                for smiles in smiles_list:
                    if smiles is None:
                        continue
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles.append(Chem.MolToSmiles(mol))
                self.dataset_smiles_list = list(set(canonical_smiles))
        else:
            self.atomic_numbers = []
            self.atom_decoder = []
            self.atom_encoder = {}
            self.dataset_smiles_list = []
            self.max_n_nodes = 0
            self.n_node_dist = {}


    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        _loss, _metric = self.density_estimation(batch)
        all_loss += _loss
        metric.update(_metric)

        return all_loss, metric

    def density_estimation(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        
        if "graph" in batch.keys():
            # NOTE ignore remove mean over mask for now
            x = None
            h = None
            if len(self.condition) > 0:
                #TODO to write this
                context = prepare_context_pyG(self.condition, batch, self.property_norms, 
                                        normalization_method=self.normalize_condition).to(
                    dtype=torch.float32, device=self.device
                )
                assert_correctly_masked(context, node_mask)
            else:
                context = None
            batch["graph"].pos = remove_mean_pyG(batch["graph"].pos, batch["graph"].batch)
            nll = self.model(x=None, h=None, mol_graph=batch, reference_indices=self.reference_indices)
            N = batch["graph"].natoms
        else:
            node_mask = batch["node_mask"].unsqueeze(2)
            edge_mask = batch["edge_mask"]
            x = batch["coords"]
            h = batch["node_feature"]
            charges = batch["charges"].unsqueeze(2)
            

            x = remove_mean_with_mask(x, node_mask)
            if self.augment_noise > 0 and not(self.reference_indices):
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(
                    x.size(), x.device, node_mask
                )
                x = x + eps * self.augment_noise
                x = remove_mean_with_mask(x, node_mask)
            if self.data_augmentation:
                x = random_rotation(x).detach()

            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            bs, n_nodes, n_dims = x.size()
            assert_correctly_masked(x, node_mask)
            edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
            h = {"categorical": h, "integer": charges}

            if len(self.condition) > 0:
                context = prepare_context(self.condition, batch, self.property_norms, 
                                        normalization_method=self.normalize_condition).to(
                    dtype=torch.float32, device=self.device
                )
                assert_correctly_masked(context, node_mask)
            else:
                context = None
            nll = self.model(x, h, node_mask, edge_mask, context, reference_indices=self.reference_indices)

            N = node_mask.squeeze(2).sum(1).long()
        log_pN = self.node_dist_model.log_prob(N)
        assert nll.size() == log_pN.size()
        # print(nll, log_pN)
        nll = nll - log_pN

        if self.sp_regularizer is not None and self.training:
            nll = self.sp_regularizer(nll)
        # loss = nll
        loss = nll.mean(0)
        metric["train negative log likelihood"] = loss
        all_loss += loss

        return all_loss, metric

    def predict_and_target(self, batch):
        all_loss = self._evaluate(batch)
        all_loss = all_loss.unsqueeze(0)
        dummy_tensor = torch.zeros_like(all_loss)
        return all_loss, dummy_tensor

    def evaluate(self, all_loss, dummy_tensor):
        metric = {}
        metric["Val negative log likelihood"] = all_loss.mean()
        return metric


    def _evaluate(self, batch):
        pred = None

        if "graph" in batch.keys():
            # NOTE ignore remove mean over mask for now
            x = None
            h = None
            if len(self.condition) > 0:
                #TODO to write this
                context = prepare_context_pyG(self.condition, batch, self.property_norms, 
                                        normalization_method=self.normalize_condition).to(
                    dtype=torch.float32, device=self.device
                )
                assert_correctly_masked(context, node_mask)
            else:
                context = None
            nll = self.model(x=None, h=None, mol_graph=batch, reference_indices=self.reference_indices)
            N = batch["graph"].natoms
        else:
            node_mask = batch["node_mask"].unsqueeze(2)
            edge_mask = batch["edge_mask"]
            x = batch["coords"]
            h = batch["node_feature"]
            charges = batch["charges"].unsqueeze(2)
                        

            if self.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(
                    x.size(), x.device, node_mask
                )
                x = x + eps * self.augment_noise
    
            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)
            bs, n_nodes, n_dims = x.size()
            assert_correctly_masked(x, node_mask)
            edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
            h = {"categorical": h, "integer": charges}

            if len(self.condition) > 0:
                context = prepare_context(self.condition, batch, self.property_norms, 
                                        normalization_method=self.normalize_condition).to(
                    dtype=torch.float32, device=self.device
                )
                assert_correctly_masked(context, node_mask)
            else:
                context = None
            N = node_mask.squeeze(2).sum(1).long()
  
            nll = self.model(x, h, node_mask, edge_mask, context, 
                            reference_indices=self.reference_indices)

        log_pN = self.node_dist_model.log_prob(N)
        assert nll.size() == log_pN.size()
        nll = nll - log_pN
        loss = nll.mean(0)

        return loss



    def sample_chain(self, n_nodes: int, n_tries: int, keep_frames: int = 100):
        """
        Sample a molecule for visualizing the diffusion process.

        Parameters:
        - n_nodes (int): Number of nodes in the molecular graph.
        - n_tries (int): Number of attempts to find a stable molecule.
        - keep_frames (int): Number of frames to keep. Default is 100.

        Returns:
        Tuple[Tensor, Tensor, Tensor]: One-hot encoding of atoms, charges, and positions.
        """
        N_SAMPLE = 1
        if (len(self.condition) > 0) and (self.prop_dist_model is not None):
            context = self.prop_dist_model.sample(n_nodes).unsqueeze(1).unsqueeze(0)
            context = context.repeat(1, n_nodes, 1).to(self.device)
        else:
            context = None

        node_mask = torch.ones(N_SAMPLE, n_nodes, 1).to(self.device)

        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        edge_mask = edge_mask.repeat(N_SAMPLE, 1, 1).view(-1, 1).to(self.device)

        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = self.model.sample_chain(
                N_SAMPLE,
                n_nodes,
                node_mask,
                edge_mask,
                context,
                keep_frames=keep_frames,
            )
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()

            zs = [self.atomic_numbers[i] for i in atom_type]

            mol_stable = check_stability(x_squeeze, zs, self.atom_decoder)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(
                torch.argmax(one_hot, dim=2), num_classes=len(self.atom_decoder)
            )
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print("ლ(́◉◞౪◟◉‵ლ) Found stable molecule to visualize -(๑☆‿ ☆#)ᕗ")
                break
            elif i == n_tries - 1:
                print("Did not find stable molecule, showing last sample. ༼ಢ_ಢ༽")

        return one_hot, charges, x


    def sample(self, 
               nodesxsample=torch.tensor([10]),
               context=None, 
               condition_tensor=None,
               condition_mode=None,
               fix_noise=False,
               n_frames=0,
               n_retrys=0,
               t_retry=180,
               t_start=1,
               mode="ddpm",
               **kwargs):
        """
        Sample molecular structures.

        Parameters:
        - nodesxsample (Tensor): Number of nodes per sample.
        - context (Optional[Tensor]): Context tensor for sampling. Default is None.
        - condition_tensor (Optional[Tensor]): Condition tensor for sampling. Default is None.
            Note that it has to be normalized the same way as the training set.
            Size = [batch size, n_atom, n_features]
        - condition_mode (Optional[str]): Mode for conditioning. Default is None.
            Format: [condition_name]_[component_alg]
            component name can be x, h, or xh
            component_alg: SSGD, ...    
        - fix_noise (bool): Fix noise for visualization purposes. Default is False.
        - n_frames (int): Number of frames to keep. Default is 0.
        - n_retrys (int): Number of retry attempts in the event of bad molecules . Default is 0.
        - t_retrys (int): Timestep to start retrying. Default is 180.
        - mode (str): Mode for sampling. Default is "ddpm ["ddpm", "ddim"].

        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: One-hot encoding of atoms, charges, positions, and node mask.
        """
        # assert int(torch.max(nodesxsample)) <= self.max_n_nodes
        # nodesxsample = torch.where(
        #     nodesxsample > self.max_n_nodes, self.max_n_nodes, nodesxsample
        # )
        batch_size = nodesxsample.size(0)

        if batch_size > 1:
            node_mask = torch.zeros(batch_size, self.max_n_nodes)
            nnode = self.max_n_nodes
        else:
            nnode = int(nodesxsample[0])
            node_mask = torch.zeros(batch_size, nnode)

        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * nnode * nnode, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)

        if (len(self.condition) > 0) and (self.prop_dist_model is not None):
            if context is None:
                context = self.prop_dist_model.sample_batch(nodesxsample)
            context = (
                context.unsqueeze(1).repeat(1, nnode, 1).to(self.device) * node_mask
            )
        else:
            context = None

        if mode == "ddpm":
            x, h, chain = self.model.sample(
                batch_size,
                nnode,
                node_mask,
                edge_mask,
                context,
                condition_tensor,
                condition_mode,
                fix_noise=fix_noise,
                n_frames=n_frames,
                n_retrys=n_retrys,
                t_retry=t_retry,
                t_start=t_start,
                **kwargs
            )
        elif mode == "ddim":    
            x, h, chain = self.model.sample_ddim(
                batch_size,
                nnode,
                node_mask,
                edge_mask,
                context,
                # condition_tensor,
                # condition_mode,
                fix_noise=fix_noise,
                # n_frames=n_frames,
                # n_retrys=n_retrys,
                # t_retry=t_retry,
                **kwargs # eta, n_steps, save_frame
            )

        
        if chain is not None:
            # Prepare entire chain.
            if isinstance(chain, torch.Tensor):
                x = chain[:, :, 0:3]
                one_hot = chain[:, :, 3:-1]
                one_hot = F.one_hot(
                    torch.argmax(one_hot, dim=2), num_classes=len(self.atom_decoder)
                )
                charges = torch.round(chain[:, :, -1:]).long()
            elif isinstance(chain, list):
                x_0 = chain[0][:, :, 0:3]
                one_hot_0 = chain[0][:, :, 3:-1]
                one_hot_0 = F.one_hot(
                    torch.argmax(one_hot_0, dim=2), num_classes=len(self.atom_decoder)
                )
                charges_0 = torch.round(chain[0][:, :, -1:]).long()
                
                x_retrys = []
                one_hot_retrys = []
                charges_retrys = []
                for i in range(chain[1].shape[0]):
                    x_i = chain[1][i][:, :, 0:3]
                    one_hot_i = chain[1][i][:, :, 3:-1]
                    one_hot_i = F.one_hot(
                        torch.argmax(one_hot_i, dim=2), num_classes=len(self.atom_decoder)
                    )
                    charges_i = torch.round(chain[1][i][:, :, -1:]).long()
                    x_retrys.append(x_i)
                    one_hot_retrys.append(one_hot_i)
                    charges_retrys.append(charges_i)
                one_hot = [one_hot_0, one_hot_retrys]
                charges = [charges_0, charges_retrys]
                x = [x_0, x_retrys]
        else:
            one_hot = h["categorical"]
            charges = h["integer"]
        return one_hot, charges, x, node_mask




    def sample_around_xh_target(self, nodesxsample=torch.tensor([10]), 
                                xh_target=None, context=None, fix_noise=False):
                            
        """
        Sample molecular structures.

        Parameters:
        - nodesxsample (Tensor): Number of nodes per sample.
        - xh_target (Tensor): target xh: [batch size, n_atom, n_features]
        - context (Optional[Tensor]): Context tensor for sampling. Default is None.
        - fix_noise (bool): Fix noise for visualization purposes. Default is False.

        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: One-hot encoding of atoms, charges, positions, and node mask.
        """
        # assert int(torch.max(nodesxsample)) <= self.max_n_nodes
        nodesxsample = torch.where(
            nodesxsample > self.max_n_nodes, self.max_n_nodes, nodesxsample
        )
        batch_size = len(nodesxsample)

        if batch_size > 1:
            node_mask = torch.zeros(batch_size, self.max_n_nodes)
            nnode = self.max_n_nodes
        else:
            nnode = int(nodesxsample[0])
            node_mask = torch.zeros(batch_size, nnode)

        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * nnode * nnode, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)

        if (len(self.condition) > 0) and (self.prop_dist_model is not None):
            if context is None:
                context = self.prop_dist_model.sample_batch(nodesxsample)
            context = (
                context.unsqueeze(1).repeat(1, nnode, 1).to(self.device) * node_mask
            )
        else:
            context = None

        x, h = self.model.sample_around_xh(
            batch_size,
            nnode,
            node_mask,
            edge_mask,
            context,
            xh_target,
            fix_noise=fix_noise,
        )

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h["categorical"]
        charges = h["integer"]

        assert_correctly_masked(one_hot.float(), node_mask)
        assert_correctly_masked(charges.float(), node_mask)

        return one_hot, charges, x, node_mask
    
    def sample_conditonal(
        self,
        nodesxsample=torch.tensor([10]),
        target_value=[0],
        fix_noise=False,
        mode="ddpm",
    ):
        """
        Sample molecular structures conditioned on a property value.
        Only works if the model is trained with a property distribution.

        The interval should be wider than the bin width of the property distribution.
        If the interval is too narrow, the model might just get the same molecule.

        Parameters:
        - nodesxsample (Tensor): Number of nodes per sample.
        - target_value (List[float]): Target values for conditional sampling.
        - fix_nose (bool): Fix noise for visualization purposes. Default is False.
        - mode (str): Mode for sampling. Default is "ddpm ["ddpm", "ddim"].
        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: One-hot encoding of atoms, charges, positions, and node mask.
        """
  

        context = []
        for i, key in enumerate(self.prop_dist_model.distributions):
            if self.normalize_condition is not None:
                if self.normalize_condition == "mad":
                    mean, mad = (
                        self.prop_dist_model.normalizer[key]["mean"],
                        self.prop_dist_model.normalizer[key]["mad"],
                    )
                    val = (target_value[i] - mean) / (mad)
                elif self.normalize_condition == "maxmin":   
                    mean, min, max = (
                        self.prop_dist_model.normalizer[key]["mean"],
                        self.prop_dist_model.normalizer[key]["min"],
                        self.prop_dist_model.normalizer[key]["max"],
                    )
                    val = 2 * (target_value[i] - min) / (max - min) - 1    
                elif "value" in self.normalize_condition: # "value_n where n is the value to normalize"
                    value = float(self.normalize_condition.split("_")[1])
                    val = target_value[i] / value
                else:
                    raise ValueError(f"Unknown normalization method: {self.normalize_condition}")
                   
            else:
                val = target_value[i]
            context_row = torch.tensor(
                 [val]
            ).unsqueeze(1)
            context.append(context_row)
        context = torch.cat(context, dim=1).float().to(self.device)
        one_hot, charges, x, node_mask = self.sample(nodesxsample, context, fix_noise, mode=mode)
        return one_hot, charges, x, node_mask

    def sample_guidance(
        self,
        target_function,
        nodesxsample=torch.tensor([10]),
        scale=1,
        max_norm=10,
        std=1.0,
        fix_noise=False,
        scheduler=None,
        guidance_at=0,
        guidance_stop=1,    
        guidance_ver=1,
        n_backwards=0,
        h_weight=1,
        x_weight=1,
        debug=False,
    ):
        """
        Sample molecular structures with guidance from target function.

        Parameters:
        - target_function (Callable[[Tensor], Tensor]): Target function for guidance. Higher value, better
        - nodesxsample (Tensor): Number of nodes per sample. Default is torch.tensor([10]).
        - scale (float): Scale factor for gradient guidance. Default is 1.0.
        - max_norm (float): Initial maximum norm for the gradients. Default is 10.0.
        - std (float): Standard deviation of the noise. Default is 1.0.
        - fix_noise (bool): Fix noise for visualization purposes. Default is False.
        - scheduler (RateScheduler): Rate scheduler. Default is None.
            The scheduler should have a step method that takes the energy and the current scale as input.
        - guidance_at (int): The timestep at which to apply guidance [0-1]  0 = since beginning. Default is 1.
        - guidance_stop (int): The timestep at which to stop applying guidance [0-1]  1 = until the end. Default is 0.  
        - guidance_ver (int): The version of the guidance. Default is 1. [0,1,2,cfg,cfg_gg]
        - n_backwards (int): Number of backward steps. Default is 0.
        - h_weight (float): Weight for the gradient of atom feature. Default is 1.0.
        - x_weight (float): Weight for the gradient of cartesian coordinate. Default is 1.0.
        - debug (bool): Debug mode. Default is False.
            Save gradient norms, max gradients, clipping coefficients, and energies to files.

        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Positions, one-hot encoding of atoms, node mask, and edge mask.
        """
        # assert int(torch.max(nodesxsample)) <= self.max_n_nodes
        # nodesxsample = torch.where(
        #     nodesxsample > self.max_n_nodes, self.max_n_nodes, nodesxsample
        # )
        batch_size = nodesxsample.size(0)
        if batch_size > 1:
            node_mask = torch.zeros(batch_size, self.max_n_nodes)
            nnode = self.max_n_nodes
        else:
            nnode = int(nodesxsample[0])
            node_mask = torch.zeros(batch_size, nnode)

        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * nnode * nnode, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)


        # sample from the EDM model
        x, h = self.model.sample_guidance(
            batch_size,
            target_function,
            node_mask,
            edge_mask,
            None,
            gg_scale=scale,
            max_norm=max_norm,
            fix_noise=fix_noise,
            std=std,
            scheduler=scheduler,
            guidance_at=guidance_at,
            guidance_stop=guidance_stop,
            guidance_ver=guidance_ver,
            n_backwards=n_backwards,
            h_weight=h_weight,
            x_weight=x_weight,
            debug=debug,
        )

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        one_hot = h["categorical"]
        charges = h["integer"]
        return one_hot, charges, x, node_mask


    def sample_guidance_conitional(
        self,
        target_function,
        target_value=[0],
        nodesxsample=torch.tensor([10]),
        gg_scale=1,
        cfg_scale=1,
        max_norm=10,
        std=1.0,
        fix_noise=False,
        scheduler=None,
        guidance_at=1,
        guidance_stop=0,
        guidance_ver=1,
        n_backwards=0,
        h_weight=1,
        x_weight=1,
        debug=False,
    ):
        """
        Sample molecular structures with guidance from target function and conditional property.

        Parameters:
        - target_function (Callable[[Tensor], Tensor]): Target function for guidance. Higher value, better
        - target_value (List[float]): Target values for conditional sampling.
        - nodesxsample (Tensor): Number of nodes per sample. Default is torch.tensor([10]).
        - gg_scale (float): Scale factor for gradient guidance. Default is 1.0.
        - cfg_scale (float): Scale factor for classifier-free guidance. Default is 1.0.
        - max_norm (float): Initial maximum norm for the gradients. Default is 10.0.
        - std (float): Standard deviation of the noise. Default is 1.0.
        - fix_noise (bool): Fix noise for visualization purposes. Default is False.
        - scheduler (RateScheduler): Rate scheduler. Default is None.
            The scheduler should have a step method that takes the energy and the current scale as input.
        - guidance_at (int): The timestep at which to apply guidance [0-1]  0 = since beginning. Default is 1.
        - guidance_stop (int): The timestep at which to stop applying guidance [0-1]  1 = until the end. Default is 0.  
        - guidance_ver (int): The version of the guidance. Default is 1. [0,1,2,cfg,cfg_gg]
        - n_backwards (int): Number of backward steps. Default is 0.
        - h_weight (float): Weight for the gradient of atom feature. Default is 1.0.
        - x_weight (float): Weight for the gradient of cartesian coordinate. Default is 1.0.
        - debug (bool): Debug mode. Default is False.
            Save gradient norms, max gradients, clipping coefficients, and energies to files.

        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Positions, one-hot encoding of atoms, node mask, and edge mask.
        """
        # assert int(torch.max(nodesxsample)) <= self.max_n_nodes
        nodesxsample = torch.where(
            nodesxsample > self.max_n_nodes, self.max_n_nodes, nodesxsample
        )
        batch_size = len(nodesxsample)
        if batch_size > 1:
            node_mask = torch.zeros(batch_size, self.max_n_nodes)
            nnode = self.max_n_nodes
        else:
            nnode = int(nodesxsample[0])
            node_mask = torch.zeros(batch_size, nnode)

        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * nnode * nnode, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)
        n_node = node_mask.size(1)  
        context = []
        for i, key in enumerate(self.prop_dist_model.distributions):
            if self.normalize_condition is not None:
                if self.normalize_condition == "mad":
                    mean, mad = (
                        self.prop_dist_model.normalizer[key]["mean"],
                        self.prop_dist_model.normalizer[key]["mad"],
                    )
                    val = (target_value[i] - mean) / (mad)
                elif self.normalize_condition == "maxmin":   
                    mean, min, max = (
                        self.prop_dist_model.normalizer[key]["mean"],
                        self.prop_dist_model.normalizer[key]["min"],
                        self.prop_dist_model.normalizer[key]["max"],
                    )
                    val = 2 * (target_value[i] - min) / (max - min) - 1   

                elif "value" in self.normalize_condition: # "value_n where n is the value to normalize"
                    value = float(self.normalize_condition.split("_")[1])
                    val = target_value[i] / value    
                else:
                    raise ValueError(f"Unknown normalization method: {self.normalize_condition}")    
            else:
                val = target_value[i]
            context_row = torch.tensor(
                 [val]
            ).unsqueeze(1)
            context.append(context_row)

        context = torch.cat(context, dim=1).float().to(self.device)
        context = context.repeat(1, n_node, 1)
        # sample from the EDM model
        x, h = self.model.sample_guidance(
            batch_size,
            target_function,
            node_mask,
            edge_mask,
            context,
            gg_scale,
            cfg_scale,
            max_norm,
            fix_noise=fix_noise,
            std=std,
            scheduler=scheduler,
            guidance_at=guidance_at,
            guidance_stop=guidance_stop,
            guidance_ver=guidance_ver,
            n_backwards=n_backwards,
            h_weight=h_weight,
            x_weight=x_weight,
            debug=debug,
        )

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        one_hot = h["categorical"]
        charges = h["integer"]
        return one_hot, charges, x, node_mask
    

    def sample_chain_guide(
        self,
        n_nodes: int,
        n_tries: int,
        target_function,
        scale: float = 1,
        max_norm=10,
        std: float = 1.0,
        scheduler=None,
        keep_frames: int = 100,
    ):
        """
        Sample a molecule for visualizing the diffusion process.

        Parameters:
        - n_nodes (int): Number of nodes in the molecular graph.
        - n_tries (int): Number of attempts to find a stable molecule.
        - target_function (Callable[[Tensor], Tensor]): Target function for guidance. Higher value, better
        - scale (float): Scale factor for guidance. Default is 1.0.
        - max_norm (float): Initial maximum norm for the gradients. Default is 10.0.
        - std (float): Standard deviation of the noise. Default is 1.0.
        - scheduler (RateScheduler): Rate scheduler. Default is None.
            The scheduler should have a step method that takes the energy and the current scale as input.
        - keep_frames (int): Number of frames to keep. Default is 100.

        Returns:
        Tuple[Tensor, Tensor, Tensor]: One-hot encoding of atoms, charges, and positions.
        """
        N_SAMPLE = 1
        if (len(self.condition) > 0) and (self.prop_dist_model is not None):
            context = self.prop_dist_model.sample(n_nodes).unsqueeze(1).unsqueeze(0)
            context = context.repeat(1, n_nodes, 1).to(self.device)
        else:
            context = None

        node_mask = torch.ones(N_SAMPLE, n_nodes, 1).to(self.device)

        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        edge_mask = edge_mask.repeat(N_SAMPLE, 1, 1).view(-1, 1).to(self.device)

        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = self.model.sample_chain_guidance(
                N_SAMPLE,
                target_function,
                node_mask,
                edge_mask,
                scale,
                max_norm,
                std=std,
                keep_frames=keep_frames,
                scheduler=scheduler,
            )
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()

            zs = [self.atomic_numbers[i] for i in atom_type]

            mol_stable = check_stability(x_squeeze, zs, self.atom_decoder)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(
                torch.argmax(one_hot, dim=2), num_classes=len(self.atom_decoder)
            )
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print("ლ(́◉◞౪◟◉‵ლ) Found stable molecule to visualize -(๑☆‿ ☆#)ᕗ")
                break
            elif i == n_tries - 1:
                print("Did not find stable molecule, showing last sample. ༼ಢ_ಢ༽")

        return one_hot, charges, x


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]



@core.Registry.register("GuidanceModelPrediction")
class GuidanceModelPrediction(Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(
        self,
        model,
        noisemodel,
        task=(),
        include_charge=True,
        metric=("mae", "rmse"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        mlp_batch_norm=True,
        readout="mean",
        mlp_dropout=0,
        std_mean=None,
        load_mlps_layer=0,
        nextra_nf=0,
        norm_values=(1.0, 1.0, 1.0),
        extra_norm_values=(),
        norm_biases=(None, 0.0, 0.0),
        weight_classes=None,
        t_max=1,
        verbose=0,
    ):

        super(GuidanceModelPrediction, self).__init__()
        self.model = model

        if self.model.__class__.__name__ in [
            "GraphTransformer",
            "GraphDiffTransformer",
        ]:
            self.architecture = "egt"
        elif self.model.__class__.__name__ in ["EGNN"]:
            self.architecture = "egcn"
        elif self.model.__class__.__name__ in ["PaiNN", "GemNetOC"]:
            self.architecture = "egnn_extra"
        
        self.metric = metric
        self.criterion = {"mse":1}

        self.gamma = noisemodel
        self.task = task
        self.include_charge = include_charge
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.verbose = verbose
        self.std_mean = std_mean
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.readout = readout
        self.t_max = t_max

        self.num_targets = len(self.task)
        self.ndim_extra = nextra_nf
        self.n_dims = 3
        self.in_node_nf = self.model.in_node_nf - 1
        if weight_classes is None:
            self.weight_classes = torch.ones(len(self.task))
        else:
            self.weight_classes = torch.tensor(weight_classes, dtype=torch.float32)
        self.T = self.gamma.T
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.extra_norm_values = extra_norm_values

        self.mlp = None
        self.mlp_final = None
        if std_mean:
            self.std = std_mean[0]
            self.mean = std_mean[1]

        if load_mlps_layer > 0:
            hidden_dims = [self.model.hidden_nf] * (self.num_mlp_layer - 1)
            self.mlp = common.MLP(
                self.model.hidden_nf,
                hidden_dims,
                batch_norm=self.mlp_batch_norm,
                dropout=self.mlp_dropout,
            )
        self.load_mlps_layer = load_mlps_layer


        if self.num_class:
            if load_mlps_layer > 0:
                n_layer_final = self.num_mlp_layer - load_mlps_layer - 1
                self.mlp_final = common.MLP(
                    hidden_dims[n_layer_final:-1],
                    sum(self.num_class) ,
                    batch_norm=self.mlp_batch_norm,
                    dropout=self.mlp_dropout,
                )
            else:
                hidden_dims = [self.model.hidden_nf] * (self.num_mlp_layer - 1)
                self.mlp = common.MLP(
                    self.model.hidden_nf,
                    hidden_dims + [sum(self.num_class) ],
                    batch_norm=self.mlp_batch_norm,
                    dropout=self.mlp_dropout,
                )


    def preprocess(self, train_set, valid_set=None, test_set=None):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)

        if train_set is not None:

            for sample in train_set:
                if not sample.get("labeled", True):
                    continue
                for task in self.task:
                    if not math.isnan(sample[task]):
                        values[task].append(sample[task])
            mean = []
            std = []
            weight = []
            num_class = []
            for task, w in self.task.items():
                value = torch.tensor(values[task])
                mean.append(value.float().mean())
                std.append(value.float().std())
                weight.append(w)
                if value.ndim > 1:
                    num_class.append(value.shape[1])
                elif value.dtype == torch.long:
                    task_class = value.max().item()
                    if task_class == 1 and "bce" in self.criterion:
                        num_class.append(1)
                    else:
                        num_class.append(task_class + 1)
                else:
                    num_class.append(1)
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
            self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
            self.num_class = self.num_class or num_class

            hidden_dims = [self.model.hidden_nf] * (self.num_mlp_layer - 1)

            if self.mlp is None:
                self.mlp = common.MLP(
                    self.model.hidden_nf,
                    hidden_dims + [sum(self.num_class) ],
                    batch_norm=self.mlp_batch_norm,
                    dropout=self.mlp_dropout,
                )
            if self.load_mlps_layer > 0:
                n_layer_final = self.num_mlp_layer - self.load_mlps_layer - 1
                self.mlp_final = common.MLP(
                    hidden_dims[n_layer_final:-1],
                    sum(self.num_class) ,
                    batch_norm=self.mlp_batch_norm,
                    dropout=self.mlp_dropout,
                )

            self.train_set_size = len(train_set)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.weight_classes = self.weight_classes.to(device)
        else:
            self.train_set_size = 0
            
            
    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0
        
        if self.normalization:
            loss = F.mse_loss(
                (pred - self.mean) / self.std,
                (target - self.mean) / self.std,
                reduction="none",
            )
        else:
            loss = F.mse_loss(pred, target, reduction="none")
        name = _get_criterion_name("mse")
        if self.verbose > 0:
            for t, l in zip(self.task, loss):
                metric["%s [%s]" % (name, t)] = l
        loss = functional.masked_mean(loss, labeled, dim=0).sum()
        metric[name] = loss
        all_loss += loss

        return all_loss, metric        


    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get(
            "labeled", torch.ones(len(target), dtype=torch.bool, device=target.device)
        )
        target[~labeled] = math.nan
        return target
        
    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)

        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            name = _get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric     
                
    def predict(self, batch, all_loss=None, metric=None, evaluate=False):
        """"
        If evaluate is True, the data must be normalized beforehand.
        """

        h = batch["graph"].x
        charges = batch["graph"].atomic_numbers.unsqueeze(-1)
        x = batch["graph"].pos

        bs = batch["graph"].batch.max().item() + 1
        n_atoms  = batch["graph"].natoms
        n_nodes = h.size(0) 
        node_mask = torch.ones((n_nodes, 1), device=x.device)

        if self.include_charge:
            if self.ndim_extra > 0:
                h_cat = h[:, :, : self.ndim_extra]
                h_extra = h[:, :, -self.ndim_extra :]
                h = torch.cat(
                    [h_cat, charges, h_extra], dim=1
                )  # NOTE not sure about dim
            else:
                h = torch.cat([h, charges], dim=1)
        x, h  = self.normalize(x, h, node_mask)    
        if evaluate:
            z_h, z_x = h, x
        else:
            t_upper = int(self.T * self.t_max)
            t_int = torch.zeros((n_nodes, 1), device=x.device, dtype=torch.long)
            t_int_value = torch.randint(
                0, t_upper + 1, size=(bs, 1),  dtype=torch.long
            )

            n_atom_cum = 0
            for i, n_atom in enumerate(n_atoms):
                t_int[n_atom_cum : n_atom_cum + n_atom] = t_int_value[i]
                n_atom_cum += n_atom

            t = t_int / self.T
            eps_x, eps_h = self.sample_combined_position_feature_noise(
                n_samples=1, n_nodes=n_nodes, node_mask=node_mask
            )
            eps_x = eps_x.view(n_nodes, 3)
            eps_h = eps_h.view(n_nodes, -1)
            
            if self.ndim_extra > 0:
                s_eps_hint = eps_h[
                    :, self.in_node_nf - self.ndim_extra - 1
                ].unsqueeze(-1) * self.gamma.get_sigma_bar(
                    t_int=t_int, key="integer"
                )
                s_eps_hcat = eps_h[
                    : self.in_node_nf - self.ndim_extra - 1
                ] * self.gamma.get_sigma_bar(t_int=t_int, key="categorical")

                s_eps_hextra = eps_h[
                    :, -self.ndim_extra :
                ] * self.gamma.get_sigma_bar(t_int=t_int, key="extra")

                s_eps_hs = torch.cat([s_eps_hcat, s_eps_hint, s_eps_hextra], dim=1)
            else:
                s_eps_hint = eps_h[:,-1].unsqueeze(-1) * self.gamma.get_sigma_bar(
                    t_int=t_int, key="integer"
                )
                s_eps_hcat = eps_h[:, :-1] * self.gamma.get_sigma_bar(
                    t_int=t_int, key="categorical"
                )
                s_eps_hs = torch.cat([s_eps_hcat, s_eps_hint], dim=1)

            s_eps_x = eps_x * self.gamma.get_sigma_bar(
                t_int=t_int, key="pos"
            )

            s_eps = torch.cat([s_eps_x, s_eps_hs], dim=1)

            h_catp = h[
                :, : self.in_node_nf - self.ndim_extra - 1
            ] * self.gamma.get_alpha_bar(t_int=t_int, key="categorical")
            h_intp = h[
                :, self.in_node_nf - self.ndim_extra - 1
            ].unsqueeze(-1) * self.gamma.get_alpha_bar(t_int=t_int, key="integer")

            if self.ndim_extra > 0:
                h_extrap = h[:, -self.ndim_extra :] * self.gamma.get_alpha_bar(
                    t_int=t_int, key="extra"
                )
                hs = torch.cat([h_catp, h_intp, h_extrap], dim=1)
            else:
                hs = torch.cat([h_catp, h_intp], dim=1)

            xp = x * self.gamma.get_alpha_bar(t_int=t_int, key="pos")
            xh = torch.cat([xp, hs], dim=1)
            z_t = xh + s_eps
            z_h = z_t[:, 3:]
            z_x = z_t[:, :3]   


        if self.architecture == "egcn":
            
            # if not(self.include_charge):
            #     z_h = z_h[:, :-1]
            if evaluate:
                t = batch["graph"].times # size: (n_nodes, 1)
                if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
                    t = torch.full((n_nodes, 1), t, device=x.device)
                elif isinstance(t, torch.Tensor) and t.ndim == 1:
                    t = t.unsqueeze(-1).repeat(n_nodes // t.shape[0], 1)

            z_h = torch.cat([z_h, t], dim=1)
              
            edge_index = batch["graph"].edge_index
            edges = [edge_index[0], edge_index[1]]
            node_mask = None
            edge_mask = None
            h_final, _ = self.model(
                z_h, x, edges, node_mask=node_mask, edge_mask=edge_mask, use_embed=True
            )  
        elif self.architecture == "egnn_extra":
            if not(self.include_charge):
                z_h = z_h[:, :-1]

            if evaluate:
                t = batch["graph"].times
            
            z_h = torch.cat([z_h, t], dim=1)
            batch["graph"].x = z_h
            batch["graph"].pos = z_x    

            h_final, _ = self.model(batch["graph"], use_embed=True)    
               

        h_final = self.pad_data(h_final, batch, self.model.hidden_nf)   
        output = {
            "graph_feature": self.readout_f(h_final),
            "node_feature": h_final,
        }

        if self.load_mlps_layer > 0:
            x = self.mlp(output["graph_feature"])
            pred = self.mlp_final(x)
        else:
            pred = self.mlp(output["graph_feature"])
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred


    def get_adj_matrix(self, _edges_dict, n_nodes, batch_size):
        if n_nodes in _edges_dict:
            edges_dic_b = _edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [
                    torch.LongTensor(rows).to(self.device),
                    torch.LongTensor(cols).to(self.device),
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            _edges_dict[n_nodes] = {}
            return self.get_adj_matrix(_edges_dict, n_nodes, batch_size)

    def pad_data(self, array, batch, dim):
        """"
        array: torch.Tensor of shape (n_atoms, n_features)
        batch: pytorch_geometric.data.Batch
        """
        bs = batch["graph"].batch.max().item() + 1
        natoms = batch["graph"].natoms   
        n_nodes = natoms.max().item()
        array_paddded = torch.zeros(bs, n_nodes, array.shape[1]).to(self.device)
        natom_cum = 0
        for i, natom in enumerate(natoms):
            array_mol = array[natom_cum:natom_cum+natom]
            array_mol = torch.cat([array_mol, torch.zeros(n_nodes-natom, array.shape[1]).to(self.device)], dim=0)    
            array_paddded[i] = array_mol
            natom_cum += natom

        array = array_paddded.view(bs, n_nodes, dim)
        
        return array

    def readout_f(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Perform readout operation over nodes in each molecule.

        Parameters:
        - embeddings (torch.Tensor): Tensor of size (x, y, z) where x is the batch size, y is the number of nodes, and z is the feature size.

        Returns:
        torch.Tensor: Aggregated tensor of size (x, z).
        """
        if self.readout == "sum":
            return embeddings.sum(dim=1)
        elif self.readout == "mean":
            return embeddings.mean(dim=1)
        else:
            raise ValueError("Unsupported method. Choose either 'sum' or 'mean'.")
    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        # delta_log_px = -self.subspace_dimensionality(node_mask) * torch.log(
        #     torch.tensor(self.norm_values[0])
        # )

        # Casting to float in case h still has long or int type.
        h_cat = (
            (
                h[:, : self.in_node_nf - self.ndim_extra - 1].float()
                - self.norm_biases[1]
            )
            / self.norm_values[1]
            * node_mask
        )
        h_int = (
            (
                h[:, self.in_node_nf - self.ndim_extra - 1].float().unsqueeze(-1)
                - self.norm_biases[2]
            )
            / self.norm_values[2]
            * node_mask
        )

        h = torch.cat([h_cat, h_int], dim=1)
        if len(self.extra_norm_values) > 0:
            h_extra = (
                h[:, -self.ndim_extra :].float()
                / torch.tensor(self.extra_norm_values, device=x.device).view(1, -1)
                * node_mask
            )
            h = torch.cat([h, h_extra], dim=2)

        return x, h

    def sample_combined_position_feature_noise(
        self, n_samples, n_nodes, node_mask, std=1.0
    ):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
            std=std,
        )

        if self.ndim_extra > 0:

            z_h = sample_gaussian_with_mask(
                size=(
                    n_samples,
                    n_nodes,
                    self.in_node_nf - self.ndim_extra,
                ),
                device=node_mask.device,
                node_mask=node_mask,
                std=std,
            )
            z_h_extra = sample_gaussian_with_mask(
                size=(n_samples, n_nodes, self.ndim_extra),
                device=node_mask.device,
                node_mask=node_mask,
                std=std,
            )
            z_h = torch.cat([z_h, z_h_extra], dim=2)
        else:
            z_h = sample_gaussian_with_mask(
                size=(n_samples, n_nodes, self.in_node_nf),
                device=node_mask.device,
                node_mask=node_mask,
                std=std,
            )

        return z_x, z_h

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims


