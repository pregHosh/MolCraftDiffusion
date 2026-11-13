"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn as nn
from MolecularDiffusion.modules.layers.common import MLP
from MolecularDiffusion.modules.layers.graph_module import GraphModelMixin, HeadInterface

from MolecularDiffusion.modules.layers.esen.common.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
)
from MolecularDiffusion.modules.layers.esen.common.so3 import (
    CoefficientMapping,
    SO3_Grid,
)
from MolecularDiffusion.modules.layers.esen.esen_block import eSEN_Block
from MolecularDiffusion.modules.layers.esen.nn.embedding import EdgeDegreeEmbedding, ChgSpinDatasetEmbedding
from MolecularDiffusion.modules.layers.esen.nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from MolecularDiffusion.modules.layers.esen.nn.radial import EnvelopedBesselBasis, GaussianSmearing, RadialMLP
from MolecularDiffusion.modules.layers.esen.nn.so3_layers import SO3_Linear
from MolecularDiffusion.modules.layers.esen.nn.activation import GateActivation
from MolecularDiffusion.utils import remove_mean, remove_mean_pyG


class eSEN_VelocityFFN(nn.Module):
    """
    Native eSEN equivariant FFN for coordinate updates.
    Mirrors the equiformer FeedForwardNetwork pattern but operates on
    eSEN's native [N, (lmax+1)^2, sphere_channels] tensors.

    Forward: x_message -> SO3_Linear -> GateActivation -> SO3_Linear(out=1)
             -> l=1 slice [:, 1:4, 0] -> [N, 3] position delta
    """
    def __init__(self, sphere_channels: int, hidden_channels: int, lmax: int):
        super().__init__()
        self.lmax = lmax

        self.so3_linear_1 = SO3_Linear(sphere_channels, hidden_channels, lmax=lmax)
        # gate: scalar branch produces lmax * hidden_channels gating values
        self.gating_linear = nn.Linear(sphere_channels, lmax * hidden_channels)
        self.gate_act = GateActivation(lmax, lmax, hidden_channels)
        self.so3_linear_2 = SO3_Linear(hidden_channels, 1, lmax=lmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, (lmax+1)^2, sphere_channels]
        Returns:
            pos_delta: [N, 3]
        """
        # Gating scalars from l=0 only
        gating = self.gating_linear(x[:, 0:1, :])  # [N, 1, lmax*hidden]
        gating = gating.squeeze(1)                  # [N, lmax*hidden]

        x = self.so3_linear_1(x)                    # [N, (lmax+1)^2, hidden]
        x = self.gate_act(gating, x)                # [N, (lmax+1)^2, hidden]
        x = self.so3_linear_2(x)                    # [N, (lmax+1)^2, 1]

        # l=1 components are at indices 1,2,3
        return x[:, 1:4, 0]                         # [N, 3]


class GaussianFeatureEmbedding(nn.Module):
    def __init__(self, num_channels, num_basis=20, start=0.0, stop=10.0, trainable=False):
        super().__init__()
        self.num_channels = num_channels
        self.num_basis = num_basis
        self.trainable = trainable
        
        offset = torch.linspace(start, stop, num_basis)
        self.register_buffer("offset", offset)
        
        # Width of the Gaussian 
        width = (stop - start) / (num_basis - 1) if num_basis > 1 else 1.0
        self.register_buffer("width", torch.tensor(width))

        if trainable:
            self.offset = nn.Parameter(self.offset)
            self.width = nn.Parameter(self.width)

    def forward(self, x):
        # x: [..., num_channels]
        # output: [..., num_channels * num_basis]
        
        # Expand last dim to broadcast against basis
        # x: [..., C, 1]
        x_expanded = x.unsqueeze(-1)
        
        # offset: [Basis] -> Broadcasts with [..., C, 1] to produce [..., C, Basis]
        coeff = -0.5 * torch.pow(x_expanded - self.offset, 2) / torch.pow(self.width, 2)
        g = torch.exp(coeff)
        
        # Flatten last two dims
        # [..., C * Basis]
        return g.reshape(x.shape[:-1] + (-1,))

class eSEN_Backbone(nn.Module, GraphModelMixin):
    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        otf_graph: bool = False,
        max_neighbors: int = 300,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = True,
        regress_forces: bool = True,
        regress_stress: bool = False,
        # escnmd specific
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "s2",
        mlp_type: str = "grid",
        use_envelope: bool = False,
        activation_checkpointing: bool = False,
        global_attributes: bool = False,
        sphere_embedding_type: str = "embedding",
        in_node_channels: int | None = None,
        update_position: bool = False, 
        tanh: bool = False,
        coords_range: float = 10,
        normalization_factor: float = 1,
        aggregation_method: str = "sum"
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution

        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress

        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.enforce_max_neighbors_strictly = False
        self.activation_checkpointing = activation_checkpointing
        
        # Store in_node_channels for external access (e.g., GuidanceModelPrediction)
        self.in_node_channels = in_node_channels

        self.mlp_type = mlp_type
        self.use_envelope = use_envelope
        self.update_position = update_position
        # rotation utils
        Jd_list = torch.load(
            os.path.join(os.path.dirname(__file__), "../../layers/equiformer_v2/Jd.pt"),
            weights_only=True,
        )
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])
        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)
        self.hidden_nf = self.sphere_channels * self.sph_feature_size
        # lmax_lmax for node, lmax_mmax for edge
        self.SO3_grid = nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=grid_resolution, rescale=True
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=grid_resolution, rescale=True
        )

        # atom embedding
        self.sphere_embedding_type = sphere_embedding_type
        if self.sphere_embedding_type == "embedding":
            self.sphere_embedding = nn.Embedding(
                self.max_num_elements, self.sphere_channels
            )
        elif self.sphere_embedding_type == "mlp":
            self.sphere_embedding = RadialMLP(
                [in_node_channels, hidden_channels, self.sphere_channels]
            )
        elif self.sphere_embedding_type == "mixed":
            self.sphere_embedding = nn.ModuleDict(
                {
                    "embedding": nn.Embedding(
                        self.max_num_elements, self.sphere_channels
                    ),
                    "mlp": RadialMLP(
                        [in_node_channels - 1, hidden_channels, self.sphere_channels]
                    ),
                }
            )
        elif self.sphere_embedding_type == "gaussian":
            # Gaussian Smearing for all input channels
            self.n_gaussian_basis = 20 # Can make this configurable if needed
            self.gaussian_embedding = GaussianFeatureEmbedding(
                num_channels=in_node_channels, 
                num_basis=self.n_gaussian_basis, 
                start=0, 
                stop=100
            )
            # Linear projection from Gaussian expansion to sphere_channels
            self.sphere_embedding_proj = nn.Linear(
                in_node_channels * self.n_gaussian_basis, 
                self.sphere_channels
            )
        else:
            raise ValueError(f"Unknown sphere embedding type: {sphere_embedding_type}")
        

        # charge / spin embedding
        self.global_attributes = global_attributes
        if global_attributes:
            self.chg_spin_emb_type = "pos_emb" # ["pos_emb", "lin_emb", "rand_emb"]
            self.cs_emb_grad = False
            self.dataset_emb_grad = False
            self.dataset_list = []
            
            self.charge_embedding = ChgSpinDatasetEmbedding(
                self.chg_spin_emb_type,
                "charge",
                self.sphere_channels,
                grad=self.cs_emb_grad,
            )
            self.spin_embedding = ChgSpinDatasetEmbedding(
                self.chg_spin_emb_type,
                "spin",
                self.sphere_channels,
                grad=self.cs_emb_grad,
            )
    
            self.dataset_embedding = ChgSpinDatasetEmbedding(
                self.chg_spin_emb_type,
                "dataset",
                self.sphere_channels,
                grad=self.cs_emb_grad,
            )
            # mix charge, spin, dataset embeddings
            self.mix_csd = nn.Linear(3 * self.sphere_channels, self.sphere_channels)

        # edge distance embedding
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                2.0,
            )
        elif self.distance_function == "bessel":
            self.distance_expansion = EnvelopedBesselBasis(
                num_radial=self.num_distance_basis,
                cutoff=cutoff,
            )
            self.distance_expansion.offset = [self.cutoff]
            self.distance_expansion.num_output = self.num_distance_basis
        else:
            raise ValueError("Unknown distance function")

        # equivariant initial embedding
        if self.sphere_embedding_type == "mlp":
            self.source_embedding = RadialMLP(
                [1,  self.edge_channels]
            )
            self.target_embedding = RadialMLP(
                [1,  self.edge_channels]
            )
        else:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        if self.sphere_embedding_type == "gaussian":
            # Override edge embeddings for Gaussian mode to accept floats
            # Input dim is 1 (Atomic Number) -> Projected to edge_channels
            self.source_embedding = nn.Sequential(
                GaussianFeatureEmbedding(num_channels=1, num_basis=self.n_gaussian_basis, start=0, stop=max_num_elements),
                nn.Linear(1 * self.n_gaussian_basis, self.edge_channels)
            )
            self.target_embedding = nn.Sequential(
                GaussianFeatureEmbedding(num_channels=1, num_basis=self.n_gaussian_basis, start=0, stop=max_num_elements),
                nn.Linear(1 * self.n_gaussian_basis, self.edge_channels)
            )

        if self.sphere_embedding_type == "gaussian":
            # Override edge embeddings for Gaussian mode to accept floats
            # Input dim is 1 (Atomic Number) -> Projected to edge_channels
            self.source_embedding = nn.Sequential(
                GaussianFeatureEmbedding(num_channels=1, num_basis=self.n_gaussian_basis, start=0, stop=max_num_elements),
                nn.Linear(1 * self.n_gaussian_basis, self.edge_channels)
            )
            self.target_embedding = nn.Sequential(
                GaussianFeatureEmbedding(num_channels=1, num_basis=self.n_gaussian_basis, start=0, stop=max_num_elements),
                nn.Linear(1 * self.n_gaussian_basis, self.edge_channels)
            )

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,
            cutoff=self.cutoff,
            mappingReduced=self.mappingReduced,
            out_mask=self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
                self.lmax, self.mmax
            ),
            use_envelope=use_envelope,
        )

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.d_model = self.sphere_channels * (self.lmax + 1) ** 2
        self.norm_type = norm_type
        self.act_type = act_type

        # Initialize the blocks for each layer
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = eSEN_Block(
                self.sphere_channels,
                self.hidden_channels,
                self.lmax,
                self.mmax,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channels_list,
                self.cutoff,
                self.norm_type,
                self.act_type,
                self.mlp_type,
                self.use_envelope,
            )
            self.blocks.append(block)

        if self.update_position:
            # Per-layer equivariant coordinate update via native eSEN FFN (equiformer-style)
            coordinate_modules = nn.ModuleDict()
            for n in range(num_layers):
                coordinate_modules[f"update_{n}"] = eSEN_VelocityFFN(
                    sphere_channels=sphere_channels,
                    hidden_channels=hidden_channels,
                    lmax=self.lmax,
                )
            self.coordinate_modules = coordinate_modules
        else:
            self.coordinate_modules = None
            
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )

    def get_rotmat_and_wigner(self, edge_distance_vecs):
        edge_rot_mat = init_edge_rot_mat(
            edge_distance_vecs, rot_clip=(not self.direct_forces)
        )

        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_rot_mat.dtype)
            for l in range(self.lmax + 1)
        ]

        wigner = rotation_to_wigner(
            edge_rot_mat,
            0,
            self.lmax,
            Jd_buffers,
            rot_clip=(not self.direct_forces),
        )
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        return edge_rot_mat, wigner, wigner_inv

    #TODO check graph keys
    def generate_graph(self, *args, **kwargs):
        graph = super().generate_graph(*args, **kwargs)
        return {
            "edge_index": graph.edge_index,
            "edge_distance": graph.edge_distance,
            "edge_distance_vec": graph.edge_distance_vec,
            "cell_offsets": None, #graph.cell_offsets,
            "offset_distances": None,
            "neighbors": None,
            "node_offset": 0,
            "batch_full": graph.batch_full,# graph.batch_full, 
            "atomic_numbers_full": graph.atomic_numbers_full,
        }

    def csd_embedding(self, charge, spin, dataset):
        # Add charge, spin, and dataset embeddings
        chg_emb = self.charge_embedding(charge)
        spin_emb = self.spin_embedding(spin)
        
        assert dataset is not None
        dataset_emb = self.dataset_embedding(dataset)
        return torch.nn.SiLU()(
            self.mix_csd(torch.cat((chg_emb, spin_emb, dataset_emb), dim=1))
        )
     
        
    def forward(self, data) -> dict[str, torch.Tensor]:
        ###############################################################
        # gradient-based forces/stress
        ###############################################################

   
        if self.global_attributes:
            csd_mixed_emb = self.csd_embedding(
                charge=data.get("charge", torch.zeros_like(data.atomic_numbers)),
                spin=data.get("spin", torch.zeros_like(data.atomic_numbers)) + 1,
                dataset=data.get("domain", None),
            )
        else:
            csd_mixed_emb = None

        graph_dict = self.generate_graph(data)

        _, wigner, wigner_inv = self.get_rotmat_and_wigner(
            graph_dict["edge_distance_vec"]
        )
     

        ###############################################################
        # Initialize node embeddings
        ###############################################################
       
        x_message = torch.zeros(
            data.pos.shape[0],
            self.sph_feature_size,
            self.sphere_channels,
            device=data.pos.device,
            dtype=data.pos.dtype,
        )

        if self.sphere_embedding_type == "embedding":
            h_node = self.sphere_embedding(data.atomic_numbers)
        elif self.sphere_embedding_type == "mlp":
           
            h_node = self.sphere_embedding(torch.cat([data.atomic_numbers.unsqueeze(-1).float(), data.x], dim=-1))
        elif self.sphere_embedding_type == "mixed":
            h_node = self.sphere_embedding["embedding"](data.atomic_numbers) + \
                     self.sphere_embedding["mlp"](data.x)
        elif self.sphere_embedding_type == "gaussian":
            # Concatenate inputs: [N, 1] + [N, Extra] -> [N, C]
            continuous_input = torch.cat([data.atomic_numbers.unsqueeze(-1).float(), data.x], dim=-1)
            # Apply Gaussian Smearing -> [N, C * Basis]
            h_gauss = self.gaussian_embedding(continuous_input.unsqueeze(0)).squeeze(0) # Unsqueeze for batch dim if needed, or update module to handle 2D
            # Project
            h_node = self.sphere_embedding_proj(h_gauss)
        else:
            raise ValueError(f"Unknown sphere embedding type: {self.sphere_embedding_type}")

        if self.global_attributes:
            x_message[:, 0, :] = h_node + csd_mixed_emb
        else:
            x_message[:, 0, :] = h_node

        # edge degree embedding           
        edge_distance_embedding = self.distance_expansion(graph_dict["edge_distance"])
       
        # Edge embedding: handle different embedding types
        # For nn.Embedding (embedding/mixed): need Long indices
        # For RadialMLP (mlp): can use float continuous values
        # For gaussian: use float continuous values
        if self.sphere_embedding_type in ["mlp", "gaussian"]:
            # Use float for continuous atomic number handling (e.g., guidance mode)
            source_vec = data.atomic_numbers[graph_dict["edge_index"][0]].unsqueeze(-1).float()
            target_vec = data.atomic_numbers[graph_dict["edge_index"][1]].unsqueeze(-1).float()
        else:
            # embedding or mixed: requires Long indices for nn.Embedding
            source_vec = data.atomic_numbers[graph_dict["edge_index"][0]]
            target_vec = data.atomic_numbers[graph_dict["edge_index"][1]]
        source_embedding = self.source_embedding(
            source_vec
        )
        target_embedding = self.target_embedding(
            target_vec
        )

        x_edge = torch.cat(
            (edge_distance_embedding,
             source_embedding, 
             target_embedding,
             ), dim=1
        )
        x_message = self.edge_degree_embedding(
            x_message,
            x_edge,
            graph_dict["edge_distance"],
            graph_dict["edge_index"],
            wigner_inv,
        )

        if self.update_position:
            pos = data.pos

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        if graph_dict["edge_index"].shape[1] != 0:
            for i in range(self.num_layers):
                if self.activation_checkpointing:
                    x_message = torch.utils.checkpoint.checkpoint(
                        self.blocks[i],
                        x_message,
                        x_edge,
                        graph_dict["edge_distance"],
                        graph_dict["edge_index"],
                        wigner,
                        wigner_inv,
                        graph_dict["node_offset"],
                        use_reentrant=False,
                    )
                else:
                    x_message = self.blocks[i](
                        x_message,
                        x_edge,
                        graph_dict["edge_distance"],
                        graph_dict["edge_index"],
                        wigner,
                        wigner_inv,
                        node_offset=graph_dict["node_offset"],
                    )
                if self.coordinate_modules is not None:
                    pos_delta = self.coordinate_modules[f"update_{i}"](x_message)  # [N, 3]
                    pos = pos + pos_delta

        # Final layer norm
        x_message = self.norm(x_message)
        
        results = {
            "x": x_message.view(-1, self.d_model),
            "num_atoms": data.num_atoms,
            "batch": data.batch,
            "token_idx": data.token_idx,
        }
        if self.update_position:
            results["pos"] = pos
        return results
        
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_Linear,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_Linear))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)



class eSEN_Encoder(eSEN_Backbone):
    """eSEN Encoder for VAE tasks.
    
    Wraps eSEN_Backbone with fixed configuration for VAE encoding:
    - No force/stress regression
    - No position updates
    - Standard embedding
    - No global attributes
    - No OTF graph (assumes pre-computed or standard radius graph)
    """
    
    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        max_neighbors: int = 300,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        # escnmd specific
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "s2",
        mlp_type: str = "grid",
        use_envelope: bool = False,
        activation_checkpointing: bool = False,
        sphere_embedding_type: str = "embedding",
        # Force-disabled or fixed args
        **kwargs
    ):
        # Enforce encoder-specific config
        kwargs.pop("regress_forces", None)
        kwargs.pop("direct_forces", None)
        kwargs.pop("regress_stress", None)
        kwargs.pop("update_position", None)
        kwargs.pop("global_attributes", None)
        kwargs.pop("otf_graph", None)
        
        super().__init__(
            max_num_elements=max_num_elements,
            sphere_channels=sphere_channels,
            lmax=lmax,
            mmax=mmax,
            grid_resolution=grid_resolution,
            otf_graph=True,
            max_neighbors=max_neighbors,
            use_pbc=False,  # VAE usually handles single molecules
            use_pbc_single=False,
            cutoff=cutoff,
            edge_channels=edge_channels,
            distance_function=distance_function,
            num_distance_basis=num_distance_basis,
            direct_forces=False,
            regress_forces=False,
            regress_stress=False,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            norm_type=norm_type,
            act_type=act_type,
            mlp_type=mlp_type,
            use_envelope=use_envelope,
            activation_checkpointing=activation_checkpointing,
            global_attributes=False,
            sphere_embedding_type=sphere_embedding_type,
            update_position=False,
            **kwargs
        )


class MLP_EFS_Head(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        backbone.energy_block = None
        backbone.force_block = None
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        backbone.direct_forces = False

    # @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_key = "energy"
        forces_key = "forces"
        stress_key = "stress"

        outputs = {}

        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze()
        ).view(-1, 1, 1)

        energy = torch.zeros(
            len(data["natoms"]), device=data["pos"].device, dtype=node_energy.dtype
        )
        energy.index_add_(0, data["batch"], node_energy.view(-1))
        outputs[energy_key] = energy

        if self.regress_stress:
            grads = torch.autograd.grad(
                [energy.sum()],
                [data["pos"], emb["displacement"]],
                create_graph=self.training,
            )
            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(data["cell"]).abs().unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
            virial = torch.neg(virial)
            outputs[forces_key] = forces
            outputs[stress_key] = stress.view(-1, 9)
            data["cell"] = emb["orig_cell"]
        elif self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy.sum(), data["pos"], create_graph=self.training
                )[0]
            )
            outputs[forces_key] = forces
        return outputs



class MLP_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "sum"):
        super().__init__()
        self.reduce = reduce

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze()
        ).view(-1, 1, 1)

        energy = torch.zeros(
            len(data_dict["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy.index_add_(0, data_dict["batch"], node_energy.view(-1))
        if self.reduce == "sum":
            return {"energy": energy}
        elif self.reduce == "mean":
            return {"energy": energy / data_dict["natoms"]}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )



class Linear_Force_Head(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        forces = self.linear(emb["node_embedding"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        return {"forces": forces}



class eSEN_dynamics(nn.Module, GraphModelMixin):
    def __init__(
        self,
        condition_time=True,
        context_node_nf=0,
        use_adapter_module: bool = False,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        max_neighbors: int = 300,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        # escnmd specific
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "s2",
        mlp_type: str = "grid",
        use_envelope: bool = False,
        activation_checkpointing: bool = False,
        sphere_embedding_type: str = "embedding",
        in_node_channels: int | None = None,
        tanh: bool = False,
        coords_range: float = 10,
        normalization_factor: float = 1,
        n_mlp_layers_out: int = 3,
        aggregation_method: str = "sum",
        otf_graph: bool = False,
    ):
        super().__init__()

        self.n_dims = 3
        self.condition_time = condition_time
        self.context_node_nf = context_node_nf
        self.use_adapter_module = use_adapter_module
        in_node_nf = in_node_channels + 1  # atom type + extra feats + time + contexts
        if use_adapter_module:
            in_node_nf += context_node_nf
        self.egnn = eSEN_Backbone(
            max_num_elements=max_num_elements,
            sphere_channels=sphere_channels,
            lmax=lmax,
            mmax=mmax,
            grid_resolution=grid_resolution,
            otf_graph=otf_graph,
            max_neighbors=max_neighbors,
            use_pbc=False,
            use_pbc_single=False,
            cutoff=cutoff,
            edge_channels=edge_channels,
            distance_function=distance_function,
            num_distance_basis=num_distance_basis,
            direct_forces=False,
            regress_forces=False,
            regress_stress=False,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            norm_type=norm_type,
            act_type=act_type,
            mlp_type=mlp_type,
            use_envelope=use_envelope,
            activation_checkpointing=activation_checkpointing,
            global_attributes=False,
            sphere_embedding_type=sphere_embedding_type,
            in_node_channels=in_node_nf,
            update_position=True,
            tanh=tanh,
            coords_range=coords_range,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
        )
        self.sph_feature_size = int((lmax + 1) ** 2)
        hidden_nf = sphere_channels * self.sph_feature_size
        self.embedding_out = MLP(
            hidden_nf,
            [hidden_nf] * n_mlp_layers_out + [in_node_nf],
            batch_norm=False,
            dropout=0,
        )
    def _forward(self, mol_graph, return_hidden=False):
        """
        Forward pass through eSEN dynamics.
        
        Args:
            mol_graph: Dict with 'graph' (PyG Batch), 't' (timestep), 'context' (optional)
            return_hidden: If True, also return hidden features before final projection
            
        Returns:
            If return_hidden=False: continuous_out [N, 3 + in_node_nf]
            If return_hidden=True: (continuous_out, hidden_features [N, hidden_nf])
        """
        x = mol_graph["graph"].pos
        h = mol_graph["graph"].x
        if self.condition_time:
            if isinstance(mol_graph, dict):
                h_time = mol_graph["t"]
            else:
                h_time = mol_graph["t"]
                
            if h is None:
                h = h_time
            else:
                h = torch.cat([h, h_time], dim=1)

        if hasattr(mol_graph["graph"], 'context') and not(self.use_adapter_module):
            # We're conditioning, awesome!
            context = mol_graph["graph"].context # (nnodes, n_contexts)
            h = torch.cat([h, context], dim=1)
        else:
            context = None
        
        # Update batch.x
        mol_graph["graph"].x = h
        mol_graph["graph"].num_atoms = mol_graph["graph"].natoms
        mol_graph["graph"].token_idx = torch.zeros(mol_graph["graph"].num_nodes).long() #!! device

        # segment_coo (used inside eSEN backbone) requires edge_index[1] to be sorted
        edge_index = mol_graph["graph"].edge_index
        perm = edge_index[1].argsort()
        mol_graph["graph"].edge_index = edge_index[:, perm]

        # Pass batch to egnn
        res = self.egnn(mol_graph["graph"])
        
        # Hidden features BEFORE final projection (high-dimensional)
        hidden_features = res["x"]  # [N, sphere_channels * sph_feature_size]
        
        # Final projection to output dimension
        h_final = self.embedding_out(hidden_features)
        x_final = res["pos"]
 
        vel = (
            x_final - x
        ) # This masking operation is redundant but just in case

        if context is not None and not(self.use_adapter_module):
            # Slice off context size:
            h_final = h_final[:, : -self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]
        if torch.any(torch.isnan(vel)):
            warnings.warn("detected nan, resetting EGNN output to zero.", RuntimeWarning, stacklevel=2)
            vel = torch.zeros_like(vel)
            h_final = torch.zeros_like(h_final)
        else:
            vel = remove_mean_pyG(vel, mol_graph["graph"].batch)
        
        continuous_out = torch.cat([vel, h_final], dim=1)
        
        if return_hidden:
            return continuous_out, hidden_features
        return continuous_out

    def _forward_pyG(self, *args, **kwargs):
        """Alias for API compatibility with EnFlowMatchingPyG."""
        return self._forward(*args, **kwargs)
