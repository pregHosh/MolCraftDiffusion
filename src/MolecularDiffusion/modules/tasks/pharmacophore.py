"""
Pharmacophore Diffusion Task.

Multi-modal diffusion task for generating molecules with 4 modalities:
- x1: Molecular structure (atoms + bonds)
- x2: Shape (point cloud)
- x3: Electrostatics (point cloud)
- x4: Pharmacophores (points + directions)
"""

import logging
from typing import Dict, List, Optional, Any

import torch
import torch.nn.functional as F

from MolecularDiffusion import core
from MolecularDiffusion.modules.tasks.task import Task

logger = logging.getLogger(__name__)


@core.Registry.register("PharmacophoreGenerative")
class PharmacophoreGenerative(Task, core.Configurable):
    """
    Generative diffusion task for pharmacophore model.

    Handles training of the 4-modal pharmacophore diffusion model
    with per-modality loss weighting.
    """

    _option_members = {"task"}

    def __init__(
        self,
        model=None,
        modality_weights: Optional[Dict[str, float]] = None,
        loss_type: str = "mse",
        model_params: Optional[Dict] = None,
        **kwargs,
    ):
        super(PharmacophoreGenerative, self).__init__()

        # Store task type (required by training script)
        self.task_type = "diffusion_pharmacophore"

        # Create model if not provided
        if model is None:
            from MolecularDiffusion.modules.models.dynamics.pharmacophore_dynamics import PharmacophoreEnVariationalDiff
            model = PharmacophoreEnVariationalDiff(
                model_params=model_params,
            )
            logger.info("Created PharmacophoreEnVariationalDiff model")

        self.model = model
        self.loss_type = loss_type

        # Set modality weights
        if modality_weights is None:
            modality_weights = {'x1': 1.0, 'x2': 1.0, 'x3': 1.0, 'x4': 1.0}
        self.modality_weights = modality_weights

        # Training script expects task_module.task to point to the actual task
        # Use object.__setattr__ to bypass custom __setattr__ in Task base class
        object.__setattr__(self, 'task', self)

        logger.info(f"PharmacophoreGenerative initialized with modality weights: {modality_weights}")

    def build(self):
        """Build the task (no-op, model already instantiated)."""
        logger.info("PharmacophoreGenerative task built")

    def preprocess(self, train_set=None, valid_set=None, test_set=None):
        """
        Preprocess datasets.

        Args:
            train_set: Training dataset
            valid_set: Validation dataset
            test_set: Test dataset
        """
        if train_set is not None:
            if len(train_set) == 0:
                raise ValueError("Training set is empty.")
            logger.info(f"Preprocessed training set with {len(train_set)} samples")

    def forward(self, batch):
        """
        Forward pass through the model.

        Args:
            batch: HeteroData batch from pharmacophore_collate_fn

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Forward through model — batch is HeteroData, model handles conversion
        input_dict, output_dict = self.model(batch)

        # Compute losses per modality
        loss_x1 = self._compute_x1_loss(input_dict, output_dict)
        loss_x2 = self._compute_x2_loss(input_dict, output_dict)
        loss_x3 = self._compute_x3_loss(input_dict, output_dict)
        loss_x4 = self._compute_x4_loss(input_dict, output_dict)

        # Aggregate with weights
        total_loss = (
            self.modality_weights.get('x1', 1.0) * loss_x1 +
            self.modality_weights.get('x2', 1.0) * loss_x2 +
            self.modality_weights.get('x3', 1.0) * loss_x3 +
            self.modality_weights.get('x4', 1.0) * loss_x4
        )

        metrics = {
            'loss_x1': loss_x1.detach() if isinstance(loss_x1, torch.Tensor) else torch.tensor(loss_x1),
            'loss_x2': loss_x2.detach() if isinstance(loss_x2, torch.Tensor) else torch.tensor(loss_x2),
            'loss_x3': loss_x3.detach() if isinstance(loss_x3, torch.Tensor) else torch.tensor(loss_x3),
            'loss_x4': loss_x4.detach() if isinstance(loss_x4, torch.Tensor) else torch.tensor(loss_x4),
            'loss_total': total_loss.detach() if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss),
        }

        return total_loss, metrics

    def _is_mock_mode(self, output_dict: Dict, modality: str) -> bool:
        """Check if we're in mock mode (no real model output)."""
        out = output_dict.get(modality, {})
        if not out:
            return True
        decoder = out.get('decoder', {})
        if not decoder:
            return True
        denoiser = decoder.get('denoiser', {})
        return not denoiser

    def _compute_x1_loss(self, input_dict: Dict, output_dict: Dict) -> torch.Tensor:
        """Compute x1 (structure) denoising loss: pos + feature + bond."""
        if 'x1' not in input_dict:
            return torch.tensor(0.0, requires_grad=True)

        x1_dec = input_dict['x1']['decoder']
        device = x1_dec['pos'].device if 'pos' in x1_dec else torch.device('cpu')

        if self._is_mock_mode(output_dict, 'x1'):
            return torch.tensor(0.1, device=device, requires_grad=True)

        denoiser = output_dict['x1']['decoder']['denoiser']
        mask = ~x1_dec['virtual_node_mask']

        pos_loss = torch.mean(
            (x1_dec['pos_noise'] - denoiser['pos_out'])[mask] ** 2.0
        )
        feature_loss = torch.mean(
            (x1_dec['x_noise'] - denoiser['x_out'])[mask] ** 2.0
        )

        bond_loss = torch.zeros_like(feature_loss)
        if 'bond_edge_x_noise' in x1_dec and 'bond_edge_x_out' in denoiser:
            true_noise = x1_dec['bond_edge_x_noise']
            pred_noise = denoiser['bond_edge_x_out']
            bond_mask = x1_dec['bond_edge_mask']
            bond_loss = (
                torch.mean((true_noise - pred_noise)[bond_mask] ** 2.0) +
                torch.mean((true_noise - pred_noise)[~bond_mask] ** 2.0)
            ) * 0.5

        return pos_loss + feature_loss + bond_loss

    def _compute_x2_loss(self, input_dict: Dict, output_dict: Dict) -> torch.Tensor:
        """Compute x2 (shape) denoising loss: pos only."""
        if 'x2' not in input_dict:
            return torch.tensor(0.0, requires_grad=True)

        x2_dec = input_dict['x2']['decoder']
        device = x2_dec['pos'].device

        if self._is_mock_mode(output_dict, 'x2'):
            return torch.tensor(0.1, device=device, requires_grad=True)

        denoiser = output_dict['x2']['decoder']['denoiser']
        mask = ~x2_dec['virtual_node_mask']

        pos_loss = torch.mean(
            (x2_dec['pos_noise'] - denoiser['pos_out'])[mask] ** 2.0
        )
        return pos_loss

    def _compute_x3_loss(self, input_dict: Dict, output_dict: Dict) -> torch.Tensor:
        """Compute x3 (electrostatics) denoising loss: feature + pos."""
        if 'x3' not in input_dict:
            return torch.tensor(0.0, requires_grad=True)

        x3_dec = input_dict['x3']['decoder']
        device = x3_dec['pos'].device

        if self._is_mock_mode(output_dict, 'x3'):
            return torch.tensor(0.1, device=device, requires_grad=True)

        denoiser = output_dict['x3']['decoder']['denoiser']
        mask = ~x3_dec['virtual_node_mask']

        feature_loss = torch.mean(
            (x3_dec['x_noise'] - denoiser['x_out'].squeeze())[mask] ** 2.0
        )
        pos_loss = torch.mean(
            (x3_dec['pos_noise'] - denoiser['pos_out'])[mask] ** 2.0
        )
        return feature_loss + pos_loss

    def _compute_x4_loss(self, input_dict: Dict, output_dict: Dict) -> torch.Tensor:
        """Compute x4 (pharmacophores) denoising loss: feature + pos + direction."""
        if 'x4' not in input_dict:
            return torch.tensor(0.0, requires_grad=True)

        x4_dec = input_dict['x4']['decoder']
        device = x4_dec['pos'].device

        if self._is_mock_mode(output_dict, 'x4'):
            return torch.tensor(0.1, device=device, requires_grad=True)

        denoiser = output_dict['x4']['decoder']['denoiser']
        mask = ~x4_dec['virtual_node_mask']

        # Handle empty pharmacophores (all virtual nodes)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        feature_loss = torch.mean(
            (x4_dec['x_noise'] - denoiser['x_out'].squeeze())[mask] ** 2.0
        )
        pos_loss = torch.mean(
            (x4_dec['pos_noise'] - denoiser['pos_out'])[mask] ** 2.0
        )
        direction_loss = torch.mean(
            (x4_dec['direction_noise'] - denoiser['direction_out'])[mask] ** 2.0
        )
        return feature_loss + pos_loss + direction_loss

    def predict(self, batch: Dict[str, Any], all_loss=None, metric=None):
        """Return empty predictions for generative models (evaluation not supported)."""
        return torch.tensor([])

    def target(self, batch: Dict[str, Any]):
        """Return empty targets for generative models (evaluation not supported)."""
        return torch.tensor([])

    def evaluate(self, pred, target):
        """Return empty metrics for generative models (evaluation not supported)."""
        return {}
