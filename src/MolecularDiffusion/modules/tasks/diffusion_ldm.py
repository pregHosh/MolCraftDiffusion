"""
LDM (VAE + DiT) tasks for MolecularDiffusion.

This module provides:
- VAETaskFactory / VAETask: Variational Autoencoder for learning latent representations
- LDMTaskFactory / LDMTask: Latent Diffusion Model for generation

Stage 1: Train VAE to learn latent space
Stage 2: Train DiT on frozen VAE latents
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from torch_geometric.utils import to_dense_batch
from MolecularDiffusion.modules.models.en_diffusion import DistributionNodes


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution for VAE posterior."""
    
    def __init__(self, mean, logvar):
        self.mean = mean
        self.logvar = logvar
        self.std = torch.exp(0.5 * logvar)
    
    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)
    
    def kl(self):
        """KL divergence against standard normal."""
        return 0.5 * torch.sum(
            self.mean.pow(2) + self.std.pow(2) - 1 - self.logvar,
            dim=-1
        ).mean()
    
    def mode(self):
        return self.mean


class VAETaskFactory:
    """Factory for building VAE task."""
    
    def __init__(
        self,
        task_type: str,
        encoder: dict,
        decoder: dict,
        latent_dim: int = 8,
        loss_weights: Optional[dict] = None,
        eval_reconstruction: bool = False,
        save_reconstruction: bool = False,
        output_path: str = "",
        augment_rotation: bool = False,
        augment_noise: float = 0.0,
        pos_scale: float = 1.0,
        train_set: Optional[torch.utils.data.Dataset] = None,
        **kwargs
    ):
        self.task_type = task_type
        self.pos_scale = pos_scale
        self.encoder_config = encoder
        self.decoder_config = decoder
        self.latent_dim = latent_dim
        self.loss_weights = loss_weights or {"pos": 1.0, "atom_types": 1.0, "kl": 0.0001}
        self.eval_reconstruction = eval_reconstruction
        self.save_reconstruction = save_reconstruction
        self.output_path = output_path
        self.augment_rotation = augment_rotation
        self.augment_noise = augment_noise
        self.train_set = train_set
        self.kwargs = kwargs
    
    def build(self):
        """Instantiate and return the VAE task."""
        # Handle both already-instantiated modules (from Hydra recursive) and config dicts
        if isinstance(self.encoder_config, nn.Module):
            encoder = self.encoder_config
        else:
            encoder = instantiate(self.encoder_config)
        
        if isinstance(self.decoder_config, nn.Module):
            decoder = self.decoder_config
        else:
            decoder = instantiate(self.decoder_config)

        print(f"VAE pos_scale = {self.pos_scale} (decoder length unit / Angstrom)")
        self.task = VAETask(
            encoder=encoder,
            decoder=decoder,
            latent_dim=self.latent_dim,
            loss_weights=self.loss_weights,
            eval_reconstruction=self.eval_reconstruction,
            save_reconstruction=self.save_reconstruction,
            output_path=self.output_path,
            augment_rotation=self.augment_rotation,
            augment_noise=self.augment_noise,
            pos_scale=self.pos_scale,
            task_type=self.task_type,
        )
        return self.task


class VAETask(nn.Module):
    """Variational Autoencoder for molecular latent representation."""
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int = 8,
        loss_weights: Optional[dict] = None,
        eval_reconstruction: bool = False,
        save_reconstruction: bool = False,
        output_path: str = "",
        augment_rotation: bool = False,
        augment_noise: float = 0.0,
        pos_scale: float = 1.0,
        task_type: str = "vae",
    ):
        super().__init__()

        # Length unit the decoder's `pos_head` works in, relative to Angstrom.
        # 1.0 = Angstrom directly (every checkpoint trained before this key).
        # Upstream ADiT trains in nanometres (10.0): the loss target is
        # divided by it and every decoded position multiplied back.
        self.pos_scale = pos_scale
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.loss_weights = loss_weights or {"pos": 1.0, "atom_types": 1.0, "kl": 0.0001}
        self.eval_reconstruction = eval_reconstruction
        self.save_reconstruction = save_reconstruction
        self.output_path = output_path
        self.augment_rotation = augment_rotation
        self.augment_noise = augment_noise
        
        # Quantization layers (encoder hidden -> latent)
        d_model = encoder.d_model
        self.quant_conv = nn.Linear(d_model, 2 * latent_dim, bias=False)
        self.post_quant_conv = nn.Linear(latent_dim, decoder.d_model, bias=False)
        
        # Must be the *configured* name, not a generic one: the checkpoint
        # stamps it and `cli/generate.py` refuses a mismatching config.
        self.task_type = task_type
        self.split = "train"  # Track train/valid/test
        self._batch_idx = 0  # For sample indexing
        
        # Reconstruction evaluator (optional)
        if self.eval_reconstruction:
            try:
                from MolecularDiffusion.utils.geom_utils import MoleculeReconstructionEvaluator
                self.reconstruction_evaluator = MoleculeReconstructionEvaluator()
            except ImportError as e:
                print(f"Warning: Could not import MoleculeReconstructionEvaluator: {e}")
                self.reconstruction_evaluator = None
        else:
            self.reconstruction_evaluator = None
        
    def _adapt_batch(self, batch):
        """Adapt batch to encoder input format.
        
        Handles:
        - dict with 'graph' key (from Lightning DataModule wrapping PyG)
        - dict format (from PointCloud collator)
        - PyG Data object format (from graph collator)
        """
        
        # Check if batch is a dict wrapping a PyG Batch (e.g., from Lightning)
        if isinstance(batch, dict) and 'graph' in batch:
            batch = batch['graph']  # Unwrap the PyG Batch
        if isinstance(batch, dict):
            # Convert dict to a simple namespace-like object for attribute access
            class BatchWrapper:
                pass
            wrapped = BatchWrapper()
            
            # Map PointCloud fields to expected names
            # PointCloud collator uses: coords, charges, node_feature, node_mask, natoms
            if "coords" in batch:
                wrapped.pos = batch["coords"]
            elif "pos" in batch:
                wrapped.pos = batch["pos"]
            else:
                raise KeyError(f"Batch dict missing 'coords' or 'pos'. Available keys: {list(batch.keys())}")
            
            if "charges" in batch:
                wrapped.atom_types = batch["charges"].long()
            elif "atomic_numbers" in batch:
                wrapped.atom_types = batch["atomic_numbers"].long()
            else:
                raise KeyError(f"Batch dict missing 'charges' or 'atomic_numbers'. Available keys: {list(batch.keys())}")
            wrapped.atomic_numbers = wrapped.atom_types
            
            # Handle natoms - can be tensor or need computing from node_mask
            if "natoms" in batch:
                wrapped.num_atoms = batch["natoms"]
            elif "node_mask" in batch:
                wrapped.num_atoms = batch["node_mask"].sum(dim=1).long()
            else:
                wrapped.num_atoms = torch.tensor([wrapped.pos.size(1)], device=wrapped.pos.device)
            wrapped.natoms = wrapped.num_atoms
            
            # Create batch indices from natoms
            B, N = wrapped.pos.shape[:2]
            wrapped.batch = torch.arange(B, device=wrapped.pos.device).repeat_interleave(
                torch.tensor([N] * B, device=wrapped.pos.device)
            )
            
            # Token indices
            wrapped.token_idx = torch.arange(N, device=wrapped.pos.device).repeat(B)
            
            # Flatten pos and atom_types from (B, N, ...) to (B*N, ...)
            node_mask = batch.get("node_mask", torch.ones(B, N, device=wrapped.pos.device))
            flat_mask = node_mask.bool().flatten()
            wrapped.pos = wrapped.pos.reshape(-1, 3)[flat_mask]
            wrapped.atom_types = wrapped.atom_types.flatten()[flat_mask]
            wrapped.atomic_numbers = wrapped.atom_types
            wrapped.batch = wrapped.batch[flat_mask]
            wrapped.token_idx = wrapped.token_idx[flat_mask]
            
            # Silenced defaults for molecules. The cell is all-zero, as the
            # PyG molecule caches store it: the Equiformer encoder turns cell
            # vectors into edge-angle features, and a fixed non-zero cell
            # would leak absolute orientation into them. A zero cell makes
            # those features 0/0 = NaN, which the encoder maps to 0.
            wrapped.frac_coords = torch.zeros_like(wrapped.pos)
            wrapped.cell = wrapped.pos.new_zeros(B, 3, 3)
            
            return wrapped
        
        # PyG Data/Batch format
        if not hasattr(batch, 'atom_types'):
            batch.atom_types = batch.atomic_numbers
        if not hasattr(batch, 'num_atoms'):
            batch.num_atoms = batch.natoms
        # Silenced defaults for molecules
        if not hasattr(batch, 'frac_coords'):
            batch.frac_coords = torch.zeros_like(batch.pos)
        if not hasattr(batch, 'cell'):  # zero cell: see the dict branch above
            batch.cell = batch.pos.new_zeros(batch.num_atoms.size(0), 3, 3)
        return batch
    
    def encode(self, batch):
        """Encode batch to latent distribution."""
        batch = self._adapt_batch(batch)

        # The reconstruction loss is translation-invariant, but the
        # Transformer encoder embeds Cartesian positions directly. Center a
        # shallow copy so arbitrary database offsets cannot leak into the
        # VAE/ADiT latent while leaving the caller's coordinates unchanged.
        encoder_batch = copy.copy(batch)
        pos_mean = scatter(batch.pos, batch.batch, dim=0, reduce="mean")
        encoder_batch.pos = batch.pos - pos_mean[batch.batch]
        encoded = self.encoder(encoder_batch)
        
        # Project to latent: mean and logvar
        h = self.quant_conv(encoded["x"])  # (n, 2*latent_dim)
        mean, logvar = h.chunk(2, dim=-1)
        
        posterior = DiagonalGaussianDistribution(mean, logvar)
        
        return posterior, encoded
    
    def decode(self, z, encoded):
        """Decode latent to reconstruction."""
        # Project back to decoder dimension
        x = self.post_quant_conv(z)
        
        # Prepare decoder input
        decoder_input = {
            "x": x,
            "num_atoms": encoded["num_atoms"],
            "batch": encoded["batch"],
            "token_idx": encoded["token_idx"],
        }
        
        return self.decoder(decoder_input)
    
    def forward(self, batch):
        """Training forward pass.
        
        Args:
            batch: PyG Data batch with atom_types/atomic_numbers, pos, etc.
            
        Returns:
            loss: Scalar training loss
            stats: Dictionary of loss components
        """
        batch = self._adapt_batch(batch)
        
        # Random rotation augmentation (training only)
        if self.training and self.augment_rotation:
            from MolecularDiffusion.utils import random_rotation_matrix, apply_rotation_augmentation
            with torch.no_grad():
                rot_mat = random_rotation_matrix(device=batch.pos.device, dtype=batch.pos.dtype)
                apply_rotation_augmentation(batch, rot_mat, rotate_cell=False)  # Disabled for molecules
        
        # Noise augmentation (training only)
        # Mimics Denoising VAE: corrupt input, but latent learns to reconstruct CLEAN input
        clean_batch_pos = None
        clean_batch_atom_types = None
        
        if self.training and self.augment_noise > 0.0:
            import numpy as np
            clean_batch_pos = batch.pos.clone()
            clean_batch_atom_types = batch.atom_types.clone()
            
            total_atoms = batch.num_atoms.sum().item()
            num_perturbed = int(total_atoms * self.augment_noise)
            
            if num_perturbed > 0:
                # 1. Mask atom types (set to 0)
                perturbed_idx_types = torch.from_numpy(
                    np.random.choice(total_atoms, num_perturbed, replace=False)
                ).to(batch.pos.device)
                batch.atom_types[perturbed_idx_types] = 0
                batch.atomic_numbers = batch.atom_types # Sync
                
                # 2. Add noise to positions
                perturbed_idx_pos = torch.from_numpy(
                    np.random.choice(total_atoms, num_perturbed, replace=False)
                ).to(batch.pos.device)
                corruption_scale = 0.1
                noise = torch.randn_like(batch.pos[perturbed_idx_pos]) * corruption_scale
                batch.pos[perturbed_idx_pos] += noise
        
        # Encode
        posterior, encoded = self.encode(batch)
        z = posterior.sample()
        
        # Decode
        recon = self.decode(z, encoded)
        
        # Restore clean data for loss calculation
        if clean_batch_pos is not None:
             batch.pos = clean_batch_pos
             batch.atom_types = clean_batch_atom_types
             batch.atomic_numbers = clean_batch_atom_types
        
        # Losses
        # Position reconstruction MSE - zero-centered for translation invariance

        pos_pred = recon["pos"]
        pos_true = batch.pos / self.pos_scale
        batch_idx = encoded["batch"]
        # Subtract per-molecule centroids
        pos_mean_pred = scatter(pos_pred, batch_idx, dim=0, reduce="mean")[batch_idx]
        pos_mean_true = scatter(pos_true, batch_idx, dim=0, reduce="mean")[batch_idx]
        pos_loss = F.mse_loss(pos_pred - pos_mean_pred, pos_true - pos_mean_true)
        
        # Atom type cross-entropy
        atom_loss = F.cross_entropy(
            recon["atom_types"],
            batch.atom_types
        )
        
        # KL divergence
        kl_loss = posterior.kl()
        
        # Weighted total loss
        loss = (
            self.loss_weights["pos"] * pos_loss +
            self.loss_weights["atom_types"] * atom_loss +
            self.loss_weights["kl"] * kl_loss
        )
        
        stats = {
            "loss": loss.item(),
            "pos_loss": pos_loss.item(),
            "atom_loss": atom_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        
        # Collect predictions for reconstruction evaluation during validation
        if self.eval_reconstruction and self.reconstruction_evaluator is not None:
            if self.split in ("valid", "val", "test"):
                self._collect_reconstruction_samples(batch, recon, encoded)
        
        return loss, stats
    
    def _collect_reconstruction_samples(self, batch, recon, encoded):
        """Collect predicted and ground truth samples for reconstruction evaluation.
        
        Centers positions per-molecule for proper RMSD/match computation.
        """
        num_atoms = encoded["num_atoms"]
        batch_idx = encoded["batch"]
        
        # Get predicted atom types (argmax of logits)
        pred_atom_types = recon["atom_types"].argmax(dim=-1)  # (n_total,)
        pred_pos = recon["pos"] * self.pos_scale  # (n_total, 3), back to Angstrom
        
        # Get ground truth
        gt_atom_types = batch.atom_types  # (n_total,)
        gt_pos = batch.pos  # (n_total, 3)
        
        # Split by molecule in batch and CENTER each molecule
        start_idx = 0
        for mol_idx, n_atoms in enumerate(num_atoms.tolist()):
            sample_idx = self._batch_idx * len(num_atoms) + mol_idx
            
            # Extract molecule positions
            pred_mol_pos = pred_pos[start_idx:start_idx + n_atoms].detach()
            gt_mol_pos = gt_pos[start_idx:start_idx + n_atoms].detach()
            
            # Center each molecule (subtract centroid)
            pred_mol_pos_centered = pred_mol_pos - pred_mol_pos.mean(dim=0, keepdim=True)
            gt_mol_pos_centered = gt_mol_pos - gt_mol_pos.mean(dim=0, keepdim=True)
            
            # Predicted (centered)
            self.reconstruction_evaluator.append_pred_array({
                "atom_types": pred_atom_types[start_idx:start_idx + n_atoms].detach().cpu().numpy(),
                "pos": pred_mol_pos_centered.cpu().numpy(),
                "sample_idx": sample_idx,
            })
            
            # Ground truth (centered)
            self.reconstruction_evaluator.append_gt_array({
                "atom_types": gt_atom_types[start_idx:start_idx + n_atoms].detach().cpu().numpy(),
                "pos": gt_mol_pos_centered.cpu().numpy(),
                "sample_idx": sample_idx,
            })
            
            start_idx += n_atoms
        
        self._batch_idx += 1
    
    def on_validation_epoch_start(self):
        """Clear evaluator at the start of validation epoch."""
        self._batch_idx = 0
        if self.reconstruction_evaluator is not None:
            self.reconstruction_evaluator.clear()
    
    def on_validation_epoch_end(self, current_epoch: int = 0) -> Dict:
        """Compute reconstruction metrics at the end of validation epoch.
        
        Args:
            current_epoch: Current epoch number
        
        Returns:
            Dict of reconstruction metrics (match_rate, mean_rms_dist)
        """
        if self.reconstruction_evaluator is None:
            return {}
        
        if len(self.reconstruction_evaluator.pred_arrays_list) == 0:
            return {}
        
        # Set device for evaluator
        self.reconstruction_evaluator.device = next(self.parameters()).device
        
        metrics = self.reconstruction_evaluator.get_metrics(current_epoch=current_epoch)
        
        # Save XYZ files if enabled and output_path is set
        if self.save_reconstruction and self.output_path:
            import os
            save_dir = os.path.join(self.output_path, "reconstruct_xyz", f"epoch_{current_epoch}")
            print(f"Saving reconstructed molecules to {save_dir}")
            self.reconstruction_evaluator.save_molecules(save_dir)
        
        return metrics
    
    def reconstruct(self, batch, sample=True):
        """Reconstruct input batch."""
        posterior, encoded = self.encode(batch)
        z = posterior.sample() if sample else posterior.mode()
        return self.decode(z, encoded)

    def predict_and_target(self, batch):
        """Evaluation pass for Engine compatibility."""
        loss, stats = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        dummy_target = torch.zeros_like(loss)
        return loss, dummy_target

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics from aggregated predictions."""
        metrics = {"val_loss": pred.mean()}
        recon_metrics = self.on_validation_epoch_end(current_epoch=0)
        metrics.update(recon_metrics)
        self.on_validation_epoch_start()  # Reset for next evaluation
        return metrics


class LDMTaskFactory:
    """Factory for building LDM (Latent Diffusion Model) task."""
    
    def __init__(
        self,
        task_type: str,
        autoencoder_ckpt: str,
        denoiser: dict,
        interpolant: dict,
        augment_rotation: bool = False,
        train_set: Optional[torch.utils.data.Dataset] = None,
        allow_autoencoder_override: bool = False,
        **kwargs
    ):
        self.task_type = task_type
        self.autoencoder_ckpt = autoencoder_ckpt
        # A checkpoint's own VAE must not replace the one named above; see LDMTask.load_state_dict.
        self.allow_autoencoder_override = allow_autoencoder_override
        self.denoiser_config = denoiser
        self.interpolant_config = interpolant
        self.augment_rotation = augment_rotation
        self.train_set = train_set
        self.kwargs = kwargs
    
    def build(self):
        """Instantiate and return the LDM task."""
        # Load frozen VAE
        print(f"Loading VAE from: {self.autoencoder_ckpt}")
        vae_checkpoint = torch.load(self.autoencoder_ckpt, map_location="cpu", weights_only=False)
        
        # Handle both raw state_dict and Lightning checkpoint
        if "state_dict" in vae_checkpoint:
            state_dict = vae_checkpoint["state_dict"]
            # Remove 'task.' prefix if present
            state_dict = {k.replace("task.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = vae_checkpoint
        
        # Rebuild VAE from config in checkpoint
        if self.autoencoder_ckpt.endswith(".pkl"):
             hparams = vae_checkpoint.get("hyperparameters", {})
             vae_config = hparams.get("task") # Engine format
        else:
             hparams = vae_checkpoint.get("hyper_parameters", {})
             vae_config = hparams.get("model_config", hparams.get("tasks", None)) # Lightning format
        
        # Debug: print checkpoint structure
        if vae_config is None:
            print(f"[DEBUG] Checkpoint keys: {list(vae_checkpoint.keys())}")
            print(f"[DEBUG] hparams keys: {list(hparams.keys()) if hparams else 'None'}")
            raise ValueError(
                f"Cannot extract VAE config from checkpoint. "
                f"Expected 'task' (for .pkl) or 'model_config'/'tasks' (for .ckpt) in hyperparameters. "
                f"Checkpoint keys: {list(vae_checkpoint.keys())}, "
                f"hparams keys: {list(hparams.keys()) if hparams else 'None'}"
            )
        
        # If vae_config is already a built model (deserialized as object)
        if hasattr(vae_config, "load_state_dict") and not isinstance(vae_config, dict):
             vae = vae_config
             print("VAE loaded directly from checkpoint object")
        else:
             # Ensure vae_config is a dict (it might be an object if deserialized with config_dict)
             if hasattr(vae_config, "config_dict"):
                 vae_config = vae_config.config_dict()
             
             # Ensure 'task_type', 'encoder', 'decoder' are present
             if vae_config is not None and "task_type" not in vae_config:
                 if "tasks" in vae_config:
                     vae_config = vae_config["tasks"]
                 else:
                     # Remove non-factory keys that might be saved in model_config
                     vae_config = {k: v for k, v in vae_config.items() if k not in ("_target_",)}

             vae_factory = VAETaskFactory(**vae_config)
             vae = vae_factory.build()

        vae.load_state_dict(state_dict, strict=False)
        vae.eval()
        vae.requires_grad_(False)
        
        # Handle both already-instantiated module and config dict
        if isinstance(self.denoiser_config, nn.Module):
            denoiser = self.denoiser_config
            # Verify d_x matches VAE latent_dim
            if hasattr(denoiser, "d_x") and denoiser.d_x != vae.latent_dim:
                print(f"WARNING: Denoiser d_x ({denoiser.d_x}) does not match VAE latent_dim ({vae.latent_dim})!")
                print(f"Re-instantiating denoiser with d_x = {vae.latent_dim}")
        if not isinstance(self.denoiser_config, nn.Module):
            # Auto-infer d_x from VAE latent_dim
            if hasattr(self.denoiser_config, '_target_') or isinstance(self.denoiser_config, (dict, DictConfig)):
                denoiser_cfg = OmegaConf.to_container(self.denoiser_config, resolve=True) if not isinstance(self.denoiser_config, (dict, DictConfig)) else dict(self.denoiser_config)
                denoiser_cfg["d_x"] = vae.latent_dim  # Override with VAE's latent_dim
                print(f"Auto-setting DiT d_x = {vae.latent_dim} from VAE latent_dim")
                denoiser = instantiate(OmegaConf.create(denoiser_cfg))
            else:
                denoiser = instantiate(self.denoiser_config)
        
        self.task = LDMTask(
            autoencoder=vae,
            denoiser=denoiser,
            interpolant_config=self.interpolant_config,
            augment_rotation=self.augment_rotation,
            task_type=self.task_type,
            autoencoder_ckpt=self.autoencoder_ckpt,
            allow_autoencoder_override=self.allow_autoencoder_override,
        )
        return self.task


class FlowMatchingInterpolant:
    """Flow matching interpolant for latent diffusion.
    
    Matches reference implementation from all-atom-diffusion-transformer.
    Key conventions:
    - x_0 is NOISE (prior)
    - x_1 is CLEAN DATA (target)
    - Time flows from 0 (noise) to 1 (clean)
    
    Args:
        min_t: Minimum time to sample (avoids boundary singularities)
        corrupt: Whether to corrupt samples during training
        num_timesteps: Number of timesteps for sampling
        self_condition: Whether to use self-conditioning
        self_condition_prob: Probability of using self-conditioning during training
    """
    
    def __init__(
        self,
        min_t: float = 1e-2,
        corrupt: bool = True,
        num_timesteps: int = 100,
        self_condition: bool = False,
        self_condition_prob: float = 0.5,
    ):
        self.min_t = min_t
        self.corrupt = corrupt
        self.num_timesteps = num_timesteps
        self.self_condition = self_condition
        self.self_condition_prob = self_condition_prob
        self.device = "cpu"  # Will be set at runtime
    
    def _sample_t(self, batch_size):
        """Sample timesteps in [min_t, 1 - min_t] range."""
        t = torch.rand(batch_size, device=self.device)
        return t * (1 - 2 * self.min_t) + self.min_t
    
    def _centered_gaussian(self, batch_size, num_tokens, emb_dim):
        """Sample mean-centered Gaussian noise for translation invariance."""
        noise = torch.randn(batch_size, num_tokens, emb_dim, device=self.device)
        return noise - torch.mean(noise, dim=-2, keepdims=True)
    
    def _corrupt_x(self, x_1, t, token_mask, diffuse_mask):
        """Corrupt clean data x_1 to noisy x_t.
        
        x_t = (1-t) * x_0 + t * x_1, where x_0 is noise, x_1 is clean
        """
        x_0 = self._centered_gaussian(*x_1.shape)
        x_t = (1 - t[..., None]) * x_0 + t[..., None] * x_1
        # Apply masks: only diffuse where diffuse_mask is True
        x_t = x_t * diffuse_mask[..., None] + x_1 * (~diffuse_mask[..., None])
        return x_t * token_mask[..., None]
    
    def corrupt_batch(self, batch):
        """Corrupts a batch of data for training.
        
        Args:
            batch: Dict with keys:
                - x_1: Clean data (B, N, d)
                - token_mask: Valid token mask (B, N)
                - diffuse_mask: Mask for tokens to diffuse (B, N)
        
        Returns:
            Dict with added keys:
                - x_t: Noisy data
                - t: Sampled timesteps
        """
        import copy
        noisy_batch = copy.copy(batch)
        
        x_1 = batch["x_1"]
        token_mask = batch["token_mask"]
        diffuse_mask = batch.get("diffuse_mask", token_mask)  # Default: diffuse all valid tokens
        batch_size = x_1.shape[0]
        
        # Sample timesteps [B, 1]
        t = self._sample_t(batch_size)[:, None]
        noisy_batch["t"] = t
        
        # Apply corruption
        if self.corrupt:
            x_t = self._corrupt_x(x_1, t, token_mask, diffuse_mask)
        else:
            x_t = x_1
        
        if torch.any(torch.isnan(x_t)):
            raise ValueError("NaN in x_t during corruption")
        
        noisy_batch["x_t"] = x_t
        return noisy_batch
    
    def _x_vector_field(self, t, x_1, x_t):
        """Compute flow velocity field: dx/dt = (x_1 - x_t) / (1 - t)"""
        return (x_1 - x_t) / (1 - t)
    
    def _x_euler_step(self, d_t, t, x_1, x_t):
        """Take one Euler step in the flow ODE."""
        assert d_t > 0
        x_vf = self._x_vector_field(t, x_1, x_t)
        return x_t + x_vf * d_t
    
    def sample(
        self,
        batch_size,
        num_tokens,
        emb_dim,
        model,
        mask=None,
        num_timesteps=None,
        x_0=None,
    ):
        """Generate samples by integrating the flow ODE.
        
        Args:
            batch_size: Number of samples
            num_tokens: Max tokens per sample
            emb_dim: Latent dimension
            model: Denoiser model (predicts x_1 from x_t, t)
            mask: Valid token mask (B, N)
            num_timesteps: Number of integration steps
            x_0: Initial noise (if None, sampled)
        
        Returns:
            Dict with tokens_traj and clean_traj lists
        """
        # Set up initial prior samples
        if x_0 is None:
            x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
        if mask is None:
            mask = torch.ones(batch_size, num_tokens, device=self.device, dtype=torch.bool)
        
        # Set up time schedule
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        ts = torch.linspace(self.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        
        tokens_traj = [x_0]
        clean_traj = []
        x_sc = None
        
        for t_2 in ts[1:]:
            x_t_1 = tokens_traj[-1]
            t = torch.ones((batch_size, 1), device=self.device) * t_1
            d_t = t_2 - t_1
            
            # Run denoiser model
            with torch.no_grad():
                pred_x_1 = model(x_t_1, t.squeeze(1), mask, x_sc=x_sc)
            
            # Store prediction
            clean_traj.append(pred_x_1)
            if self.self_condition:
                x_sc = pred_x_1
            
            # Take Euler step
            x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1, x_t_1)
            tokens_traj.append(x_t_2)
            t_1 = t_2
        
        # Final step at t=1
        t_1 = ts[-1]
        x_t_1 = tokens_traj[-1]
        t = torch.ones((batch_size, 1), device=self.device) * t_1
        with torch.no_grad():
            pred_x_1 = model(x_t_1, t.squeeze(1), mask, x_sc=x_sc)
        clean_traj.append(pred_x_1)
        tokens_traj.append(pred_x_1)
        
        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}
    
    def __repr__(self):
        return f"{self.__class__.__name__}(num_timesteps={self.num_timesteps}, self_condition={self.self_condition})"


class LDMTask(nn.Module):
    """Latent Diffusion Model task."""
    
    def __init__(
        self,
        autoencoder: VAETask,
        denoiser: nn.Module,
        interpolant_config: dict,
        augment_rotation: bool = False,
        task_type: str = "ldm",
        autoencoder_ckpt: Optional[str] = None,
        allow_autoencoder_override: bool = False,
    ):
        super().__init__()
        
        self.autoencoder = autoencoder
        # Which VAE the config asked for, and whether a checkpoint may replace it.
        self.autoencoder_ckpt = autoencoder_ckpt
        self.allow_autoencoder_override = allow_autoencoder_override
        self.denoiser = denoiser
        self.augment_rotation = augment_rotation
        
        # Freeze autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        # Interpolant
        self.interpolant = FlowMatchingInterpolant(
            min_t=interpolant_config.get("min_t", 1e-2),
            corrupt=interpolant_config.get("corrupt", True),
            num_timesteps=interpolant_config.get("num_timesteps", 100),
            self_condition=interpolant_config.get("self_condition", False),
            self_condition_prob=interpolant_config.get("self_condition_prob", 0.5),
        )
        
        # The configured name (e.g. `diffusion_adit`), as for VAETask above.
        self.task_type = task_type

        # For compatibility with generation
        self.prop_dist_model = None
        self.max_n_nodes = 100
        self.node_dist_model = None

    @property

    def T(self):
        """Diffusion steps (T) for compatibility with GenerativeFactory."""
        return self.interpolant.num_timesteps

    @T.setter
    def T(self, value):
        self.interpolant.num_timesteps = value

    @property
    def fm_num_timesteps(self):
        """Flow matching steps alias for compatibility."""
        return self.T

    @fm_num_timesteps.setter
    def fm_num_timesteps(self, value):
        self.T = value

    @property
    def model(self):
        """tasks_generate.py compatibility: exposes self as the model interface."""
        return self

    def preprocess(self, train_set):
        """Compute node distribution from training set."""
        if train_set is None:
            return
            
        # Determine node counts
        n_node_dist = {}
        max_n_nodes = 0
        
        # Determine dataset format and iterate
        # Assuming PyG dataset (list of Data or similar)
        # Check first item to guess format like in diffusion.py
        is_graph_dict = False
        if len(train_set) > 0:
            if isinstance(train_set[0], dict) and "graph" in train_set[0]:
                is_graph_dict = True
        
        # It's expensive to iterate if dataset is large, but that's what diffusion.py does
        # We can optimize if needed, but stick to diffusion.py logic for now
        for sample in train_set:
            if is_graph_dict:
                n_node = sample["graph"].num_nodes
            elif hasattr(sample, "num_nodes"):
                n_node = int(sample.num_nodes)
            elif "num_atoms" in sample: # Dict case
                 n_node = int(sample["num_atoms"])
            elif "pos" in sample: # Dict case fallback
                 n_node = sample["pos"].shape[0]
            else:
                 continue # Should not happen
                 
            if n_node > max_n_nodes:
                max_n_nodes = n_node
                
            if n_node in n_node_dist:
                n_node_dist[n_node] += 1
            else:
                n_node_dist[n_node] = 1
                
        self.n_node_dist = n_node_dist
        self.max_n_nodes = max_n_nodes
        
        print("---------------Creating node distribution model-----------------")
        self.node_dist_model = DistributionNodes(self.n_node_dist)
    
    @torch.no_grad()
    def encode(self, batch):
        """Encode batch with frozen VAE."""
        self.autoencoder.eval()
        posterior, encoded = self.autoencoder.encode(batch)
        z = posterior.mode()  # Use mode for training stability
        return z, encoded
    
    def forward(self, batch):
        """Training forward pass.
        
        Matches reference implementation:
        - Model predicts clean data x_1 (not noise)
        - Uses time-dependent loss scaling: norm_scale = 1 - min(t, 0.9)
        - Uses corrupt_batch for proper flow matching
        
        Args:
            batch: PyG Data batch
            
        Returns:
            loss: Scalar denoising loss
            stats: Dictionary of loss components
        """
        import random
        import numpy as np
        
        # Adapt batch first (handles dict -> PyG conversion)
        batch = self.autoencoder._adapt_batch(batch)
        
        # Random rotation augmentation (training only, before encoding)
        if self.training and self.augment_rotation:
            from MolecularDiffusion.utils import random_rotation_matrix, apply_rotation_augmentation
            with torch.no_grad():
                rot_mat = random_rotation_matrix(device=batch.pos.device, dtype=batch.pos.dtype)
                apply_rotation_augmentation(batch, rot_mat, rotate_cell=False)
        
        # Encode to latent with frozen VAE
        with torch.no_grad():
            posterior, encoded = self.autoencoder.encode(batch)
            # Sample from posterior (reference uses .sample(), not .mode())
            x_1 = posterior.sample()
        
        device = x_1.device
        batch_idx = encoded["batch"]
        
        # Convert to dense batch
        x_1, mask = to_dense_batch(x_1, batch_idx)
        B, N, d = x_1.shape
        
        # Set interpolant device
        self.interpolant.device = device
        
        # Prepare batch dict for interpolant
        dense_batch = {
            "x_1": x_1,
            "token_mask": mask,
            "diffuse_mask": mask,  # Diffuse all valid tokens
        }
        
        # Corrupt batch using interpolant
        noisy_batch = self.interpolant.corrupt_batch(dense_batch)
        
        # Self-conditioning
        x_sc = None
        if self.interpolant.self_condition and random.random() < self.interpolant.self_condition_prob:
            with torch.no_grad():
                x_sc = self.denoiser(
                    noisy_batch["x_t"],
                    noisy_batch["t"].squeeze(1),
                    mask,
                    x_sc=None
                )
        
        # Run denoiser - predicts clean x_1
        pred_x_1 = self.denoiser(
            noisy_batch["x_t"],
            noisy_batch["t"].squeeze(1),
            mask,
            x_sc=x_sc
        )
        
        # Compute loss with time-dependent normalization (reference criterion)
        gt_x_1 = noisy_batch["x_1"]
        t = noisy_batch["t"]  # (B, 1)
        
        # norm_scale = 1 - min(t, 0.9)
        norm_scale = 1 - torch.min(t, torch.tensor(0.9, device=device))
        x_error = (gt_x_1 - pred_x_1) / norm_scale.unsqueeze(-1)
        
        # Masked loss
        loss_mask = noisy_batch["token_mask"] * noisy_batch["diffuse_mask"]
        loss_denom = torch.sum(loss_mask, dim=-1, keepdim=True) * pred_x_1.size(-1)
        loss_denom = loss_denom.clamp(min=1.0)  # Avoid division by zero
        
        x_loss = torch.sum(x_error**2 * loss_mask.unsqueeze(-1), dim=(-1, -2)) / loss_denom.squeeze(-1)
        loss = x_loss.mean()
        
        # Build stats dict with time-stratified logging
        stats = {"loss": loss.item(), "x_loss": x_loss.mean().item()}
        
        # Time-stratified loss
        num_bins = 4
        flat_losses = x_loss.detach().cpu().numpy().flatten()
        flat_t = t.detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
        
        for t_bin in range(num_bins):
            bin_mask = bin_idx == t_bin
            if bin_mask.sum() > 0:
                bin_start = int(bin_edges[t_bin] * 100)
                bin_end = int(bin_edges[t_bin + 1] * 100)
                stats[f"x_loss t=[{bin_start},{bin_end})"] = float(flat_losses[bin_mask].mean())
        
        stats["t_avg"] = float(flat_t.mean())
        
        return loss, stats
    
    def predict_and_target(self, batch):
        """Evaluation pass for Engine compatibility."""
        loss, stats = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        dummy_target = torch.zeros_like(loss)
        return loss, dummy_target

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics from aggregated predictions."""
        return {"val_loss": pred.mean()}
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        nodesxsample: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        **kwargs
    ):
        """Generate molecules by sampling from the latent diffusion model.
        
        Uses proper flow matching ODE integration from min_t to 1.
        
        Args:
            batch_size: Number of molecules to generate
            nodesxsample: Tensor of molecule sizes
            num_steps: Number of denoising steps (defaults to self.interpolant.num_timesteps)
            
        Returns:
            Tuple of (one_hot, charges, coords, node_mask)
        """
        # Default num_steps to self.interpolant.num_timesteps
        if num_steps is None:
            num_steps = self.interpolant.num_timesteps

        device = next(self.parameters()).device
        
        if nodesxsample is None:
            nodesxsample = torch.randint(5, 30, (batch_size,))
        
        batch_size = len(nodesxsample)
        max_atoms = int(nodesxsample.max().item())
        
        # Create mask
        mask = torch.arange(max_atoms, device=device)[None, :] < nodesxsample[:, None].to(device)
        
        latent_dim = self.autoencoder.latent_dim
        
        # Set interpolant device
        self.interpolant.device = device
        
        # Use interpolant's sample method for proper flow ODE integration
        sample_result = self.interpolant.sample(
            batch_size=batch_size,
            num_tokens=max_atoms,
            emb_dim=latent_dim,
            model=self.denoiser,
            mask=mask,
            num_timesteps=num_steps,
        )
        
        # Get final clean prediction
        z_clean = sample_result["clean_traj"][-1]
        
        # Decode with VAE
        z_flat = z_clean[mask]  # Flatten to (n_total, latent_dim)
        
        # Reconstruct token index and batch
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(nodesxsample.to(device))
        token_idx = torch.cat([torch.arange(n, device=device) for n in nodesxsample])
        
        # Project and decode
        x = self.autoencoder.post_quant_conv(z_flat)
        decoder_input = {
            "x": x,
            "num_atoms": nodesxsample.to(device),
            "batch": batch_idx,
            "token_idx": token_idx,
        }
        recon = self.autoencoder.decoder(decoder_input)
        
        # Convert to output format
        atom_logits = recon["atom_types"]  # (n_total, max_elements)
        atom_types = atom_logits.argmax(dim=-1)  # (n_total,)
        # getattr: a VAE object pickled before `pos_scale` existed is Angstrom.
        coords = recon["pos"] * getattr(self.autoencoder, "pos_scale", 1.0)  # (n_total, 3)
        
        # Pad to dense format
        one_hot = torch.zeros(batch_size, max_atoms, atom_logits.size(-1), device=device)
        charges = torch.zeros(batch_size, max_atoms, dtype=torch.long, device=device)
        pos = torch.zeros(batch_size, max_atoms, 3, device=device)
        node_mask = mask.float()
        
        # Fill in values
        flat_idx = 0
        for i, n in enumerate(nodesxsample):
            one_hot[i, :n] = F.softmax(atom_logits[flat_idx:flat_idx+n], dim=-1)
            charges[i, :n] = atom_types[flat_idx:flat_idx+n]
            pos[i, :n] = coords[flat_idx:flat_idx+n]
            flat_idx += n
        
        return one_hot, charges, pos, node_mask
    
    @property
    def device(self):
        return next(self.parameters()).device

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Never let a checkpoint's own VAE replace the one `tasks.autoencoder_ckpt` loaded.

        Both paths land here: the warm start (`trainer.load_weights_from`) and a full
        checkpoint restore. An LDM checkpoint stores `autoencoder.*` beside `denoiser.*`,
        with identical names and shapes to the VAE this task has just built and FROZEN from
        the config, so those tensors overwrite it silently -- no error, no shape mismatch.
        That is how two 100k-step fine-tunes ended up training through the wrong latent
        space. The configured VAE wins; the checkpoint's copy is dropped and the denoiser
        loads as usual. Set `tasks.allow_autoencoder_override: true` to restore the old
        behaviour (e.g. to reproduce a checkpoint trained before this guard existed).
        """
        incoming = {k: v for k, v in state_dict.items() if k.startswith("autoencoder.")}
        if incoming and getattr(self, "autoencoder_ckpt", None) \
                and not getattr(self, "allow_autoencoder_override", False):
            current = self.state_dict()
            differing = sum(
                1
                for k, v in incoming.items()
                if k in current
                and (v.shape != current[k].shape
                     or not torch.equal(v.detach().cpu(), current[k].detach().cpu()))
            )
            if differing:
                print(
                    f"[LDMTask] Ignoring {len(incoming)} autoencoder.* tensor(s) from the "
                    f"incoming state_dict: {differing} of them differ from the VAE loaded "
                    f"from tasks.autoencoder_ckpt ({self.autoencoder_ckpt}), which is kept. "
                    f"Only the denoiser is loaded. Pass tasks.allow_autoencoder_override=true "
                    f"to let the checkpoint's VAE win instead."
                )
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith("autoencoder.")}
            strict = False
        return super().load_state_dict(state_dict, strict=strict, **kwargs)


