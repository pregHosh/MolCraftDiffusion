"""
Diffusion Transformer (DiT) denoiser.

Copyright (c) Meta Platforms, Inc. and affiliates.
Adapted from: https://github.com/facebookresearch/DiT
"""

import math

import torch
import torch.nn as nn


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_dim, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, 0, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def get_pos_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings."""
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


#################################################################################
#                               Transformer blocks                              #
#################################################################################


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def modulate(x, shift, scale):
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(_x, _x, _x, key_padding_mask=mask, need_weights=False)[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiT(nn.Module):
    """Diffusion Transformer denoiser.

    Args:
        d_x: Latent token dimension (must match VAE latent_dim)
        d_model: Model hidden dimension
        num_layers: Number of Transformer layers
        nhead: Number of attention heads
        mlp_ratio: Ratio of hidden to input dimension in MLP
        class_dropout_prob: Probability of dropping class labels for CFG
        num_datasets: Number of datasets (for multi-dataset training, default 2)
        num_spacegroups: Number of spacegroups (for crystals, default 230)
        d_context: Width of a per-molecule property vector (0 = unconditional).
            When > 0 a `context_embedder` MLP maps it to d_model and the result
            is ADDED to the conditioning vector c. 0 creates no module, so the
            state_dict of every existing checkpoint is unchanged.
    """

    def __init__(
        self,
        d_x: int = 8,
        d_model: int = 384,
        num_layers: int = 12,
        nhead: int = 6,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_datasets: int = 2,
        num_spacegroups: int = 230,
        d_context: int = 0,
    ):
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.nhead = nhead

        # Input embedding (with self-conditioning: 2 * d_x)
        self.x_embedder = nn.Linear(2 * d_x, d_model, bias=True)
        self.t_embedder = TimestepEmbedder(d_model)
        
        # Label embedders (silenced for molecule-only: use default indices)
        self.dataset_embedder = LabelEmbedder(num_datasets, d_model, class_dropout_prob)
        self.spacegroup_embedder = LabelEmbedder(num_spacegroups, d_model, class_dropout_prob)

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(d_model, d_x)
        self.d_context = 0
        if d_context > 0:
            self.enable_context(d_context)
        self.initialize_weights()

    def enable_context(self, d_context: int):
        """Add the property-context MLP (idempotent). Also called by LDMTaskFactory, since
        the denoiser may already be instantiated by Hydra before condition_names is known."""
        if self.d_context == d_context:
            return
        if self.d_context:
            raise ValueError(f"DiT already has d_context={self.d_context}, asked for {d_context}")
        self.d_context = d_context
        self.context_embedder = nn.Sequential(
            nn.Linear(d_context, self.d_model), nn.SiLU(), nn.Linear(self.d_model, self.d_model)
        )
        # Fresh module (also when added post-init): same init as initialize_weights().
        for m in self.context_embedder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self._zero_context_out()

    def _zero_context_out(self):
        # Zero the context MLP's output layer so a conditional DiT warm-started
        # from an unconditional checkpoint starts as exactly that model. It still
        # receives gradient (its input is non-zero), so it does not stay zero.
        nn.init.constant_(self.context_embedder[-1].weight, 0)
        nn.init.constant_(self.context_embedder[-1].bias, 0)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding tables
        nn.init.normal_(self.dataset_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.spacegroup_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.d_context > 0:  # initialize_weights() re-ran xavier over the MLP
            self._zero_context_out()

    def forward(
        self, x, t, mask, dataset_idx=None, spacegroup=None, x_sc=None, context=None
    ):
        """Forward pass of DiT.

        Args:
            x: Noisy latent tokens (B, N, d_x)
            t: Timestep for each sample (B,) or (B, 1)
            mask: Valid token mask, True if valid (B, N)
            dataset_idx: Dataset index (B,) - default 1 for molecules
            spacegroup: Spacegroup index (B,) - default 0 for molecules
            x_sc: Self-conditioning input (B, N, d_x) - optional
            context: Per-molecule property vector (B, d_context) - required iff
                the model was built with d_context > 0 (null = the caller's
                mask_value vector; the caller owns dropout/null semantics)

        Returns:
            Predicted clean latent (B, N, d_x)
        """
        B, N, _ = x.shape
        device = x.device

        # Default conditioning for molecules
        if dataset_idx is None:
            dataset_idx = torch.ones(B, dtype=torch.long, device=device)
        if spacegroup is None:
            spacegroup = torch.zeros(B, dtype=torch.long, device=device)

        # Positional embedding
        token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        pos_emb = get_pos_embedding(token_index, self.d_model)

        # Self-conditioning and input embeddings
        if x_sc is None:
            x_sc = torch.zeros_like(x)
        x = self.x_embedder(torch.cat([x, x_sc], dim=-1)) + pos_emb

        # Conditioning embeddings
        if t.dim() == 2:
            t = t.squeeze(1)
        t_emb = self.t_embedder(t)  # (B, d)
        d_emb = self.dataset_embedder(dataset_idx, self.training)  # (B, d)
        s_emb = self.spacegroup_embedder(spacegroup, self.training)  # (B, d)
        c = t_emb + d_emb + s_emb  # (B, d)
        if getattr(self, "d_context", 0) > 0:
            if context is None:
                raise ValueError(
                    f"DiT was built with d_context={self.d_context}; forward() needs `context` "
                    "(pass the null/mask_value vector for unconditional use)."
                )
            c = c + self.context_embedder(context.to(c.dtype))
        elif context is not None:
            raise ValueError("`context` given but DiT was built with d_context=0.")

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, ~mask)  # (B, N, d)

        # Prediction layer
        x = self.final_layer(x, c)  # (B, N, d_x)
        x = x * mask[..., None]
        return x

    def forward_with_cfg(self, x, t, mask, cfg_scale, dataset_idx=None, spacegroup=None, x_sc=None):
        """Forward with classifier-free guidance."""
        half_x = x[: len(x) // 2]
        combined_x = torch.cat([half_x, half_x], dim=0)
        model_out = self.forward(combined_x, t, mask, dataset_idx, spacegroup, x_sc)

        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps
