"""Geometry-aware (Point-Transformer-style) attention stack for TABASCO.

Selected with ``TransformerModule(implementation="point_transformer")``. Same
skeleton as ``Transformer`` in ``transformer.py`` (pre-LN residual blocks,
``Transition`` FFN, final LayerNorm) with the self-attention swapped for one
that sees relative geometry, so ``implementation="reimplemented"`` is the
matched control that isolates the attention change.

Relative geometry ``x_i - x_j`` is embedded ONCE per forward
(``PairGeometry``) and shared by every layer: the backbone never moves
coordinates internally, so recomputing it per layer would be wasted work.
Each layer reads it through its own small linear heads.

Two attention modes (``pt_attention``):

``bias``    scalar multi-head attention plus a per-head additive bias from the
            pair embedding (Graphormer-3D / AF3 pair-bias style). The bias
            projection is zero-initialised, so training starts from plain MHA.
``vector``  PTv2 grouped vector attention (Wu et al., 2022; Pointcept
            ``GroupedVectorAttention``): ``relation = (k_j - q_i) * pe_mul +
            pe_rel``, a weight-encoding MLP to one weight per channel group,
            softmax over neighbours, grouped aggregation of ``v_j + pe_val``.
            Deviations from the reference: dense masked attention over all
            atoms instead of kNN; LayerNorm instead of BatchNorm (padded atoms
            would corrupt batch statistics); one linear head per layer from the
            shared pair embedding instead of a per-layer MLP on raw xyz; the
            relation is ``rel_dim`` wide (default 64, the reference uses the
            full channel width: ``rel_dim=dim``) and the value bias is a
            separate head applied through the linearity trick below. Both
            exist because full-width pair tensors made this arm ~4x slower
            than the plain transformer (memory-bound, B x N x N x C).

``use_direction=False`` drops the unit vector from the pair features, which
makes the attention pattern rotation-invariant (distances only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from MolecularDiffusion.modules.layers.tabasco.transition import Transition


def _rbf(d: Tensor, d_max: float, n: int) -> Tensor:
    # Same Gaussian expansion as modules/layers/gvp/gvp.py::_rbf, copied
    # because importing that module pulls in dgl.
    mu = torch.linspace(0.0, d_max, n, device=d.device, dtype=d.dtype)
    return torch.exp(-(((d.unsqueeze(-1) - mu) * n / d_max) ** 2))


class PairGeometry(nn.Module):
    """``(B, N, 3)`` coordinates -> ``(B, N, N, pair_dim)`` pair embedding."""

    def __init__(
        self,
        pair_dim: int = 64,
        rbf_dim: int = 32,
        rbf_dmax: float = 12.0,
        use_direction: bool = True,
    ):
        super().__init__()
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.use_direction = use_direction
        self.mlp = nn.Sequential(
            nn.Linear(rbf_dim + 3 * use_direction, pair_dim),
            nn.SiLU(),
            nn.Linear(pair_dim, pair_dim),
        )

    def forward(self, coords: Tensor) -> Tensor:
        delta = coords.unsqueeze(2) - coords.unsqueeze(1)  # [b, i, j] = x_i - x_j
        # clamped squared norm: finite value and gradient for i == j / overlaps
        dist = delta.square().sum(-1).clamp_min(1e-8).sqrt()
        feats = _rbf(dist, self.rbf_dmax, self.rbf_dim)
        if self.use_direction:
            feats = torch.cat([feats, delta / dist.unsqueeze(-1)], dim=-1)
        return self.mlp(feats)


class GeoAttention(nn.Module):
    """Self-attention with access to the pair embedding ``e``.

    ``forward(x, e, key_padding_mask)`` with ``x`` ``(B, N, C)``, ``e``
    ``(B, N, N, pair_dim)`` (``e[b, i, j]`` embeds ``x_i - x_j``) and a bool
    ``key_padding_mask`` ``(B, N)`` where True marks padded atoms.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        pair_dim: int,
        mode: str = "vector",
        rel_dim: int = 64,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        if mode not in ("bias", "vector"):
            raise ValueError(f"Invalid pt_attention: {mode}")
        self.dim, self.num_heads, self.mode = dim, num_heads, mode

        if mode == "bias":
            self.qkv = nn.Linear(dim, 3 * dim)
            self.pair_bias = nn.Linear(pair_dim, num_heads)
            nn.init.zeros_(self.pair_bias.weight)
            nn.init.zeros_(self.pair_bias.bias)
        else:
            groups = num_heads
            self.q = nn.Sequential(nn.Linear(dim, rel_dim), nn.LayerNorm(rel_dim), nn.ReLU())
            self.k = nn.Sequential(nn.Linear(dim, rel_dim), nn.LayerNorm(rel_dim), nn.ReLU())
            self.v = nn.Linear(dim, dim)
            self.pe_mul = nn.Linear(pair_dim, rel_dim)
            self.pe_rel = nn.Linear(pair_dim, rel_dim)
            self.pe_val = nn.Linear(pair_dim, dim)
            self.weight_enc = nn.Sequential(
                nn.Linear(rel_dim, groups),
                nn.LayerNorm(groups),
                nn.ReLU(),
                nn.Linear(groups, groups),
            )
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, e: Tensor, key_padding_mask: Tensor) -> Tensor:
        b, n, c = x.shape
        h = self.num_heads
        # Finite fill (not -inf): a fully padded molecule gets uniform attention
        # instead of NaN; its output is masked away by the caller. The fill must
        # fit the dtype of the tensor it fills, which under autocast is not x.dtype.
        if self.mode == "bias":
            q, k, v = self.qkv(x).view(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4)
            bias = self.pair_bias(e).permute(0, 3, 1, 2)  # (B, H, N, N)
            mask = bias.masked_fill(
                key_padding_mask[:, None, None, :], torch.finfo(bias.dtype).min
            )
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return self.out(o.transpose(1, 2).reshape(b, n, c))

        q, k, v = self.q(x), self.k(x), self.v(x)
        rel = k.unsqueeze(1) - q.unsqueeze(2)  # (B, N, N, rel_dim): k_j - q_i
        rel = rel * self.pe_mul(e) + self.pe_rel(e)
        w = self.weight_enc(rel)
        w = w.masked_fill(key_padding_mask[:, None, :, None], torch.finfo(w.dtype).min)
        a = w.softmax(dim=2).to(e.dtype)  # over neighbours j, one weight per group
        dc = c // h
        o = torch.einsum("bijg,bjgc->bigc", a, v.view(b, n, h, dc))
        # sum_j a_ij * Linear(e_ij) == Linear(sum_j a_ij * e_ij) (weights sum to 1
        # over j), so the value bias never needs a per-pair C-wide tensor.
        ebar = torch.einsum("bijg,bijp->bigp", a, e)
        wv = self.pe_val.weight.view(h, dc, -1).to(ebar.dtype)
        o = o + torch.einsum("gcp,bigp->bigc", wv, ebar) + self.pe_val.bias.view(h, dc)
        return self.out(o.reshape(b, n, c))


class PointTransformerBlock(nn.Module):
    """``TransformerBlock`` with ``GeoAttention`` in place of ``AttentionBlock``."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        pair_dim: int,
        pt_attention: str,
        rel_dim: int = 64,
        mlp_dim: int | None = None,
        activation_type: str = "gelu",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = GeoAttention(dim, num_heads, pair_dim, pt_attention, rel_dim)
        self.ff_block = Transition(
            dim=dim,
            hidden_dim=mlp_dim,
            activation_type=activation_type,
            layer_norm=True,
        )

    def forward(self, x: Tensor, e: Tensor, padding_mask: Tensor) -> Tensor:
        x = x + self.attn(self.norm(x), e, padding_mask)
        return x + self.ff_block(x)


class PointTransformer(nn.Module):
    """Drop-in for ``Transformer`` that additionally takes ``coords``."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        pt_attention: str = "vector",
        pair_dim: int = 64,
        rbf_dim: int = 32,
        rbf_dmax: float = 12.0,
        use_direction: bool = True,
        rel_dim: int = 64,
        mlp_dim: int | None = None,
        activation_type: str = "gelu",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.geometry = PairGeometry(pair_dim, rbf_dim, rbf_dmax, use_direction)
        self.layers = nn.ModuleList(
            PointTransformerBlock(
                dim,
                num_heads,
                pair_dim,
                pt_attention,
                rel_dim=rel_dim,
                mlp_dim=mlp_dim,
                activation_type=activation_type,
                norm_eps=norm_eps,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, x: Tensor, padding_mask: Tensor, coords: Tensor) -> Tensor:
        """``x`` (B, N, dim), ``padding_mask`` (B, N) True = padded, ``coords`` (B, N, 3)."""
        padding_mask = padding_mask.bool()
        e = self.geometry(coords)
        for layer in self.layers:
            x = layer(x, e, padding_mask)
        return self.norm(x)
