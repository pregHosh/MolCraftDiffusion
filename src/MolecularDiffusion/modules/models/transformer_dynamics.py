"""Dense self-attention adapter for ``EnVariationalDiffusion``.

Binds a plain (non-equivariant) Transformer to the
``dynamics._forward(t, xh, node_mask, edge_mask, context)`` contract that
``EnVariationalDiffusion.phi`` (``modules/models/en_diffusion.py:234``)
calls.

Novel-model ablation (``docs/model_novel/diffusion_transformer/
INTEGRATION_PLAN.md``): swaps EGCL's equivariant pairwise-difference
message passing (``modules/models/egcl.py::EGNN_dynamics``) for plain
multi-head self-attention over absolute coordinates, with no distance/edge
features and no built-in rotation-equivariance -- the hypothesis is that
the platform's existing rotation-augmentation flag
(``GeomMolecularGenerative.data_augmentation``) substitutes for the
architectural guarantee.

Revision 2026-09-09: a real cluster run (300k steps, hyperparameters matched
to the EGCL baseline) still landed far below EGCL's `valid_posebuster`. Two
concrete asymmetries against both EGCL and this platform's other
non-equivariant 3D transformer (ADiT's `DiT`,
``modules/models/ldm/denoisers/dit.py``) were identified and fixed here:

1. No near-zero output-head init -- EGCL's coordinate-update head is
   Xavier-init'd with ``gain=0.001`` (``modules/layers/conv.py``); DiT
   zero-inits its final projection and AdaLN gates. This file's output head
   previously had neither.
2. Time was injected once, additively, at the input -- every one of the 9
   attention blocks was otherwise unconditioned, unlike DiT's per-block
   AdaLN-zero conditioning.

Both are fixed by reusing DiT's own blocks (``DiTBlock``, ``FinalLayer``)
by import rather than re-deriving the same recipe -- this platform's other
non-equivariant 3D transformer already solved this exact problem, on the
same platform, so there's no reason to reinvent it. ``Transformer``/
``AttentionBlock`` (``modules/layers/tabasco/``) are no longer used here as
a result; ``TimeFourierEncoding`` still is, feeding the same time embedding
into every block's AdaLN conditioning instead of only the input sum.
"""

from __future__ import annotations

import torch
from torch import nn

from MolecularDiffusion.modules.layers.tabasco.positional_encoder import (
    SinusoidEncoding,
    TimeFourierEncoding,
)
from MolecularDiffusion.modules.models.ldm.denoisers.dit import (
    DiTBlock,
    FinalLayer,
)
from MolecularDiffusion.utils.geom_utils import remove_mean_with_mask_v2


class TransformerDynamics(nn.Module):
    """Denoising network: plain multi-head self-attention, EDM interface.

    Args:
        in_node_nf: Node feature channels the diffusion model expects
            back (atom-type one-hot + atomic number [+ extra values]),
            excluding time and context, which are added internally.
        context_node_nf: Conditioning channels, concatenated to the node
            features before embedding.
        n_dims: Spatial dimensions (3).
        hidden_dim: Transformer token width.
        num_layers: Number of ``DiTBlock``s.
        num_heads: Multi-head self-attention heads.
        mlp_dim: Feed-forward hidden width; ``None`` defaults to
            ``4 * hidden_dim`` (``DiTBlock``'s own default ``mlp_ratio``).
        dropout: Currently a no-op -- ``DiTBlock``'s attention/MLP
            hardcode zero dropout internally and silently ignore extra
            kwargs. Kept as a constructor arg (this platform's other
            dynamics wrappers all expose one) rather than removed, since
            the config default is already 0.0; flagged here so a nonzero
            value doesn't look like it did something.
        activation_type: Currently a no-op for the same reason --
            ``DiTBlock``'s MLP hardcodes ``GELU(approximate="tanh")``.
            Kept for config-shape compatibility with the pre-revision
            file; the bundled config's default was already ``"gelu"``.
        add_sinusoid_posenc: Ablation-only knob, off by default. Atoms
            are an unordered set here, so this has no principled reason
            to help; it exists only for later ablation curiosity and is
            never load-bearing.
    """

    def __init__(
        self,
        in_node_nf: int,
        context_node_nf: int = 0,
        n_dims: int = 3,
        hidden_dim: int = 192,
        num_layers: int = 9,
        num_heads: int = 8,
        mlp_dim: int | None = None,
        dropout: float = 0.0,  # noqa: ARG002 -- see class docstring
        activation_type: str = "gelu",  # noqa: ARG002 -- see class docstring
        add_sinusoid_posenc: bool = False,
    ) -> None:
        super().__init__()
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.n_dims = n_dims
        self.add_sinusoid_posenc = add_sinusoid_posenc

        self.pos_embed = nn.Linear(n_dims, hidden_dim, bias=False)
        self.feat_embed = nn.Linear(
            in_node_nf + context_node_nf, hidden_dim, bias=False
        )
        self.time_encoding = TimeFourierEncoding(hidden_dim)

        mlp_ratio = (mlp_dim / hidden_dim) if mlp_dim is not None else 4.0
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(num_layers)
            ]
        )
        self.final_layer = FinalLayer(hidden_dim, n_dims + in_node_nf)
        if add_sinusoid_posenc:
            self.sinusoid_posenc = SinusoidEncoding(hidden_dim)

        self._zero_init_adaln()

    def _zero_init_adaln(self) -> None:
        """Zero-init every AdaLN gate/scale/shift layer + final proj.

        Exactly as DiT's own ``initialize_weights`` does
        (``modules/models/ldm/denoisers/dit.py``) -- each block starts as
        a near-identity function and the final layer starts predicting
        all-zero eps, rather than default-Kaiming-uniform noise, so early
        training updates start small instead of large and untrained.
        """
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # -- EnVariationalDiffusion dynamics interface --------------------- #

    def _forward(
        self,
        t: torch.Tensor,
        xh: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,  # noqa: ARG002 -- see class docstring
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict eps for a dense padded batch.

        Args:
            t: scalar or ``(B,)``/``(B, 1)`` diffusion time in ``[0, 1]``.
            xh: ``(B, N, 3 + in_node_nf)`` noisy positions ++ features.
            node_mask: ``(B, N, 1)``, 1 = valid node, 0 = padding.
            edge_mask: unused. Every dynamics wrapper under
                ``EnVariationalDiffusion`` shares this call signature
                (``en_diffusion.py:234``); the dense ``edge_mask`` here
                is always ``node_mask`` outer-producted with itself, no
                extra information the attention ``key_padding_mask``
                built from ``node_mask`` alone doesn't already carry.
            context: ``(B, N, context_node_nf)`` or ``None``.

        Returns:
            ``(B, N, 3 + in_node_nf)``, zero on padded rows, with the
            position channels projected to the zero-CoM subspace that
            ``EnVariationalDiffusion`` asserts everywhere.
        """
        b, n, _ = xh.shape
        x = xh[..., : self.n_dims]
        h = xh[..., self.n_dims :]
        if context is not None and self.context_node_nf > 0:
            h = torch.cat([h, context], dim=-1)

        # TimeFourierEncoding asserts its own output shape (B, dim) and
        # requires a strict 1-D (B,) input -- unlike PaiNN's own
        # FourierTimeFeatures, which wants (N, 1). This is also the AdaLN
        # conditioning vector `c` every DiTBlock/FinalLayer takes below --
        # unlike the pre-revision file, time is no longer also summed
        # additively into the input tokens, matching DiT's own recipe.
        if torch.numel(t) == 1:
            t_flat = t.reshape(1).expand(b)
        else:
            t_flat = t.reshape(b)
        time_emb = self.time_encoding(t_flat)

        tokens = self.pos_embed(x) + self.feat_embed(h)
        if self.add_sinusoid_posenc:
            tokens = tokens + self.sinusoid_posenc(b, n)

        # This platform's node_mask is 1 = valid, but DiTBlock's `mask`
        # arg (like Transformer's padding_mask) is a key_padding_mask,
        # True = ignore -- inverted here.
        key_padding_mask = ~node_mask.squeeze(-1).bool()
        for block in self.blocks:
            tokens = block(tokens, time_emb, key_padding_mask)
        raw_out = self.final_layer(tokens, time_emb)

        # Zero padded rows BEFORE the CoM projection -- load-bearing, not
        # stylistic. remove_mean_with_mask_v2's mean is
        # sum(pos, dim=1) / num_valid_nodes; unzeroed padding would bias
        # every molecule's centroid without raising, and the corruption
        # would only surface later as assert_mean_zero_with_mask failing
        # deep inside EnVariationalDiffusion.
        out = raw_out * node_mask
        vel = (
            remove_mean_with_mask_v2(out[..., : self.n_dims], node_mask)
            * node_mask
        )

        if torch.any(torch.isnan(vel)):
            vel = torch.zeros_like(vel)
            out = torch.zeros_like(out)

        return torch.cat([vel, out[..., self.n_dims :]], dim=2)

    # Deliberately no `_forward_pyG`: its only caller
    # (`en_diffusion.py::phi_pyg`) is reached only for batches carrying a
    # "graph" key, which this model's `data_type: pointcloud` batches
    # never set (see INTEGRATION_PLAN.md's scope decision).
