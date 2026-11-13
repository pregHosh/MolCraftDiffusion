"""Calling conventions of the three network roles the platform's models fall into.

These are documentation and type-checking aids only. Nothing inherits from them,
they are not ``runtime_checkable``, and no call site depends on them, so existing
models, checkpoints and configs are unaffected. ``tests/test_model_protocols.py``
checks real implementers against them with ``inspect``.

Mask polarity differs by role and each wrapper converts once at its boundary:
``node_mask`` is 1 for a real node (EDM family); ``padding_mask`` is 1 for a
padded slot (TABASCO-style backbones).

Not covered on purpose: ported upstream shapes that the tasks call directly
(``ligandiff``, ``diffsbdd``, ``diffpharma``, ``oareactdiff`` put ``t`` elsewhere,
``difflinker.Dynamics`` adds ``linker_mask``).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple

import torch


class DenseDynamics(Protocol):
    """EDM-style denoising network on padded dense batches.

    Used by ``edm.en_diffusion.EnVariationalDiffusion``. Implemented by
    ``EGNN_dynamics``, ``EGT_dynamics``, ``GCDMDynamics``, ``PaiNNDynamics``,
    ``TransformerDynamics``, ``GFMDiffDynamics``, ``EquiformerV2_dynamics`` and
    ``geoldm``'s ``EGNN_dynamics_QM9``. Most expose ``forward`` as a thin wrapper
    over ``_forward``.

    ``xh`` is ``(B, N, n_dims + features)`` (coordinates then features),
    ``node_mask`` is ``(B, N, 1)`` with 1 for real nodes, ``edge_mask`` is
    ``(B * N * N, 1)``, ``t`` is the diffusion time. Returns the prediction for
    ``xh`` with the same shape.
    """

    def __call__(
        self,
        t: torch.Tensor,
        xh: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...


class PygDynamics(Protocol):
    """Denoising network on a PyG batch.

    Used by ``edm.en_diffusion_pyg.EnVariationalDiffusionPyG``. Implemented by
    ``EGT_dynamics_PyG``, ``EGNN_dynamics_PyGAdapter`` and ``eSEN_dynamics``;
    each implements ``_forward(mol_graph[, return_hidden])`` behind a generic
    ``forward``. ``mol_graph`` carries the batch graph (and the noisy node
    features) and the result is the per-node prediction.
    """

    def __call__(self, mol_graph: Any, *args: Any, **kwargs: Any) -> torch.Tensor: ...


class PaddedBackbone(Protocol):
    """TABASCO-style backbone: padded tensors, ``padding_mask`` is 1 for padding.

    Implemented by the ``tabasco_egnn``, ``tabasco_equiformer`` and ``tabasco_gvp``
    backbones. ``coords`` is ``(B, N, 3)``, ``atomics`` is ``(B, N, atom_dim)``,
    ``t`` is ``(B,)`` in [0, 1]. Returns ``(coords, atom_logits)``.
    """

    def __call__(
        self,
        coords: torch.Tensor,
        atomics: torch.Tensor,
        padding_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
