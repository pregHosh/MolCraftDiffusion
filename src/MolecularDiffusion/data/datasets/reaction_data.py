"""A shared, model-neutral container for a chemical reaction.

Transition-state generators all take the same thing -- *a reactant side, a
product side, and a transition state to predict* -- and disagree only about
which channel each side is carried in. This module is that shape, and
nothing more. It imports the platform's canonical bond vocabulary and
nothing from ``modules/``, so a model-specific data layer can sit beside it
(``oareactdiff_data.py``) without either one owning the other.

Three real reaction models were read at source to size this contract; every
field below is demanded by at least one of them.

===========================  ============  ============  ==============
                             OA-ReactDiff  GoFlow        RitS
===========================  ============  ============  ==============
one shared atom ordering     yes           yes           yes
TS target                    3D coords     3D coords     3D coords
reactant/product side is     geometry      connectivity  connectivity
their 3D endpoints are used  **yes**       no            no
bonds                        none          22-class      9-class
stereo                       none          vestigial     bond classes
===========================  ============  ============  ==============

Only OA-ReactDiff is implemented in this pass. The other two are recorded
because they are what makes the shape a contract rather than one model's
struct with a general-sounding name.

Two rules that are not fields
-----------------------------

**``bond_index`` is DIRECTED, and per side.** Deliberately *not* the
upper-triangular storage rule of ``graph3d_dataset.py``. RitS encodes
tetrahedral chirality as an antisymmetric directed 3-cycle ``a -> c -> b ->
a`` whose orientation is the only thing distinguishing R from S; symmetrising
or canonicalising to ``i < j`` would silently racemize a whole corpus.
Per-side rather than one shared union index, because a side has to be
self-describing -- the union edge set both siblings actually feed their
networks is ``(bmat_r + bmat_p).nonzero()``, derived in *their* collate.

**``pos`` is real geometry or it is ``None``.** Never a zero placeholder,
never a copy of the TS standing in for a missing endpoint. Both patterns
exist upstream in RitS, and a container that stored them would make
:attr:`ReactionSide.has_geometry` lie. The rule costs nothing and removes the
need for a per-slot validity flag.

Two PyG gotchas the layout dodges, both found live in GoFlow: PyG
node-offsets **any** attribute whose name contains ``"index"``, so scalar
metadata lives in :attr:`Reaction.meta` and is never named ``*_index``; and
PyG deletes attributes assigned ``None``, so optional fields need presence
checks rather than null checks once they reach a ``Data`` object.

What is deliberately absent
---------------------------

*No representation enum.* The populated fields **are** the representation;
:attr:`ReactionSide.has_geometry` / :attr:`ReactionSide.has_connectivity`
derive it in one line each. A stored enum is a second source of truth that
can desync from the tensors it describes.

*No ``fc``.* None of the three models carries a per-atom signed formal
charge: OA-ReactDiff has no formal charge at all, GoFlow folds
``GetFormalCharge`` into its ``feat`` one-hot, and RitS's ``charges`` is a
per-molecule net charge broadcast to every atom. :attr:`ReactionSide.feat`
and :attr:`Reaction.meta` cover both.

*No stereo channel -- yet.* RitS packs E/Z and chirality into bond classes
5-8, which are not bonds and do not belong in a channel every other model
reads as bond order. The honest fix, purely additive and deferred until a
model actually populates it, is one more optional pair on
:class:`ReactionSide`: ``stereo_index (2, S)``, directed, and
``stereo_type (S,)`` over a separate ``{E, Z, CHI_STAR, CHI_CYCLE}``
vocabulary. Recorded here so that day is an addition, not a redesign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

# `component.dataset -> core -> runmodes.train.data -> data.dataset` is a
# pre-existing circular import in this tree: importing anything under
# `data/component/` first leaves `component.dataset` half-built. Touching
# `data.dataset` first breaks the cycle. Same one-line workaround, and the
# same reason, as `tests/test_graph3d_dataset.py:15-17`.
import MolecularDiffusion.data.dataset  # noqa: F401
from MolecularDiffusion.data.datasets.graph3d_dataset import (
    BOND_VOCAB,
    N_BOND_CLASSES,
)

__all__ = ["BOND_VOCAB", "N_BOND_CLASSES", "Reaction", "ReactionSide"]


@dataclass
class ReactionSide:
    """One side of a reaction -- reactant or product.

    Every field is optional because which ones are populated *is* the
    representation. A side carrying only ``pos`` is a 3D endpoint
    (OA-ReactDiff); a side carrying only bonds and ``feat`` is a
    connectivity-only condition (GoFlow, RitS).

    Attributes:
        pos: ``(n, 3)`` real 3D coordinates, or ``None``. Never a
            placeholder -- see the module docstring.
        bond_index: ``(2, E)`` **directed** edges, this side's own bonds
            only, or ``None``. Not upper-triangular.
        bond_type: ``(E,)`` classes over the canonical five-entry
            :data:`BOND_VOCAB` (``0=none, 1=SINGLE, 2=DOUBLE, 3=TRIPLE,
            4=AROMATIC``), aligned with ``bond_index``, or ``None``.
        feat: ``(n, F)`` per-atom descriptors, or ``None``. The vocabulary
            behind ``F`` is dataset-derived and travels with the model's own
            data module, not with this container.
    """

    pos: Optional[torch.Tensor] = None
    bond_index: Optional[torch.Tensor] = None
    bond_type: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None

    @property
    def has_geometry(self) -> bool:
        """Whether this side carries real 3D coordinates."""
        return self.pos is not None

    @property
    def has_connectivity(self) -> bool:
        """Whether this side carries bonds."""
        return self.bond_index is not None

    def validate(self, n_atoms: int) -> None:
        """Check every populated field against the reaction's atom count.

        Args:
            n_atoms: length of the owning :class:`Reaction`'s ``z``.

        Raises:
            ValueError: on any shape disagreement. These are all silent
                corruption if they get through -- a mis-sized ``pos``
                broadcasts, and a ``bond_index`` out of range indexes the
                wrong atoms rather than erroring.
        """
        if self.pos is not None and self.pos.shape != (n_atoms, 3):
            raise ValueError(
                f"pos has shape {tuple(self.pos.shape)}, expected "
                f"({n_atoms}, 3) to match the reaction's z."
            )
        if self.feat is not None and self.feat.shape[0] != n_atoms:
            raise ValueError(
                f"feat has {self.feat.shape[0]} rows, expected {n_atoms}."
            )
        if (self.bond_index is None) != (self.bond_type is None):
            raise ValueError(
                "bond_index and bond_type must both be present or both be "
                "None; one without the other is an unlabelled edge set."
            )
        if self.bond_index is None or self.bond_type is None:
            return
        if self.bond_index.dim() != 2 or self.bond_index.shape[0] != 2:
            raise ValueError(
                f"bond_index has shape {tuple(self.bond_index.shape)}, "
                "expected (2, E)."
            )
        if self.bond_type.shape != (self.bond_index.shape[1],):
            raise ValueError(
                f"bond_type has {tuple(self.bond_type.shape)} entries but "
                f"bond_index has {self.bond_index.shape[1]} edges."
            )
        if self.bond_index.numel() and int(self.bond_index.max()) >= n_atoms:
            raise ValueError(
                f"bond_index references atom "
                f"{int(self.bond_index.max())} but the reaction has only "
                f"{n_atoms} atoms."
            )
        if self.bond_type.numel() and (
            int(self.bond_type.min()) < 0
            or int(self.bond_type.max()) >= N_BOND_CLASSES
        ):
            raise ValueError(
                f"bond_type values must be in [0, {N_BOND_CLASSES}); got "
                f"[{int(self.bond_type.min())}, "
                f"{int(self.bond_type.max())}]. Remap the source "
                "vocabulary onto BOND_VOCAB rather than widening it."
            )


@dataclass
class Reaction:
    """One reaction: a shared atom set, two sides, and a TS target.

    ``z`` is hoisted here rather than duplicated per side because all three
    surveyed models guarantee a single shared atom ordering across reactant,
    transition state and product. Making it one tensor makes that guarantee
    structural instead of an assertion each model rewrites differently.

    ``ts_pos`` is a bare tensor rather than a third :class:`ReactionSide`:
    no surveyed model predicts TS *bonds*, so the symmetry would be
    decoration. Promote it the day one does.

    Attributes:
        z: ``(n,)`` atomic numbers. **This is the shared ordering** -- every
            other per-atom tensor in the reaction is aligned to it.
        reactant: the reactant side.
        product: the product side.
        ts_pos: ``(n, 3)`` transition-state coordinates -- the prediction
            target. ``None`` at inference time, when there is nothing to
            predict against.
        meta: free-form per-reaction scalars and strings (reaction id,
            SMILES, energies). Never batched, and never named ``*_index``
            (PyG node-offsets any such key).
    """

    z: torch.Tensor
    reactant: ReactionSide
    product: ReactionSide
    ts_pos: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate both sides and the TS target against ``z``."""
        if self.z.dim() != 1:
            raise ValueError(
                f"z has shape {tuple(self.z.shape)}, expected (n,)."
            )
        n_atoms = int(self.z.shape[0])
        self.reactant.validate(n_atoms)
        self.product.validate(n_atoms)
        if self.ts_pos is not None and self.ts_pos.shape != (n_atoms, 3):
            raise ValueError(
                f"ts_pos has shape {tuple(self.ts_pos.shape)}, expected "
                f"({n_atoms}, 3) to match z."
            )

    def __len__(self) -> int:
        """Number of atoms."""
        return int(self.z.shape[0])
