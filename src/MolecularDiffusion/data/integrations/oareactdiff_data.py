"""Dataset / collate / DataModule for OA-ReactDiff on Transition1x.

Ported from ``oa_reactdiff/dataset/base_dataset.py`` and
``oa_reactdiff/dataset/transition1x.py`` (commit 543aaa8, MIT), restructured
so the *item* is the shared :class:`~MolecularDiffusion.data.component.
reaction_data.Reaction` container and the *batch* is upstream's own
``(representations, conditions)`` tuple.

Why that split. Upstream's dataset materialises a flat dict of
index-suffixed keys -- ``pos_0``, ``one_hot_1``, ``charge_2`` -- where the
suffix silently encodes "reactant / transition state / product". That is
unreadable and unshareable: nothing in the layout says which index is the
target. :class:`Reaction` names the three roles; the positional layout is
rebuilt in :func:`oareactdiff_collate`, immediately before the tensors reach
``EnVariationalDiffusion.forward``, which is the only code that wants it.
Nothing under ``data/`` was modified to make this work -- the module is
selected purely by ``_target_`` from
``configs/data/oareactdiff_ts1x_dataset.yaml``, exactly as
``kgdiff_data.py`` and ``pmdm_data.py`` are.

The split is the published one, and it is not two files
----------------------------------------------------------

``train_addprop.pkl`` and ``valid_addprop.pkl`` contain the **same 10,073
reactions**. They differ only in their ``use_ind`` list -- 9,000 entries in
one, 1,073 in the other. So ``use_by_ind: true`` is what actually splits the
data; pointing both at the same file would train and validate on the same
reactions. Intersected with ``single_fragment``, the effective sizes under
the released settings are 6,733 train / 783 valid reactions (doubled by
``swapping_react_prod``).

The five-wide atom one-hot is derived, not stored
-------------------------------------------------

Upstream builds it in the dataset via ``ATOM_MAPPING``; here it is built in
the collate from the reaction's ``z``. Keeping it derived means the shared
container never carries a model-specific vocabulary, and changing the
vocabulary never means reconverting anything. There is nothing to convert in
the first place: the data module reads the upstream pickles directly.

``charge`` is the atomic number
-------------------------------

Upstream's ``charges`` column holds Z, not a formal charge
(``base_dataset.py:170-176``), and the ninth column of the model's ``xh``
tensor is that same Z. There is no formal-charge channel anywhere in this
model, which is why :class:`Reaction` has no ``fc`` field.
"""

from __future__ import annotations

import logging
import os
import pickle
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from MolecularDiffusion.data.datasets.reaction_data import (
    Reaction,
    ReactionSide,
)

logger = logging.getLogger(__name__)

#: ``ATOM_MAPPING`` (``oa_reactdiff/dataset/base_dataset.py:9-15``): atomic
#: number -> one-hot column. The order is fixed by the released weights.
ATOM_MAPPING: Dict[int, int] = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

#: Column order of the one-hot, i.e. the ``atom_vocab`` ``save_xyz_file``
#: decodes against. F is in the vocabulary but absent from Transition1x.
OAREACTDIFF_ATOM_VOCAB: List[str] = ["H", "C", "N", "O", "F"]

N_ELEMENT = len(ATOM_MAPPING)

#: Object order inside every batch. Fixed by ``fragment_names`` in the
#: checkpoint (``["R", "TS", "P"]``), which is what binds each object to its
#: own encoder/decoder pair -- reordering these silently swaps weights.
FRAGMENT_ORDER = ("reactant", "transition_state", "product")

#: Index of the transition state in :data:`FRAGMENT_ORDER`. The generated
#: object; the other two are held fixed during inpainting.
TS_INDEX = 1


def _to_pos(raw: Any, n_atoms: int) -> torch.Tensor:
    """One object's coordinates as a ``(n_atoms, 3)`` float32 tensor.

    The pickles store padded arrays for some rows, so the atom count is
    taken from ``num_atoms`` and the rest is sliced off -- exactly what
    ``BaseDataset.process_molecules`` does.
    """
    return torch.tensor(
        np.asarray(raw, dtype=np.float32)[:n_atoms], dtype=torch.float32
    )


class OAReactDiffTS1xDataset(Dataset):
    """Transition1x reactions, one :class:`Reaction` per item.

    Args:
        pkl_path: an upstream ``*_addprop.pkl`` / ``train.pkl``.
        center: subtract each object's own centre of mass. On by default and
            effectively mandatory -- the diffusion process operates on the
            zero-CoM subspace per object, and ``inpaint`` re-centres anyway.
        zero_charge: replace the Z channel with zeros. The released
            checkpoint was trained with real Z, so leave this off unless you
            are training from scratch and know why.
        single_frag_only: keep only reactions whose reactant and product are
            each a single connected fragment (the released setting).
        swapping_react_prod: also emit every reaction backwards, with
            reactant and product exchanged and the same transition state.
            Doubles the set; the released checkpoint was trained with it on.
        use_by_ind: honour the pickle's published ``use_ind`` split. **This
            is what separates train from validation** -- see the module
            docstring.
        limit: cap the number of source reactions (before swapping). For
            smoke tests; ``None`` uses everything.
    """

    def __init__(
        self,
        pkl_path: str,
        center: bool = True,
        zero_charge: bool = False,
        single_frag_only: bool = True,
        swapping_react_prod: bool = True,
        use_by_ind: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Transition1x pickle not found: {pkl_path}. The preprocessed "
                "pickles ship inside the OA-ReactDiff repo at "
                "oa_reactdiff/data/transition1x/; the raw corpus is at "
                "https://gitlab.com/matschreiner/Transition1x."
            )
        with open(pkl_path, "rb") as handle:
            raw = pickle.load(handle)  # noqa: S301 - upstream's own artifact

        self.pkl_path = pkl_path
        self.center = center
        self.zero_charge = zero_charge
        self.swapping_react_prod = swapping_react_prod
        self.raw = raw

        n_total = len(raw["single_fragment"])
        if single_frag_only:
            keep = {
                i for i in range(n_total) if int(raw["single_fragment"][i]) == 1
            }
        else:
            keep = set(range(n_total))
        if use_by_ind:
            keep &= {int(i) for i in raw["use_ind"]}
        # sorted(), not set-iteration order: upstream relies on CPython's
        # small-int hashing to come out ascending, which is true but is not a
        # guarantee. Same order, spelled out.
        self._indices: List[int] = sorted(keep)
        if limit is not None:
            self._indices = self._indices[:limit]

        logger.info(
            "OA-ReactDiff TS1x: %d/%d reactions from %s "
            "(single_frag_only=%s, use_by_ind=%s)%s",
            len(self._indices),
            n_total,
            os.path.basename(pkl_path),
            single_frag_only,
            use_by_ind,
            " x2 (swapping_react_prod)" if swapping_react_prod else "",
        )

    def __len__(self) -> int:
        """Number of items, counting the swapped copies."""
        return len(self._indices) * (2 if self.swapping_react_prod else 1)

    def _object(self, name: str, row: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(pos, z)`` for one object of one source reaction."""
        obj = self.raw[name]
        n_atoms = int(obj["num_atoms"][row])
        pos = _to_pos(obj["positions"][row], n_atoms)
        if self.center:
            pos = pos - pos.mean(dim=0)
        z = torch.tensor(
            np.asarray(obj["charges"][row], dtype=np.int64)[:n_atoms],
            dtype=torch.int64,
        )
        return pos, z

    def __getitem__(self, idx: int) -> Reaction:
        """One reaction, with the reverse direction in the upper half."""
        n_src = len(self._indices)
        row = self._indices[idx % n_src]
        swapped = idx >= n_src
        # FRAG_MAPPING (transition1x.py:8-12): the swapped copy reads the
        # product into the reactant slot and vice versa, keeping the same TS.
        r_name, p_name = (
            ("product", "reactant") if swapped else ("reactant", "product")
        )

        r_pos, r_z = self._object(r_name, row)
        ts_pos, ts_z = self._object("transition_state", row)
        p_pos, p_z = self._object(p_name, row)
        if not (
            torch.equal(r_z, ts_z) and torch.equal(ts_z, p_z)
        ):  # pragma: no cover - holds on all 10,073 rows
            raise ValueError(
                f"row {row}: reactant/TS/product disagree on atom ordering. "
                "Reaction assumes one shared ordering; a mismatch here would "
                "silently pair the wrong atoms across the three objects."
            )

        return Reaction(
            z=r_z,
            reactant=ReactionSide(pos=r_pos),
            product=ReactionSide(pos=p_pos),
            ts_pos=ts_pos,
            meta={
                "rxn": self.raw["reactant"]["rxn"][row],
                "smi_reactant": self.raw[r_name]["smi"][row],
                "smi_product": self.raw[p_name]["smi"][row],
                "ediff": float(self.raw[r_name]["ediff"][row]),
                "swapped": swapped,
                "row": row,
            },
        )


def oareactdiff_collate(
    batch: Sequence[Reaction], zero_charge: bool = False
) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
    """Reactions -> upstream's ``(representations, conditions)`` tuple.

    Reproduces ``BaseDataset.collate_fn`` (``base_dataset.py:52-88``) but
    reads named fields instead of parsing index suffixes out of key strings.
    The output is exactly what ``EnVariationalDiffusion.forward`` takes, so
    **no adapter runs on the training path**.

    Each of the three ``representations`` entries carries:

    ``size``     ``(B,)``      atoms per sample in this object
    ``pos``      ``(sum n, 3)``  float32 coordinates
    ``one_hot``  ``(sum n, 5)``  int64, derived from ``z`` via ATOM_MAPPING
    ``charge``   ``(sum n, 1)``  int64 atomic number (or zeros)
    ``mask``     ``(sum n,)``    int64 scatter index -> which sample a row is

    Dtypes are upstream's: the one-hot and charge stay integral and are
    promoted to float by ``torch.cat`` when ``xh`` is assembled.

    ``conditions`` is ``(B, 1)`` of zeros. The released checkpoint has
    ``condition_nf=1`` and was fed constant zeros throughout
    (``transition1x.py:141-147``), so the channel exists in the weights but
    carries no information -- do not mistake it for a place to put a
    property.

    Args:
        batch: reactions to collate. Every one needs ``ts_pos``.
        zero_charge: zero the atomic-number channel.

    Returns:
        ``(representations, conditions)``.

    Raises:
        ValueError: if any reaction is missing its transition state. There is
            no meaningful placeholder -- the TS slot supplies the atom
            identities the ``pos_only`` sampler copies through.
    """
    missing = [i for i, rxn in enumerate(batch) if rxn.ts_pos is None]
    if missing:
        raise ValueError(
            f"reaction(s) {missing} have no ts_pos. Both the training loss "
            "and the inpainting sampler read the TS object's atom identities "
            "from it."
        )

    sizes = torch.tensor([len(rxn) for rxn in batch], dtype=torch.int64)
    mask = torch.repeat_interleave(
        torch.arange(len(batch), dtype=torch.int64), sizes
    )
    atom_idx = torch.cat(
        [
            torch.tensor(
                [ATOM_MAPPING[int(z)] for z in rxn.z.tolist()],
                dtype=torch.int64,
            )
            for rxn in batch
        ]
    )
    one_hot = F.one_hot(atom_idx, num_classes=N_ELEMENT)
    charge_z = torch.cat([rxn.z for rxn in batch]).view(-1, 1)
    charge = torch.zeros_like(charge_z) if zero_charge else charge_z

    positions = {
        "reactant": [rxn.reactant.pos for rxn in batch],
        "transition_state": [rxn.ts_pos for rxn in batch],
        "product": [rxn.product.pos for rxn in batch],
    }
    representations = [
        {
            "size": sizes,
            "pos": torch.cat(positions[name], dim=0),
            # Each object gets its own tensor object, never a shared one:
            # the normalizer mutates `representations[ii][...]` in place.
            "one_hot": one_hot.clone(),
            "charge": charge.clone(),
            "mask": mask.clone(),
        }
        for name in FRAGMENT_ORDER
    ]
    conditions = torch.zeros((len(batch), 1), dtype=torch.int64)
    return representations, conditions


class OAReactDiffDataModule:
    """DataModule contract: ``load()`` + ``train_set``/``valid_set``/``test_set``.

    Two dataset instances over **two pickles that hold the same reactions**;
    the ``use_ind`` list inside each is what makes them a split. See the
    module docstring. ``test_set`` reuses the validation pickle because
    upstream ships no ``test.pkl``.
    """

    def __init__(
        self,
        train_pkl: str,
        valid_pkl: Optional[str] = None,
        center: bool = True,
        zero_charge: bool = False,
        single_frag_only: bool = True,
        swapping_react_prod: bool = True,
        use_by_ind: bool = True,
        batch_size: int = 4,
        num_workers: int = 0,
        limit: Optional[int] = None,
        atom_vocab: Optional[List[str]] = None,
        task_type: str = "diffusion_oareactdiff",
        **kwargs: Any,  # noqa: ARG002 - data_type/use_ohe_feature/etc.
    ) -> None:
        self.train_pkl = train_pkl
        self.valid_pkl = valid_pkl or train_pkl
        self.center = center
        self.zero_charge = zero_charge
        self.single_frag_only = single_frag_only
        self.swapping_react_prod = swapping_react_prod
        self.use_by_ind = use_by_ind
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit = limit
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(OAREACTDIFF_ATOM_VOCAB)
        )
        self.task_type = task_type
        self.kwargs = kwargs

        self.train_set: Optional[OAReactDiffTS1xDataset] = None
        self.valid_set: Optional[OAReactDiffTS1xDataset] = None
        self.test_set: Optional[OAReactDiffTS1xDataset] = None
        self.collate_fn = partial(
            oareactdiff_collate, zero_charge=zero_charge
        )

    def _split(self, pkl_path: str) -> OAReactDiffTS1xDataset:
        return OAReactDiffTS1xDataset(
            pkl_path,
            center=self.center,
            zero_charge=self.zero_charge,
            single_frag_only=self.single_frag_only,
            swapping_react_prod=self.swapping_react_prod,
            use_by_ind=self.use_by_ind,
            limit=self.limit,
        )

    def load(self) -> None:
        """Build the train / valid / test datasets."""
        if self.valid_pkl == self.train_pkl and self.use_by_ind:
            logger.warning(
                "train_pkl and valid_pkl are the same file (%s). Both splits "
                "will use the SAME use_ind list, so validation reactions are "
                "training reactions. Point valid_pkl at valid_addprop.pkl.",
                self.train_pkl,
            )
        self.train_set = self._split(self.train_pkl)
        self.valid_set = self._split(self.valid_pkl)
        self.test_set = self.valid_set  # upstream ships no test.pkl
        logger.info(
            "OA-ReactDiff splits: train=%d valid=%d test=%d",
            len(self.train_set),
            len(self.valid_set),
            len(self.test_set),
        )
