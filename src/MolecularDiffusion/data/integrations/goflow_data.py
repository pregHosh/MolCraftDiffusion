"""Dataset / collate / DataModule for GoFlow on RDB7.

The item is the shared :class:`~MolecularDiffusion.data.component.
reaction_data.Reaction` container (built once by
``docs/model_integrations/goflow/scripts/convert_dataset.py`` from RDB7's
raw ``.csv``/``.xyz`` and the frozen ``feat_dict.pkl`` -- see
``INTEGRATION_PLAN.md``, Data & Pretrained Provenance); the batch is the
native PyG ``Batch`` GotenNet's forward wants, built here.

Bond union, canonical -> native
--------------------------------

``ReactionSide.bond_index``/``bond_type`` are stored in the platform's
canonical 5-class vocabulary (``0=none, 1=SINGLE, 2=DOUBLE, 3=TRIPLE,
4=AROMATIC``), directed, per side -- the container's own rule. GotenNet's
vendored ``_extend_condensed_graph_edge`` expects a **native-scale packed
union**: ``edge_index (2, E)`` over the union of reactant and product bonds,
``edge_type (E,) = bond_type_r_native * 22 + bond_type_p_native``
(``cgr_graph_utils.py`` decodes it back with ``// 22``, ``% 22``). Both
directions live only here, in :func:`goflow_collate`, per
``INTEGRATION_PLAN.md``'s Bond Representation Mapping:

1. canonical -> native is a lookup (identity for 0/1/2/3, ``4 -> 12``;
   lossless on this corpus -- 0/1/2/3/12 are the only native values RDB7
   ever produces, verified at conversion time);
2. the dense union per side is built with ``to_dense_adj``, mirroring
   upstream's ``adj = r_adj_perm + p_adj_perm`` numpy version
   (``utils/datasets.py:116-121``);
3. each side's native type is read off at every nonzero union pair (``0``
   if that side has none there);
4. the two are packed with the same ``* 22 +`` scheme.

``atom_type`` is the reaction's raw atomic number (``z``), read by
``AtomCGREmbedding``'s ``nn.Embedding(100, ...)`` directly -- there is no
fixed atom vocabulary on the conditioning side (only on ``sample()``'s
output side; see ``diffusion_goflow.py``).
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj

from MolecularDiffusion.data.datasets.reaction_data import Reaction

logger = logging.getLogger(__name__)

#: RDKit's native ``BondType`` enum position -> canonical ``BOND_VOCAB``
#: index, i.e. the inverse of the plan's Bond Representation Mapping table.
#: Index by canonical class (0..4) to get the native scale back.
_CANONICAL_TO_NATIVE = torch.tensor([0, 1, 2, 3, 12], dtype=torch.long)

#: ``len(BOND_TYPES)`` in ``cgr_graph_utils.py`` / upstream's ``utils/chem.py``
#: -- the packing base, not a platform constant.
NATIVE_BOND_VOCAB_SIZE = 22


def _native_bond_matrix(bond_index: Optional[torch.Tensor], bond_type: Optional[torch.Tensor], n: int) -> torch.Tensor:
    """One side's canonical bonds -> a dense ``(n, n)`` native-scale matrix.

    ``0`` everywhere a side has no bond, matching upstream's convention that
    ``UNSPECIFIED`` and "no bond" share native index 0.
    """
    if bond_index is None or bond_index.numel() == 0:
        return torch.zeros(n, n, dtype=torch.long)
    native = _CANONICAL_TO_NATIVE[bond_type]
    return to_dense_adj(bond_index, edge_attr=native, max_num_nodes=n)[0].long()


def _build_data(reaction: Reaction) -> Data:
    """One :class:`Reaction` -> the PyG ``Data`` GotenNet's forward wants."""
    n = len(reaction)
    r_mat = _native_bond_matrix(reaction.reactant.bond_index, reaction.reactant.bond_type, n)
    p_mat = _native_bond_matrix(reaction.product.bond_index, reaction.product.bond_type, n)

    union_mask = (r_mat != 0) | (p_mat != 0)
    row, col = union_mask.nonzero(as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    edge_type = r_mat[row, col] * NATIVE_BOND_VOCAB_SIZE + p_mat[row, col]

    data = Data(
        atom_type=reaction.z,
        r_feat=reaction.reactant.feat,
        p_feat=reaction.product.feat,
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=n,
    )
    # Present for training and corpus-driven generation; simply absent for
    # a blind R/P-only query -- PyG drops a None-assigned attribute, so it
    # is left off the Data entirely rather than assigned None (see
    # reaction_data.py's module docstring on this exact gotcha).
    if reaction.ts_pos is not None:
        data.ts_pos = reaction.ts_pos
    return data


def goflow_collate(reactions: Sequence[Reaction], n: int = 1) -> Dict[str, Any]:
    """Reactions -> ``{"batch": <PyG Batch>}``, the one collate for both
    training and :class:`~MolecularDiffusion.modules.tasks.diffusion_goflow.
    GoFlowTSGenerator`.

    ``n`` tiles the whole input sequence (``list(reactions) * n``) before
    batching. Training's ``DataLoader`` calls this with ``n=1`` on a list of
    ``batch_size`` distinct sampled reactions (one copy of each);
    ``GoFlowTSGenerator._collate`` calls it with a single-reaction list and
    ``n=num_generate`` (one reaction, tiled into ``n`` independent copies,
    each starting from its own fresh Gaussian draw at sampling time) --
    exactly ``oareactdiff_collate``'s ``[reaction] * n`` tiling, pushed
    inside the shared collate so both call sites share one function.

    Args:
        reactions: the reactions to batch.
        n: how many times to repeat the whole sequence before batching.

    Returns:
        ``{"batch": batch}``, where ``batch`` carries ``.atom_type, .r_feat,
        .p_feat, .edge_index, .edge_type, .batch`` (always) and ``.ts_pos``
        (only when every tiled reaction has one).
    """
    tiled = list(reactions) * n
    batch = Batch.from_data_list([_build_data(r) for r in tiled])
    return {"batch": batch}


class GoFlowRDB7Dataset(torch.utils.data.Dataset):
    """One RDB7 split, read from the converted ``Reaction`` pickle.

    Args:
        data_file: pickle of ``Dict[int, Reaction]`` keyed by the RDB7
            ``rxn`` id (built by ``scripts/convert_dataset.py``; the ``rxn``
            column has gaps, so this is a dict, not a list).
        feat_dict_file: the frozen ``feat_dict.pkl`` this corpus's
            ``ReactionSide.feat`` one-hot columns were built against.
            Read only to verify the vocabulary has not silently drifted
            (see below) -- the features themselves already live in
            ``data_file``, pre-encoded.
        split_path: directory holding the split pickle.
        split_file: which split pickle, e.g. ``random_split.pkl``.
        split: ``"train"``, ``"val"`` or ``"test"``. Defaults to ``"test"``
            (the held-out set) because this is also what
            :class:`~MolecularDiffusion.modules.tasks.diffusion_goflow.
            GoFlowTSGenerator` constructs for generation, where indexing
            the reactions the model trained on would be misleading.
        n_atom_rdkit_feats: the width ``feat_dict_file``'s per-descriptor
            cardinalities must sum to. A mismatch means the shipped
            ``feat_dict.pkl`` and the task's ``GotenNet(n_atom_rdkit_feats=
            ...)`` have drifted apart -- caught here, at data load, rather
            than at the first forward pass's shape mismatch.
        limit: cap the number of reactions in this split (smoke tests).
    """

    def __init__(
        self,
        data_file: str,
        feat_dict_file: str,
        split_path: str,
        split_file: str = "random_split.pkl",
        split: str = "test",
        n_atom_rdkit_feats: int = 27,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"GoFlow reaction pickle not found: {data_file}. Build it "
                "with docs/model_integrations/goflow/scripts/"
                "convert_dataset.py from RDB7's raw_data/ + processed_data/."
            )
        with open(feat_dict_file, "rb") as handle:
            feat_dict = pickle.load(handle)  # noqa: S301 - our own converter's artefact
        feat_width = sum(len(v) for v in feat_dict.values())
        if feat_width != n_atom_rdkit_feats:
            raise ValueError(
                f"feat_dict_file {feat_dict_file} sums to {feat_width} "
                f"one-hot columns, but n_atom_rdkit_feats={n_atom_rdkit_feats} "
                "was configured. Whatever produced this feat_dict.pkl no "
                "longer matches the task's GotenNet(n_atom_rdkit_feats=...) "
                "-- reconvert the dataset or fix the config, do not widen "
                "the vocabulary silently."
            )

        with open(data_file, "rb") as handle:
            reactions_by_id: Dict[int, Reaction] = pickle.load(handle)  # noqa: S301

        split_pkl = os.path.join(split_path, split_file)
        with open(split_pkl, "rb") as handle:
            split_dict = pickle.load(handle)  # noqa: S301 - upstream's own artefact
        if split not in split_dict:
            raise ValueError(
                f"split={split!r} not in {split_pkl} (keys: "
                f"{sorted(split_dict)})."
            )
        ids = list(split_dict[split])
        if limit is not None:
            ids = ids[:limit]

        missing = [i for i in ids if i not in reactions_by_id]
        if missing:
            raise KeyError(
                f"{len(missing)} reaction id(s) in {split_pkl}'s {split!r} "
                f"split are missing from {data_file} (first few: "
                f"{missing[:5]}). The converted pickle and this split file "
                "must come from the same RDB7 checkout."
            )
        self.reactions: List[Reaction] = [reactions_by_id[i] for i in ids]
        self.ids = ids

        logger.info(
            "GoFlow RDB7 %s split: %d reactions from %s (split file %s)",
            split, len(self.reactions), os.path.basename(data_file), split_file,
        )

    def __len__(self) -> int:
        return len(self.reactions)

    def __getitem__(self, idx: int) -> Reaction:
        return self.reactions[idx]


class GoFlowDataModule:
    """DataModule contract: ``load()`` + ``train_set``/``valid_set``/``test_set``.

    Three :class:`GoFlowRDB7Dataset` instances over the same converted
    pickle, one per split named in ``split_file`` (``random_split.pkl``,
    ``rxn_core_split.pkl`` or ``barrier_split.pkl`` -- all three ship
    in-tree, see ``INTEGRATION_PLAN.md``'s Hyperparameter Provenance table).
    """

    def __init__(
        self,
        data_file: str,
        feat_dict_file: str,
        split_path: str,
        split_file: str = "random_split.pkl",
        n_atom_rdkit_feats: int = 27,
        batch_size: int = 200,
        num_workers: int = 0,
        limit: Optional[int] = None,
        atom_vocab: Optional[List[str]] = None,
        task_type: str = "diffusion_goflow",
        **kwargs: Any,  # noqa: ARG002 - data_type/use_ohe_feature/etc.
    ) -> None:
        self.data_file = data_file
        self.feat_dict_file = feat_dict_file
        self.split_path = split_path
        self.split_file = split_file
        self.n_atom_rdkit_feats = n_atom_rdkit_feats
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit = limit
        # GoFlow has no output atom vocabulary of its own (see
        # diffusion_goflow.py); recorded here only because cli/train.py
        # reads atom_vocab off cfg.data regardless of _target_.
        self.atom_vocab = list(atom_vocab) if atom_vocab else ["H", "C", "N", "O", "F"]
        self.task_type = task_type
        self.kwargs = kwargs

        self.train_set: Optional[GoFlowRDB7Dataset] = None
        self.valid_set: Optional[GoFlowRDB7Dataset] = None
        self.test_set: Optional[GoFlowRDB7Dataset] = None
        self.collate_fn = goflow_collate

    def _split(self, split: str) -> GoFlowRDB7Dataset:
        return GoFlowRDB7Dataset(
            self.data_file,
            self.feat_dict_file,
            self.split_path,
            split_file=self.split_file,
            split=split,
            n_atom_rdkit_feats=self.n_atom_rdkit_feats,
            limit=self.limit,
        )

    def load(self) -> None:
        """Build the train / valid / test datasets."""
        self.train_set = self._split("train")
        self.valid_set = self._split("val")
        self.test_set = self._split("test")
        logger.info(
            "GoFlow RDB7 splits: train=%d valid=%d test=%d",
            len(self.train_set), len(self.valid_set), len(self.test_set),
        )
