"""Dataset / collate / DataModule for DiffDec scaffold-pocket-R-group complexes.

DiffDec carries three node sets per example -- a fixed 3D scaffold, a protein
pocket, and the diffused R-group -- flat-concatenated into ONE padded node set
with five per-atom masks (``scaffold_only_mask``, ``pocket_mask``,
``scaffold_mask`` = scaffold+pocket, ``rgroup_mask``, ``anchors``). That is not
the platform's single-node-set PointCloud batch, so this module brings its own
Dataset and ``collate_fn`` and is selected via ``_target_`` from
``configs/data/diffdec_dataset.yaml``. Direct precedent:
``data/component/diffsbdd_data.py``, through the same seams
(``cli/train.py:597`` -> ``.load()`` / ``.train_set`` / ``.collate_fn``, then
``data/lightning_data_module.py``'s ``collate_fn or graph_collate``).

**Storage is upstream's own preprocessed ``.pt``, read unchanged.** DiffDec
publishes ``crossdocksingle_{train,test}_full.pt`` (Zenodo record 10527451) --
a plain ``list[dict]`` of exactly the tensors its model consumes. Converting
that into an ASE db and back would be a lossy round-trip for no gain: the five
masks and the anchor flag have no home in the PointCloud db's node-feature
columns. So the DataModule reads the list directly and
:func:`diffdec_collate` below is a port of upstream ``src/datasets.py``
``collate`` (l. 126-166), pocket branch.

Each row carries::

    uuid, name              -> list attrs, passed through untouched
    positions   (N, 3)      -> float32 coordinates
    one_hot     (N, 10)     -> element one-hot over DIFFDEC_ATOM_VOCAB
    charges     (N,)        -> atomic number (unused by the model, h = one_hot)
    anchors     (N,)        -> 1.0 on the single scaffold attachment atom
    scaffold_only_mask (N,) -> scaffold atoms, pocket excluded
    pocket_mask (N,)        -> pocket atoms
    scaffold_mask (N,)      -> scaffold_only | pocket  (everything NOT noised)
    rgroup_mask (N,)        -> the diffused R-group slots
    num_atoms               -> int, list attr

The R-group is padded to a fixed 10 slots with a fake ``'#'`` atom
(upstream ``parse_rgroup``, datasets.py l. 84-124), so R-group size is
*implicit*: the model emits ``'#'`` into unused slots and the generator strips
those rows. ``'#'`` is therefore a real class of the vocabulary here, never a
padding artefact to be filtered at load time.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

#: ``const.ATOM2IDX`` (DiffDec ``src/const.py`` l. 9), in index order. The
#: released weights' 10-wide one-hot is in exactly this order, so reordering
#: this list silently invalidates every converted checkpoint.
DIFFDEC_ATOM_VOCAB = ["C", "O", "N", "F", "S", "Cl", "Br", "I", "P", "#"]

#: ``const.CHARGES_LIST``. ``'#'`` is the fake padding atom -> Z = 0.
DIFFDEC_CHARGES = [6, 8, 7, 9, 16, 17, 35, 53, 15, 0]

#: Index of the fake atom, i.e. "this R-group slot is unused".
FAKE_ATOM_INDEX = DIFFDEC_ATOM_VOCAB.index("#")

#: ``const.DATA_LIST_ATTRS`` -- never padded, never tensorised.
DATA_LIST_ATTRS = frozenset(
    {
        "uuid",
        "name",
        "scaffold_smi",
        "rgroup_smi",
        "num_atoms",
        "cat",
        "rgroup_size",
        "anchors_str",
        "edge_index",
    }
)

#: ``const.DATA_ATTRS_TO_PAD``.
DATA_ATTRS_TO_PAD = frozenset(
    {
        "positions",
        "one_hot",
        "charges",
        "anchors",
        "scaffold_mask",
        "rgroup_mask",
        "pocket_mask",
        "scaffold_only_mask",
    }
)

#: ``const.DATA_ATTRS_TO_ADD_LAST_DIM`` -- stored (N,), model wants (N, 1).
DATA_ATTRS_TO_ADD_LAST_DIM = frozenset(
    {
        "charges",
        "anchors",
        "scaffold_mask",
        "rgroup_mask",
        "pocket_mask",
        "scaffold_only_mask",
    }
)


class DiffDecDataset(Dataset):
    """One upstream ``.pt`` -> one scaffold/pocket/R-group complex per index.

    No re-centring at load time: ``DiffDecTask.forward`` removes the partial
    centre of mass itself (w.r.t. ``center_of_mass``, ``anchors`` by default),
    exactly as upstream ``model_single.py:161-169`` does, and the generator
    adds that offset back so samples land in the original pocket's frame.

    # ponytail: the whole list is read into RAM, like DiffSBDDDataset. The
    # test split is 43 complexes / 939 KB; the full train split is 1.6 GB, so
    # switch to a memory-mapped or sharded read if you train on it.
    """

    def __init__(self, pt_path: str, limit: Optional[int] = None) -> None:
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"DiffDec dataset not found: {pt_path}. Fetch it from Zenodo "
                "record 10527451 -- see "
                "docs/model_integrations/diffdec/scripts/convert_dataset.py"
            )
        # weights_only=False: the payload is plain tensors/ints/strs, but
        # torch<2.6 has no weights_only default and >=2.6 would reject the
        # bare `list` container. Provenance is the Zenodo DOI in the header.
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not isinstance(data, list) or not data:
            raise ValueError(
                f"{pt_path} is not a non-empty list of complexes "
                f"(got {type(data).__name__})"
            )
        self.data: List[Dict[str, Any]] = (
            data[:limit] if limit is not None else data
        )
        self.pt_path = pt_path

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def diffdec_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Port of upstream ``src/datasets.py::collate`` (l. 126-166).

    Pocket branch only (``pocket_mask`` is always present in the single-R-group
    CrossDocked data): ``edge_mask`` is emitted as a flat **batch-index
    vector**, not an adjacency mask. The actual graph is a 4 A radius graph
    rebuilt every forward pass by
    ``DynamicsWithPockets.get_dist_edges_4A``, which uses this vector only to
    forbid edges between different samples.
    """
    out: Dict[str, Any] = {}
    for data in batch:
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in list(out.items()):
        if key in DATA_LIST_ATTRS:
            continue
        if key in DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(
                value, batch_first=True, padding_value=0
            )
            continue
        raise ValueError(f"Unknown batch key: {key}")

    atom_mask = (
        out["scaffold_mask"].bool() | out["rgroup_mask"].bool()
    ).to(torch.float32)
    out["atom_mask"] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.shape
    # Upstream builds this with torch.cat of per-sample constant vectors; the
    # repeat_interleave is the same vector. Kept float-free (int64) because it
    # is only ever compared for equality in get_dist_edges_4A.
    out["edge_mask"] = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int64), n_nodes
    ).to(atom_mask.device)

    for key in DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out:
            out[key] = out[key][:, :, None]

    return out


def create_template(
    tensor: torch.Tensor,
    scaffold_size: int,
    rgroup_size: int,
    fill: float = 0.0,
) -> torch.Tensor:
    """Port of upstream ``datasets.py::create_template`` (l. 216-220).

    Keeps the first ``scaffold_size`` rows (scaffold + pocket, which the
    collate ordering puts first) and replaces the R-group rows with a constant
    block -- zeros for coordinates/features, ones for ``rgroup_mask``.
    """
    values_to_keep = tensor[:scaffold_size]
    values_to_add = torch.full(
        (rgroup_size, tensor.shape[1]),
        fill,
        dtype=values_to_keep.dtype,
        device=values_to_keep.device,
    )
    return torch.cat([values_to_keep, values_to_add], dim=0)


def create_templates_for_rgroup_generation_single(
    data: Dict[str, Any], rgroup_sizes: torch.Tensor
) -> Dict[str, Any]:
    """Port of upstream ``datasets.py`` l. 222-244.

    Blanks the ground-truth R-group out of a real batch, leaving the scaffold
    and pocket intact, so sampling starts from noise in the R-group slots only.
    """
    decoupled_data = []
    for i, rgroup_size in enumerate(rgroup_sizes):
        rgroup_size = int(rgroup_size)
        data_dict: Dict[str, Any] = {}
        scaffold_size = int(data["scaffold_mask"][i].squeeze().sum())
        for key, value in data.items():
            if key == "num_atoms":
                data_dict[key] = scaffold_size + rgroup_size
                continue
            if key in DATA_LIST_ATTRS:
                data_dict[key] = value[i]
                continue
            if key in DATA_ATTRS_TO_PAD:
                fill = 1.0 if key == "rgroup_mask" else 0.0
                template = create_template(
                    value[i], scaffold_size, rgroup_size, fill=fill
                )
                if key in DATA_ATTRS_TO_ADD_LAST_DIM:
                    template = template.squeeze(-1)
                data_dict[key] = template
        decoupled_data.append(data_dict)

    return diffdec_collate(decoupled_data)


class DiffDecDataModule:
    """DataModule contract: ``load()`` + ``train_set``/``valid_set``/``test_set``.

    # ponytail: no split logic -- ``val_file``/``test_file`` fall back to the
    # training file. The published test split is 43 complexes and is what the
    # smoke test uses; point each at its own ``.pt`` for a real run.
    """

    def __init__(
        self,
        root: str,
        train_file: str = "crossdocksingle_test_full.pt",
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        limit: Optional[int] = None,
        atom_vocab: Optional[List[str]] = None,
        task_type: str = "diffusion_diffdec",
        **kwargs: Any,
    ) -> None:
        self.root = root
        self.train_file = train_file
        self.val_file = val_file or train_file
        self.test_file = test_file or self.val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit = limit
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(DIFFDEC_ATOM_VOCAB)
        )
        self.task_type = task_type
        self.kwargs = kwargs

        self.train_set: Optional[DiffDecDataset] = None
        self.valid_set: Optional[DiffDecDataset] = None
        self.test_set: Optional[DiffDecDataset] = None
        self.collate_fn = diffdec_collate

    def load(self) -> None:
        cache: Dict[str, DiffDecDataset] = {}

        def _split(filename: str) -> DiffDecDataset:
            path = os.path.join(self.root, filename)
            if path not in cache:
                cache[path] = DiffDecDataset(path, limit=self.limit)
            return cache[path]

        self.train_set = _split(self.train_file)
        self.valid_set = _split(self.val_file)
        self.test_set = _split(self.test_file)
        logger.info(
            "DiffDec splits: train=%d valid=%d test=%d (root=%s)",
            len(self.train_set),
            len(self.valid_set),
            len(self.test_set),
            self.root,
        )
