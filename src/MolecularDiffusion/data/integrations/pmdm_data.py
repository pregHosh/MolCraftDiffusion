"""Dataset / collate / DataModule for PMDM pocket-ligand complexes.

PMDM carries two node sets per complex -- a diffused ligand point cloud and a
fixed full-atom pocket point cloud -- flat-concatenated with
``torch_scatter``-style batch indices. That is not the platform's padded
single-node-set PointCloud batch, so this module brings its own Dataset and
``collate_fn`` and is selected via ``_target_`` from
``configs/data/pmdm_dataset.yaml``. Direct precedent:
``data/component/diffpharma_data.py``, which does the same for a four-node-set
model through the same seams (``cli/train.py`` -> ``.load()`` /
``.train_set`` / ``.collate_fn``, then ``data/lightning_data_module.py``'s
``collate_fn or graph_collate``).

Storage is one ASE db row per complex, written offline by
``docs/model_integrations/pmdm/scripts/convert_dataset.py``:

    Atoms            -> ligand only (positions + raw element Z)
    key_value_pairs  -> name, n_lig, n_pocket
    data             -> lig_feat (N_lig, 8) rdkit atom-family flags,
                        pocket_coords, pocket_element,
                        pocket_aa_type, pocket_is_backbone

Everything in the db is a **raw integer**, never one-hot. This module owns the
one-hot expansion, so changing a vocabulary does not mean reconverting -- and
it keeps PMDM's PDB/SDF parsing (and its ``np.bool``/``np.long`` calls, all
removed in modern numpy) entirely out of the runtime path.

The collate deliberately emits **PMDM's own attribute names**, so the task
adapter is a one-line ``SimpleNamespace(**batch)``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from MolecularDiffusion.optional import require_modules

logger = logging.getLogger(__name__)

#: PMDM ``FeaturizeLigandAtom(pocket=True)`` / ``FeaturizeProteinAtom(pocket=True)``
#: element vocabularies -- identical lists (``utils/transforms.py:45-47``).
PMDM_ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17, 34, 119]

#: Human-readable names for :data:`PMDM_ELEMENTS`, i.e. the atom decoder of
#: PMDM's ``crossdock`` (pocket) dataset config.
PMDM_ATOM_VOCAB = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "others"]

#: ``FeaturizeProteinAtom.max_num_aa``
MAX_NUM_AA = 20

#: PMDM folds bromine into chlorine before featurising (its vocab has no Br).
_ELEMENT_REMAP = {35: 17}


#: PMDM ``utils/protein_ligand.py:18-28`` -- amino-acid index vocabulary.
AA_NAME_NUMBER = {
    k: i
    for i, k in enumerate(
        "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU "
        "MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
    )
}
#: Reverse of :data:`AA_NAME_NUMBER`, for writing a PDB back out.
_AA_NUMBER_NAME = {v: k for k, v in AA_NAME_NUMBER.items()}
BACKBONE_NAMES = ["CA", "C", "N", "O"]


def _one_hot_elements(z: np.ndarray) -> torch.Tensor:
    """``(N,)`` raw atomic numbers -> ``(N, len(PMDM_ELEMENTS))`` float one-hot.

    Elements outside the vocabulary become an all-zero row, exactly as
    upstream's boolean ``==`` comparison does.
    """
    z = np.array(z, dtype=np.int64)
    for src, dst in _ELEMENT_REMAP.items():
        z[z == src] = dst
    vocab = np.asarray(PMDM_ELEMENTS, dtype=np.int64)
    return torch.from_numpy((z[:, None] == vocab[None, :]).astype(np.float32))


class PMDMDataset(Dataset):
    """One converted ASE db -> one pocket-ligand complex per index.

    ``center=True`` subtracts the joint ligand+pocket centroid (upstream's
    training-time behaviour, ``center_pos_pl`` re-centres anyway); generation
    uses ``center=False`` so the sampled ligand lands in the input pocket's
    own frame.

    # ponytail: rows are read eagerly into RAM at construction, like
    # DiffPharmaDataset. 31 complexes here, fine to ~100k. Stream by row id in
    # __getitem__ if a full CrossDocked split ever needs it.
    """

    def __init__(
        self, db_path: str, center: bool = True, limit: Optional[int] = None
    ) -> None:
        require_modules("data", ["ase"])
        from ase.db import connect

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"PMDM db not found: {db_path}")

        self.db_path = db_path
        self.center = center
        self.entries: List[Dict[str, Any]] = []

        with connect(db_path) as db:
            for row in db.select():
                self.entries.append(self._row_to_item(row))
                if limit is not None and len(self.entries) >= limit:
                    break
        if not self.entries:
            raise ValueError(f"No complexes found in {db_path}")

    def _row_to_item(self, row) -> Dict[str, Any]:
        data = row.data
        # np.array(...) copies: ASE hands back read-only views, and
        # torch.from_numpy on one of those yields a non-writable tensor.
        lig_pos = np.array(row.positions, dtype=np.float32).reshape(-1, 3)
        pocket_pos = np.array(data["pocket_coords"], dtype=np.float32).reshape(-1, 3)

        if self.center:
            mean = (lig_pos.sum(0) + pocket_pos.sum(0)) / (len(lig_pos) + len(pocket_pos))
            lig_pos = lig_pos - mean
            pocket_pos = pocket_pos - mean

        lig_element_ohe = _one_hot_elements(row.numbers)
        lig_feat = torch.from_numpy(np.array(data["lig_feat"], dtype=np.float32))

        pocket_element_ohe = _one_hot_elements(data["pocket_element"])
        aa_ohe = torch.nn.functional.one_hot(
            torch.from_numpy(np.array(data["pocket_aa_type"], dtype=np.int64)),
            num_classes=MAX_NUM_AA,
        ).float()
        is_backbone = torch.from_numpy(np.array(data["pocket_is_backbone"], dtype=bool))

        return {
            "name": row.get("name", str(row.id)),
            "ligand_pos": torch.from_numpy(lig_pos),
            # what the score network diffuses (transforms.py:96)
            "ligand_atom_feature": lig_element_ohe,
            # [element | rdkit atom families] (transforms.py:92); carried for
            # parity with upstream, not read by the pocket network.
            "ligand_atom_feature_full": torch.cat([lig_element_ohe, lig_feat], dim=-1),
            "protein_pos": torch.from_numpy(pocket_pos),
            "protein_atom_feature": pocket_element_ohe,
            # [element | amino acid | is_backbone] = 31 dims (transforms.py:57-61)
            "protein_atom_feature_full": torch.cat(
                [pocket_element_ohe, aa_ohe, is_backbone.view(-1, 1).float()], dim=-1
            ),
            "protein_is_backbone": is_backbone,
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.entries[idx]


def parse_pocket_pdb(path: str) -> Dict[str, np.ndarray]:
    """Minimal PDB ATOM-line parser for a live pocket file, at generate time.

    Relocated from ``docs/model_integrations/pmdm/scripts/convert_dataset.py``
    (that script now imports this instead of keeping its own copy) so
    :class:`~MolecularDiffusion.modules.tasks.diffusion_pmdm.PMDMConstrainedGenerator`
    can read a plain ``pocket_file`` PDB directly, the same way
    ``PMDMDataset`` reads a converted db row. Returns **raw**
    arrays (element Z, amino-acid index, backbone flag) -- one-hot expansion
    is the caller's job, same convention as :class:`PMDMDataset`.
    """
    from rdkit import Chem  # noqa: PLC0415

    ptable = Chem.GetPeriodicTable()
    coords, element, aa_type, is_backbone = [], [], [], []
    with open(path) as f:
        for line in f:
            if line[0:6].strip() == "ENDMDL":
                break
            if line[0:6].strip() != "ATOM":
                continue
            symb = line[76:78].strip().capitalize() or line[13:14]
            res_name = line[17:20].strip()
            if res_name not in AA_NAME_NUMBER:
                # Non-standard residue: upstream would KeyError. Skip the
                # whole complex rather than silently mislabel it.
                raise KeyError(res_name)
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            element.append(ptable.GetAtomicNumber(symb))
            aa_type.append(AA_NAME_NUMBER[res_name])
            is_backbone.append(line[12:16].strip() in BACKBONE_NAMES)
    if not coords:
        raise ValueError(f"no ATOM records found in {path}")
    return {
        "pos": np.array(coords, dtype=np.float32),
        "element": np.array(element, dtype=np.int64),
        "aa_type": np.array(aa_type, dtype=np.int64),
        "is_backbone": np.array(is_backbone, dtype=bool),
    }


def write_pocket_pdb(pocket: Dict[str, np.ndarray], path: str) -> None:
    """Write a pocket back to a PDB that :func:`parse_pocket_pdb` can read.

    Only what the parser actually checks round-trips exactly: element,
    amino-acid identity, coordinates, and backbone-set membership (not the
    specific atom name -- ``"CA"``/``"CB"`` stand in for backbone/sidechain).
    Used by ``docs/model_integrations/pmdm/scripts/extract_smoke_pair.py`` to
    reconstruct a smoke-test pocket from a converted db row (the original raw
    PDBs were deleted after conversion).
    """
    from rdkit import Chem  # noqa: PLC0415

    ptable = Chem.GetPeriodicTable()
    n = len(pocket["pos"])
    with open(path, "w") as f:
        for i in range(n):
            elem = ptable.GetElementSymbol(int(pocket["element"][i]))
            resname = _AA_NUMBER_NAME[int(pocket["aa_type"][i])]
            name = "CA" if bool(pocket["is_backbone"][i]) else "CB"
            x, y, z = (float(v) for v in pocket["pos"][i])
            f.write(
                f"ATOM  {i + 1:5d} {name:<4s} {resname:<3s} A{1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{0.0:6.2f}{0.0:6.2f}          {elem:>2s}\n"
            )


def _read_ligand_sdf(mol_file: str) -> Dict[str, torch.Tensor]:
    """Read a ligand SDF with a real 3D conformer, at generate time.

    Positions + element one-hot only (against :data:`PMDM_ELEMENTS`) -- the
    pocket network never reads the rdkit ``ATOM_FAMILIES`` feature
    (``ligand_atom_feature_full`` is carried by :class:`PMDMDataset` for
    training parity only), so this skips that feature-factory work entirely.
    """
    from rdkit import Chem  # noqa: PLC0415

    supplier = Chem.SDMolSupplier(mol_file, removeHs=False, sanitize=False)
    mol = next(iter(supplier), None)
    if mol is None:
        raise ValueError(f"could not read a molecule from {mol_file}")
    if mol.GetNumConformers() == 0:
        raise ValueError(f"{mol_file} has no 3D conformer -- a 2D sketch is not enough.")
    conf = mol.GetConformer()
    pos = np.array(conf.GetPositions(), dtype=np.float32)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return {"pos": torch.from_numpy(pos), "atom_feature": _one_hot_elements(z)}


def fully_connected_edges(sizes: List[int]) -> torch.Tensor:
    """Per-molecule fully-connected, self-loop-free edge index with offsets.

    This is what PMDM's ``GetAdj()`` transform builds (``utils/transforms.py``:
    ``get_adj_matrix(n)``, bond type constant 2) -- the real bonds are thrown
    away before the model ever sees them.
    """
    blocks = []
    offset = 0
    for n in sizes:
        idx = torch.arange(n)
        row, col = torch.meshgrid(idx, idx, indexing="ij")
        mask = row != col
        blocks.append(torch.stack([row[mask], col[mask]], dim=0) + offset)
        offset += n
    if not blocks:
        return torch.zeros((2, 0), dtype=torch.int64)
    return torch.cat(blocks, dim=1)


def pmdm_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate complexes; emit PMDM's own attribute names."""
    lig_sizes = [len(item["ligand_pos"]) for item in batch]
    pocket_sizes = [len(item["protein_pos"]) for item in batch]
    n = len(batch)

    def cat(key: str) -> torch.Tensor:
        return torch.cat([item[key] for item in batch], dim=0)

    edge_index = fully_connected_edges(lig_sizes)
    return {
        "names": [item["name"] for item in batch],
        "ligand_pos": cat("ligand_pos"),
        "ligand_atom_feature": cat("ligand_atom_feature"),
        "ligand_atom_feature_full": cat("ligand_atom_feature_full"),
        "ligand_element_batch": torch.repeat_interleave(
            torch.arange(n, dtype=torch.int64), torch.tensor(lig_sizes)
        ),
        "ligand_bond_index": edge_index,
        "ligand_bond_type": torch.full(
            (edge_index.size(1),), 2, dtype=torch.int64
        ),
        "num_nodes_per_graph": torch.tensor(lig_sizes, dtype=torch.int64),
        "protein_pos": cat("protein_pos"),
        "protein_atom_feature": cat("protein_atom_feature"),
        "protein_atom_feature_full": cat("protein_atom_feature_full"),
        "protein_is_backbone": cat("protein_is_backbone"),
        "protein_element_batch": torch.repeat_interleave(
            torch.arange(n, dtype=torch.int64), torch.tensor(pocket_sizes)
        ),
    }


class PMDMDataModule:
    """DataModule contract: ``load()`` + ``train_set``/``valid_set``/``test_set``.

    # ponytail: one db, no split -- ``val_file``/``test_file`` default to the
    # training db. The smoke set is 31 complexes; add a real split (and LMDB
    # streaming) when a full CrossDocked run needs one.
    """

    def __init__(
        self,
        root: str,
        train_file: str = "pmdm_smoke.db",
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        center: bool = True,
        limit: Optional[int] = None,
        atom_vocab: Optional[List[str]] = None,
        task_type: str = "diffusion_pmdm",
        **kwargs: Any,
    ) -> None:
        self.root = root
        self.train_file = train_file
        self.val_file = val_file or train_file
        self.test_file = test_file or self.val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.center = center
        self.limit = limit
        self.atom_vocab = list(atom_vocab) if atom_vocab else list(PMDM_ATOM_VOCAB)
        self.task_type = task_type
        self.kwargs = kwargs

        self.train_set: Optional[PMDMDataset] = None
        self.valid_set: Optional[PMDMDataset] = None
        self.test_set: Optional[PMDMDataset] = None
        self.collate_fn = pmdm_collate

    def load(self) -> None:
        cache: Dict[str, PMDMDataset] = {}

        def _split(filename: str) -> PMDMDataset:
            path = os.path.join(self.root, filename)
            if path not in cache:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"PMDM db not found: {path}. Convert pocket/ligand pairs "
                        "first (docs/model_integrations/pmdm/scripts/convert_dataset.py)."
                    )
                cache[path] = PMDMDataset(path, center=self.center, limit=self.limit)
            return cache[path]

        self.train_set = _split(self.train_file)
        self.valid_set = _split(self.val_file)
        self.test_set = _split(self.test_file)
        logger.info(
            "PMDM splits: train=%d valid=%d test=%d (root=%s)",
            len(self.train_set),
            len(self.valid_set),
            len(self.test_set),
            self.root,
        )
