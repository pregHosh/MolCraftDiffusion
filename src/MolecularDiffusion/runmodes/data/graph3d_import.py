"""Convert MiDi's raw QM9 / GEOM-Drugs data into the platform's 3D-graph ASE db.

The platform's existing dbs (e.g. ``data/qm9.db``) store SMILES but no
``mol_block``, so they carry no bond ground truth, and SMILES atom order does
not match the xyz order. Rather than perceive bonds from geometry, this imports
from MiDi's sources, which have real bond orders.

Storage schema, one row per molecule/conformer::

    Atoms  -> explicit H, RAW uncentered positions, RDKit atom order
    data   -> smiles          canonical
              bond_index      (2, E) int32, UPPER-TRIANGULAR (i < j), real bonds only
              bond_type       (E,)  int8, 1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC
              formal_charge   (N,)  int8, RAW SIGNED
              source          "qm9" | "geom"
              split           MiDi's split label, or ""
              <extra>         optional scalar per-molecule fields (QM9 property
                              targets + jodo_idx under ``--targets``); absent
                              unless asked for, so existing dbs are unchanged

Bond class 0 ("no bond") is never stored: absence from ``bond_index`` means no
bond, and the no-bond count is derived at statistics/materialization time.

## Aromaticity

MiDi reads gdb9 with ``sanitize=False``, so RDKit performs no aromatic
perception and its QM9 bond class 4 is never populated (verified: benzene
yields bond orders {1.0, 2.0} unsanitized vs {1.0, 1.5} sanitized). We
deliberately sanitize instead, so the db keeps the richer aromatic form, and
reproduce MiDi's Kekule distribution on demand via the dataset's
``kekulize=True`` load flag rather than by storing a second lossy copy.
``--no-sanitize`` restores exact MiDi parity if you need it.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import shutil
from typing import Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "RDKit is required to import 3D-graph datasets. "
            "Install it with: conda install -c conda-forge rdkit"
        ) from exc


def mol_to_row(
    mol,
    sanitize_failed_ok: bool = False,
    extra: Optional[dict] = None,
):
    """RDKit mol -> ``(Atoms, data_dict)`` for :func:`ase.db.core.Database.write`.

    Returns ``None`` when the molecule carries a bond type outside the canonical
    vocabulary (DATIVE, UNSPECIFIED, ...). Such molecules are counted and
    skipped rather than silently coerced to SINGLE.

    ``extra`` merges caller-supplied *scalar* per-molecule fields (property
    targets, an upstream row index, ...) into the written ``data`` dict. They
    land in the row metadata, where the dataset reads them through
    ``target_fields``. Default ``None`` -> the row schema is unchanged.
    """
    from ase import Atoms
    from rdkit import Chem

    from MolecularDiffusion.data.datasets.graph3d_dataset import BOND_ORDER_TO_CLASS

    conf = mol.GetConformer()
    positions = np.array(conf.GetPositions(), dtype=np.float64)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    charges = np.array([a.GetFormalCharge() for a in mol.GetAtoms()], dtype=np.int8)

    if not (len(numbers) == len(charges) == positions.shape[0] == mol.GetNumAtoms()):
        return None

    bond_i, bond_t = [], []
    for bond in mol.GetBonds():
        order = float(bond.GetBondTypeAsDouble())
        cls = BOND_ORDER_TO_CLASS.get(order)
        if cls is None:
            return None
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bond_i.append((i, j))
        bond_t.append(cls)

    bond_index = (
        np.array(bond_i, dtype=np.int32).T
        if bond_i
        else np.zeros((2, 0), dtype=np.int32)
    )
    bond_type = np.array(bond_t, dtype=np.int8)

    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001
        if not sanitize_failed_ok:
            return None
        smiles = ""

    atoms = Atoms(numbers=numbers, positions=positions)
    data = {
        "smiles": smiles,
        "bond_index": bond_index,
        "bond_type": bond_type,
        "formal_charge": charges,
    }
    if extra:
        data.update(extra)
    return atoms, data


# ---------------------------------------------------------------------------
# Source iterators -- the ONLY per-dataset code
# ---------------------------------------------------------------------------


# Hartree -> eV and kcal/mol -> eV, and the per-column selector, exactly as
# PyG's QM9 -- and JODO's copy of it, ``JODO/datasets/qm9_dataset.py:22-28``
# -- apply them. Either one wrong silently shifts every conditioning value.
_HAR2EV = 27.211386246
_KCALMOL2EV = 0.04336414
_QM9_CONVERSION = (
    1.0, 1.0, _HAR2EV, _HAR2EV, _HAR2EV, 1.0, _HAR2EV, _HAR2EV, _HAR2EV,
    _HAR2EV, _HAR2EV, 1.0, _KCALMOL2EV, _KCALMOL2EV, _KCALMOL2EV, _KCALMOL2EV,
    1.0, 1.0, 1.0,
)
# ``cat([target[:, 3:], target[:, :3]], -1)`` over csv columns 1..19.
_QM9_REORDER = (*range(3, 19), 0, 1, 2)


def qm9_targets(raw_dir: str) -> Tuple[list, np.ndarray]:
    """``gdb9.sdf.csv`` -> ``(names, values)`` in PyG/JODO order and units.

    ``values`` is indexed by *raw SDF index*, so it is joinable with the
    ``enumerate`` in :func:`iter_qm9` before the uncharacterized skip. Names
    come from the csv header rather than a hardcoded list, so they cannot
    drift out of step with the columns they label.
    """
    path = os.path.join(raw_dir, "gdb9.sdf.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; needed for --targets.")
    with open(path) as handle:
        lines = handle.read().split("\n")
    header = lines[0].split(",")[1:20]
    rows = [[float(x) for x in line.split(",")[1:20]] for line in lines[1:-1]]
    values = np.asarray(rows, dtype=np.float64)[:, _QM9_REORDER]
    values *= np.asarray(_QM9_CONVERSION, dtype=np.float64)
    return [header[i] for i in _QM9_REORDER], values


def iter_qm9(
    raw_dir: str, sanitize: bool = True, targets: bool = False
) -> Iterator[Tuple]:
    """Yield ``(mol, split)`` -- or ``(mol, split, extra)`` -- from gdb9.

    Skips the indices listed in ``uncharacterized.txt``, exactly as
    ``others/midi/midi/datasets/qm9_dataset.py:141-143`` does.

    With ``targets=True`` each molecule additionally carries the 19 QM9
    property targets under their csv names, plus ``jodo_idx`` -- the
    *compacted* index of upstream's ``Data`` list (the SDF enumerated with
    only the uncharacterized entries removed), so a published split file that
    indexes into it is joinable.
    """
    from rdkit import Chem

    sdf = os.path.join(raw_dir, "gdb9.sdf")
    if not os.path.exists(sdf):
        raise FileNotFoundError(
            f"{sdf} not found. Download MiDi's QM9 raw files first "
            "(see --help for the URLs)."
        )

    skip = set()
    uncharacterized = os.path.join(raw_dir, "uncharacterized.txt")
    if os.path.exists(uncharacterized):
        with open(uncharacterized) as handle:
            skip = {int(x.split()[0]) - 1 for x in handle.read().split("\n")[9:-2]}
        logger.info("Skipping %d uncharacterized molecules", len(skip))

    names, values = qm9_targets(raw_dir) if targets else ([], None)

    split_of = _qm9_split_map(raw_dir)
    supplier = Chem.SDMolSupplier(sdf, removeHs=False, sanitize=sanitize)
    jodo_idx = -1
    for idx, mol in enumerate(supplier):
        if idx in skip:
            continue
        # counts every non-uncharacterized entry, as upstream does
        jodo_idx += 1
        if mol is None:
            continue
        if values is None:
            yield mol, split_of.get(idx, "")
        else:
            extra = {
                n: float(v)
                for n, v in zip(names, values[idx], strict=True)
            }
            extra["jodo_idx"] = jodo_idx
            yield mol, split_of.get(idx, ""), extra


def ensure_qm9_splits(raw_dir: str) -> None:
    """Materialize MiDi's published QM9 split csvs if they are not already there.

    Byte-for-byte the procedure in ``QM9Dataset.download``
    (``others/midi/midi/datasets/qm9_dataset.py:121-134``): shuffle
    ``gdb9.sdf.csv`` with ``random_state=42`` and cut at 100k train / remainder
    val / 10% test. Reproduced here rather than imported so the converter does
    not depend on MiDi being importable -- but it must stay in sync, or
    published numbers stop being comparable.
    """
    import numpy as np
    import pandas as pd

    paths = {n: os.path.join(raw_dir, f"{n}.csv") for n in ("train", "val", "test")}
    if all(os.path.exists(p) for p in paths.values()):
        return

    source = os.path.join(raw_dir, "gdb9.sdf.csv")
    if not os.path.exists(source):
        logger.warning("No gdb9.sdf.csv in %s; molecules will be unsplit.", raw_dir)
        return

    dataset = pd.read_csv(source)
    n_samples = len(dataset)
    n_train = 100000
    n_test = int(0.1 * n_samples)
    train, val, test = np.split(
        dataset.sample(frac=1, random_state=42), [n_train, n_samples - n_test]
    )
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.to_csv(paths[name])
    logger.info(
        "Wrote MiDi split csvs to %s (train=%d val=%d test=%d)",
        raw_dir,
        len(train),
        len(val),
        len(test),
    )


def _qm9_split_map(raw_dir: str):
    """MiDi's published 100k/val/10% split (random_state=42)."""
    split_of = {}
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas unavailable; molecules will be unsplit.")
        return split_of

    ensure_qm9_splits(raw_dir)
    for name in ("train", "val", "test"):
        path = os.path.join(raw_dir, f"{name}.csv")
        if os.path.exists(path):
            for idx in pd.read_csv(path, index_col=0).index:
                split_of[int(idx)] = name
    if split_of:
        logger.info("MiDi split: %d labelled molecules", len(split_of))
    return split_of


def iter_geom(
    pickle_dir: str, max_conformers: int = 5, sanitize: bool = True
) -> Iterator[Tuple[object, str]]:
    """Yield ``(mol, split)`` from MiDi's GEOM-Drugs pickles.

    Each pickle is a list of ``(smiles, [conformers])``; MiDi caps at 5
    conformers per molecule (``geom_dataset.py:90-91``).
    """
    from rdkit import Chem

    for split in ("train", "val", "test"):
        path = os.path.join(pickle_dir, f"{split}_data.pickle")
        if not os.path.exists(path):
            path = os.path.join(pickle_dir, f"{split}.pickle")
        if not os.path.exists(path):
            logger.warning("No pickle for split %s in %s, skipping", split, pickle_dir)
            continue
        with open(path, "rb") as handle:
            entries = pickle.load(handle)
        for entry in entries:
            conformers = entry[1] if isinstance(entry, (tuple, list)) else entry
            for mol in list(conformers)[:max_conformers]:
                if mol is None:
                    continue
                if sanitize:
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:  # noqa: BLE001
                        continue
                yield mol, split


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_graph3d_db(
    mol_iter,
    out_path: str,
    source: str,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Write rows to an ASE db. Returns a counts dict."""
    from ase.db import connect

    if os.path.exists(out_path):
        raise FileExistsError(
            f"{out_path} already exists; refusing to append to an existing db. "
            "Remove it first or choose another path."
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    counts = {"written": 0, "skipped_bond_type": 0, "skipped_no_conformer": 0}
    db = connect(out_path)
    iterator = mol_iter
    if verbose:
        from tqdm import tqdm

        iterator = tqdm(mol_iter, desc=f"Converting {source}")

    with db:
        for item in iterator:
            mol, split, *rest = item
            extra = rest[0] if rest else None
            if limit is not None and counts["written"] >= limit:
                break
            try:
                if mol.GetNumConformers() == 0:
                    counts["skipped_no_conformer"] += 1
                    continue
                row = mol_to_row(mol, extra=extra)
                if row is None:
                    counts["skipped_bond_type"] += 1
                    continue
                atoms, data = row
                data["source"] = source
                data["split"] = split
                db.write(atoms, data=data)
                counts["written"] += 1
            except Exception as exc:  # noqa: BLE001
                counts.setdefault("failed", 0)
                counts["failed"] += 1
                logger.debug("Row failed: %s", exc)
    return counts


def verify_db(out_path: str, expected: int, n_check: int = 25) -> None:
    """Re-open the db and round-trip a sample back through RDKit.

    Raises if the db does not contain what we think it does. This gates the
    ``--delete-raw`` step, so it must be strict.
    """
    from ase.db import connect

    from MolecularDiffusion.data.datasets.graph3d_dataset import build_rdkit_mol

    db = connect(out_path)
    count = db.count()
    if count != expected:
        raise ValueError(f"db has {count} rows, expected {expected}")

    checked = 0
    for row in db.select(limit=n_check):
        data = dict(row.data)
        for key in ("bond_index", "bond_type", "formal_charge", "smiles"):
            if key not in data:
                raise ValueError(f"row {row.id} missing {key}")
        bond_index = np.asarray(data["bond_index"]).reshape(2, -1)
        if bond_index.size and not (bond_index[0] < bond_index[1]).all():
            raise ValueError(f"row {row.id} bond_index is not upper-triangular")
        build_rdkit_mol(
            row.toatoms().get_atomic_numbers(),
            bond_index,
            data["bond_type"],
            data["formal_charge"],
        )
        checked += 1
    logger.info("Verified %d/%d rows in %s", checked, count, out_path)


def delete_raw(raw_dir: str, artefacts) -> None:
    """Remove ONLY the named download artefacts under ``raw_dir``.

    Never touches the produced db and never leaves ``raw_dir``. Prints every
    path before unlinking, because this is irreversible.
    """
    raw_dir = os.path.abspath(raw_dir)
    for name in artefacts:
        path = os.path.abspath(os.path.join(raw_dir, name))
        if not path.startswith(raw_dir + os.sep):
            logger.warning("Refusing to delete outside raw dir: %s", path)
            continue
        if not os.path.exists(path):
            continue
        print(f"[import-graph3d] deleting raw artefact: {path}")
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)


QM9_RAW_ARTEFACTS = ("gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="MolCraftDiff data import-graph3d",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "QM9 raw files (MiDi's sources):\n"
            "  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
            "molnet_publish/qm9.zip   (unzip -> gdb9.sdf, gdb9.sdf.csv)\n"
            "  https://ndownloader.figshare.com/files/3195404   "
            "(save as uncharacterized.txt)\n"
            "GEOM: MiDi's train/val/test .pickle files (manual download)."
        ),
    )
    parser.add_argument("source", choices=("qm9", "geom"))
    parser.add_argument("raw_dir", help="directory holding the raw download")
    parser.add_argument("out_db", help="output .db path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-conformers", type=int, default=5)
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="parse without RDKit sanitization (exact MiDi QM9 parity; "
        "bond class 4/AROMATIC will never appear)",
    )
    parser.add_argument(
        "--targets",
        action="store_true",
        help="QM9 only: also write the 19 gdb9.sdf.csv property targets "
        "(PyG/JODO ordering and units) plus jodo_idx to every row",
    )
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="delete the raw download after the db is written AND verified",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _require_rdkit()
    sanitize = not args.no_sanitize

    if args.source == "qm9":
        mol_iter = iter_qm9(
            args.raw_dir, sanitize=sanitize, targets=args.targets
        )
    else:
        if args.targets:
            parser.error("--targets is QM9 only")
        mol_iter = iter_geom(
            args.raw_dir, max_conformers=args.max_conformers, sanitize=sanitize
        )

    counts = write_graph3d_db(mol_iter, args.out_db, source=args.source, limit=args.limit)
    print(f"[import-graph3d] {counts}")

    verify_db(args.out_db, expected=counts["written"])
    print(f"[import-graph3d] verified {args.out_db}")

    if args.delete_raw:
        if args.source != "qm9":
            logger.warning(
                "--delete-raw only knows QM9's artefacts; "
                "remove GEOM pickles manually."
            )
        else:
            delete_raw(args.raw_dir, QM9_RAW_ARTEFACTS)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
