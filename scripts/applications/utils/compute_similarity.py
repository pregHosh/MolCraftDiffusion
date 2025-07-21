#!/usr/bin/env python3
"""
max_similarity.py  –  find, for every SMILES in a *target* set, the most‑similar
SMILES in a *reference* set and report the maximum similarity value.

Both sets may be Python iterables or plain text files (one SMILES per line) or CSV files with a "smiles" column when you invoke the CLI.

Example (CLI):
    python max_similarity.py target.csv reference.txt results.csv

Example (library use):
    from max_similarity import max_similarity_df
    df = max_similarity_df(target_smiles, ref_smiles)
"""

from collections.abc import Iterable
import argparse
import logging
import os
import sys

import pandas as pd
import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs
from rdkit import Chem
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _mol_from_smiles(smi: str):
    """Return an RDKit Mol or raise ValueError if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi!r}")
    return mol


def _fp_from_mol(mol, fp_bits: int = 2048):
    """Morgan (ECFP‑like) bit vector fingerprint."""
    morgan_gen = GetMorganGenerator(radius=2, fpSize=fp_bits)
    return morgan_gen.GetFingerprint(mol)

# ────────────────────────────────────────────────────────────────────────────
# Core function
# ────────────────────────────────────────────────────────────────────────────

def max_similarity_df(
    target_smiles: Iterable[str],
    ref_smiles: Iterable[str],
    fp_bits: int = 2048,
    skip_identical=True
) -> pd.DataFrame:
    """
    For every SMILES string in *target_smiles*, compute the Tanimoto similarity
    to every SMILES in *ref_smiles* and return a DataFrame with:

        - smiles                       (original target SMILES)
        - max_similarity               (float, 0–1)
        - most_similar_smiles_in_ref   (SMILES from reference set giving max)

    Molecules that fail to parse are skipped and reported via logging.
    """
    # ── 1) Parse reference set once ──────────────────────────────────────
    ref_data = []
    for rsmi in tqdm(ref_smiles, total=len(ref_smiles)):
        try:
            mol = _mol_from_smiles(rsmi)
            ref_data.append((rsmi, _fp_from_mol(mol, fp_bits)))
        except ValueError:
            logging.warning("Skipping invalid reference SMILES: %s", rsmi)

    if not ref_data:
        raise RuntimeError("Reference set contains no valid SMILES")

    ref_smiles_ok, ref_fps = zip(*ref_data)  # tuples

    # ── 2) Iterate over target set ───────────────────────────────────────
    records = []
    n_fail = 0

    for tsmi in tqdm(target_smiles, total=len(target_smiles)):
        try:
            tmol = _mol_from_smiles(tsmi)
        except ValueError:
            logging.warning("Skipping invalid target SMILES: %s", tsmi)
            n_fail += 1
            continue
        
        tfp = _fp_from_mol(tmol, fp_bits)
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(tfp, ref_fps), dtype=float)

        if skip_identical:
            mask = ~np.isclose(sims, 1.0, atol=1e-9)
            if mask.any():
                idx = np.flatnonzero(mask)
                real_idx = int(idx[np.argmax(sims[mask])])
            else:
                # all identical, pick first non-identical fallback to any
                real_idx = int(np.argmax(sims))
        else:
            real_idx = int(np.argmax(sims))

        best_sim = float(sims[real_idx])
        best_smi = ref_smiles_ok[real_idx]
        records.append({
            "smiles": tsmi,
            "max_similarity": best_sim,
            "most_similar_smiles_in_ref": best_smi,
        })

    if n_fail:
        logging.info("Discarded %d invalid target SMILES", n_fail)

    return pd.DataFrame(records)

# ────────────────────────────────────────────────────────────────────────────
# File iterator supporting .txt or .csv
# ────────────────────────────────────────────────────────────────────────────

def _iter_smiles_from_file(path: str):
    """
    Yield SMILES strings from a plain text file (one per line) or a CSV file
    with a column named 'smiles'."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(path)
        if 'smiles' not in df.columns:
            raise ValueError(f"CSV file {path} has no 'smiles' column")
        for smi in df['smiles']:
            if pd.notna(smi):
                yield str(smi).strip()
    else:
        with open(path) as fh:
            for line in fh:
                smi = line.strip()
                if smi:
                    yield smi

def _cli():
    parser = argparse.ArgumentParser(
        description="Compute per‑molecule maximum Tanimoto similarity "
                    "between a TARGET set and a REFERENCE set of SMILES."
    )
    parser.add_argument("target_file", help="Text (.txt) or CSV (.csv) file: SMILES data")
    parser.add_argument("reference_file", help="Text (.txt) or CSV (.csv) file: SMILES data")
    parser.add_argument("output_csv", help="Where to write the results as CSV")
    parser.add_argument("--bits", type=int, default=2048,
                        help="Number of bits in the Morgan fingerprint (default: 2048)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tgt = list(_iter_smiles_from_file(args.target_file))
    ref = list(_iter_smiles_from_file(args.reference_file))
    logging.info("Target SMILES: %d | Reference SMILES: %d", len(tgt), len(ref))

    df = max_similarity_df(tgt, ref, fp_bits=args.bits)
    df.to_csv(args.output_csv, index=False)
    print(df.describe())
    logging.info("Wrote %d rows to %s", len(df), args.output_csv)


if __name__ == "__main__" and sys.argv[0].endswith(".py"):
    _cli()

