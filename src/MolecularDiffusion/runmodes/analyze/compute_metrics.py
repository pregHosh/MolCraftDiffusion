import glob
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from MolecularDiffusion.optional import OptionalDependencyError, optional_import_error, require_modules

_IMPORT_ERROR = None
try:
    from MolecularDiffusion.utils.geometry.geom_utils import read_xyz_file, create_pyg_graph, correct_edges
    from MolecularDiffusion.utils.metrics.geom_metrics import (check_validity_v1,
                                                       check_chem_validity,
                                                       run_postbuster,
                                                       load_molecules_from_xyz,
                                                       check_neutrality,
                                                       xyz_to_rdkit_mol,
                                                       compute_drug_likeness)
    from MolecularDiffusion.utils.shepherd_score.extract_profiles import get_electrostatic_potential
    from MolecularDiffusion.utils.shepherd_score.generate_point_cloud import (
        get_molecular_surface, get_atomic_vdw_radii)
    from MolecularDiffusion.utils.shepherd_score.pharm_utils.pharmacophore import get_pharmacophores
    from MolecularDiffusion.utils.shepherd_score.score.constants import ALPHA, LAM_SCALING
    from MolecularDiffusion.utils.shepherd_score.score.gaussian_overlap_np import get_overlap_np
    from MolecularDiffusion.utils.shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
    from MolecularDiffusion.utils.shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np
    from MolecularDiffusion.utils.shepherd_score.alignment import (
        optimize_ROCS_overlay, optimize_ROCS_esp_overlay, optimize_pharm_overlay)
    from MolecularDiffusion.utils import smilify_xyz2mol, smilify_openbabel
    from MolecularDiffusion.utils.metrics.geom_stability import compute_molecules_stability
except ImportError as exc:
    _IMPORT_ERROR = optional_import_error("analyze", exc)

import logging
import matplotlib.pyplot as plt
try:
    from rdkit import RDLogger
except ImportError:
    RDLogger = None

# Suppress RDKit warnings
if RDLogger is not None:
    RDLogger.DisableLog('rdApp.*')

# Constants
EDGE_THRESHOLD = 4
SCALE_FACTOR = 1.2
SCORES_THRESHOLD = 3.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class _MetricsInput:
    input_path: Path
    xyz_dir: Path
    xyzs: list[str]
    source_by_xyz: dict[str, Path]
    row_by_xyz: dict[str, object]
    temp_dir: tempfile.TemporaryDirectory | None = None

    @property
    def is_db(self):
        return self.input_path.is_file() and self.input_path.suffix.lower() == ".db"

    def cleanup(self):
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def _prepare_metrics_input(input_path, portion):
    path = Path(input_path)
    if path.is_dir():
        xyzs = [
            str(p)
            for p in sorted(path.glob("*.xyz"))
            if "opt" not in p.name
        ]
        if portion < 1.0:
            random.shuffle(xyzs)
            xyzs = xyzs[:int(len(xyzs) * portion)]
        return _MetricsInput(
            input_path=path,
            xyz_dir=path,
            xyzs=xyzs,
            source_by_xyz={xyz: Path(xyz) for xyz in xyzs},
            row_by_xyz={},
        )

    if path.is_file() and path.suffix.lower() == ".db":
        from ase import io
        from ase.db import connect

        temp_dir = tempfile.TemporaryDirectory(prefix="molcraft_metrics_")
        temp_path = Path(temp_dir.name)
        xyzs = []
        source_by_xyz = {}
        row_by_xyz = {}
        with connect(str(path)) as db:
            for row in db.select():
                xyz_path = temp_path / f"row_{row.id}.xyz"
                io.write(str(xyz_path), row.toatoms())
                xyz = str(xyz_path)
                xyzs.append(xyz)
                source_by_xyz[xyz] = path
                row_by_xyz[xyz] = row
        if portion < 1.0:
            random.shuffle(xyzs)
            xyzs = xyzs[:int(len(xyzs) * portion)]
        return _MetricsInput(
            input_path=path,
            xyz_dir=temp_path,
            xyzs=xyzs,
            source_by_xyz=source_by_xyz,
            row_by_xyz=row_by_xyz,
            temp_dir=temp_dir,
        )

    raise ValueError(
        f"Unsupported metrics input '{input_path}'. Expected an XYZ directory or ASE .db file."
    )


def _add_db_row_ids(df, metrics_input):
    if not metrics_input.is_db or df is None or df.empty:
        return df
    df = df.copy()
    df["ase_db_row_id"] = [
        _row_id_for_result(row, metrics_input) for _, row in df.iterrows()
    ]
    return df


def _result_xyz_path(row, metrics_input):
    for column in ("file", "filename"):
        if column in row and pd.notna(row[column]):
            value = str(row[column])
            candidate = Path(value)
            if candidate.is_absolute() or candidate.parent != Path("."):
                if str(candidate) in metrics_input.source_by_xyz or candidate.exists():
                    return str(candidate)
            for xyz in metrics_input.xyzs:
                if Path(xyz).name == value:
                    return xyz
    return None


def _row_id_for_result(row, metrics_input):
    xyz_path = _result_xyz_path(row, metrics_input)
    row_obj = metrics_input.row_by_xyz.get(xyz_path)
    return getattr(row_obj, "id", None)


def _is_truthy_filter_value(value):
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _default_filtered_output(metrics_input):
    if metrics_input.is_db:
        return metrics_input.input_path.with_name(
            f"{metrics_input.input_path.stem}_filtered.db"
        )
    return metrics_input.input_path / "filtered_xyz"


def _default_metric_path(metrics_input, filename):
    if metrics_input.is_db:
        return str(
            metrics_input.input_path.with_name(
                f"{metrics_input.input_path.stem}_{filename}"
            )
        )
    return str(metrics_input.input_path / filename)


def _write_filtered_structures(filtered_df, metrics_input, output_path):
    output_path = Path(output_path)
    if metrics_input.is_db:
        from ase.db import connect

        if output_path.exists():
            output_path.unlink()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with connect(str(output_path)) as db_out:
            for _, result_row in filtered_df.iterrows():
                xyz_path = _result_xyz_path(result_row, metrics_input)
                row = metrics_input.row_by_xyz.get(xyz_path)
                if row is None:
                    continue
                db_out.write(
                    row.toatoms(),
                    key_value_pairs=dict(row.key_value_pairs),
                    data=dict(row.data),
                )
        return

    output_path.mkdir(parents=True, exist_ok=True)
    for _, result_row in filtered_df.iterrows():
        xyz_path = _result_xyz_path(result_row, metrics_input)
        source = metrics_input.source_by_xyz.get(xyz_path)
        if source is None:
            continue
        shutil.copy2(source, output_path / source.name)


def _apply_result_filter(result_tables, metrics_input, filter_column, filtered_output):
    if not filter_column:
        return

    matches = [
        (name, df, output_path)
        for name, df, output_path in result_tables
        if df is not None and filter_column in df.columns
    ]
    if not matches:
        available = sorted(
            {
                column
                for _, df, _ in result_tables
                if df is not None
                for column in df.columns
            }
        )
        raise ValueError(
            f"Filter column '{filter_column}' was not found in generated metrics. "
            f"Available columns: {', '.join(available)}"
        )
    if len(matches) > 1:
        table_names = ", ".join(name for name, _, _ in matches)
        raise ValueError(
            f"Filter column '{filter_column}' appears in multiple result tables "
            f"({table_names}). Run one metric type or choose a unique column."
        )

    name, df, output_path = matches[0]
    mask = df[filter_column].map(_is_truthy_filter_value)
    filtered_df = df[mask].copy()
    filtered_df.to_csv(output_path, index=False)
    structure_output = filtered_output or _default_filtered_output(metrics_input)
    _write_filtered_structures(filtered_df, metrics_input, structure_output)
    logging.info(
        f"Filtered {len(filtered_df)}/{len(df)} rows from {name} metrics to {output_path}"
    )
    logging.info(f"Filtered structures saved to {structure_output}")


def _get_split_stats(data_list, n_splits, scale=100.0):
    if len(data_list) == 0:
        return 0.0, 0.0
    if n_splits <= 1:
        return float(np.mean(data_list) * scale), 0.0
    splits = np.array_split(np.asarray(data_list), n_splits)
    split_means = [float(np.mean(s) * scale) for s in splits if len(s) > 0]
    if not split_means:
        return 0.0, 0.0
    return float(np.mean(split_means)), float(np.std(split_means))

def _perceive_mol(xyz, converter="xyz2mol", timeout=10):
    """Convert one .xyz to (smiles, mol), trying the other converter as backup.

    Returns ``(None, None)`` when both fail -- never raises, and never leaks
    state from a previous file.
    """
    order = ["xyz2mol", "openbabel"]
    if converter == "openbabel":
        order.reverse()

    for name in order:
        try:
            if name == "xyz2mol":
                smiles, mol = smilify_xyz2mol(xyz, timeout=timeout)
            else:
                smiles, mol = smilify_openbabel(xyz)
                # openbabel returns lists; take the single-molecule case
                if isinstance(smiles, (list, tuple)):
                    if len(smiles) != 1:
                        continue
                    smiles, mol = smiles[0], mol[0] if isinstance(mol, (list, tuple)) else mol
            if smiles and mol is not None:
                return smiles, mol
        except Exception:  # noqa: BLE001 -- a failed conversion is data, not an error
            continue
    return None, None


def _rdkit_valid(mol):
    """Standard validity: the molecule sanitizes and its SMILES round-trips."""
    if mol is None:
        return False
    from rdkit import Chem  # noqa: PLC0415

    try:
        probe = Chem.Mol(mol)
        Chem.SanitizeMol(probe)
        smiles = Chem.MolToSmiles(probe)
        return bool(smiles) and Chem.MolFromSmiles(smiles) is not None
    except Exception:  # noqa: BLE001
        return False


def _standardize_smiles(smiles):
    """Canonical SMILES with no explicit H and no stereo; None if unparseable.

    Every set metric compares strings, so both sides must pass through this.
    Perceived SMILES carry explicit ``[H]`` and reference sets are canonical
    heavy-atom SMILES (some with stereo): compared raw they match nothing and
    novelty sits at ~1.0 whatever the model. Formal charges and every fragment
    are kept on purpose -- the largest fragment is not taken.
    """
    from rdkit import Chem  # noqa: PLC0415

    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)
    except Exception:  # noqa: BLE001 -- a bad SMILES is data, not an error
        return None


def _set_level_metrics(smiles, train_smiles=None):
    """Uniqueness, novelty and diversity over the valid molecules.

    Generated and reference SMILES are both standardised with
    ``_standardize_smiles``; a generated SMILES RDKit cannot parse does not
    count as a valid SMILES.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem, DataStructs  # noqa: PLC0415

    raw = [s for s in smiles if s]
    smiles = [s for s in map(_standardize_smiles, raw) if s]
    if len(smiles) < len(raw):
        logging.warning(
            f"{len(raw) - len(smiles)} generated SMILES could not be parsed "
            "by RDKit and are excluded from uniqueness/novelty/diversity"
        )
    out = {"n_valid_smiles": len(smiles), "uniqueness": None,
           "novelty": None, "diversity": None}
    if not smiles:
        return out

    unique = sorted(set(smiles))
    out["uniqueness"] = len(unique) / len(smiles)

    if train_smiles:
        # one pass per run (~1 min for GEOM's 292k); only strings are kept
        known, n_skipped = set(), 0
        for s in train_smiles:
            std = _standardize_smiles(s)
            if std is None:
                n_skipped += 1
            else:
                known.add(std)
        logging.info(
            f"Reference SMILES standardised: {len(known)} unique kept, "
            f"{n_skipped} skipped as unparseable by RDKit"
        )
        out["novelty"] = sum(s not in known for s in unique) / len(unique)

    mols = [m for m in (Chem.MolFromSmiles(s) for s in unique) if m is not None]
    if len(mols) > 1:
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        fps = [gen.GetFingerprint(m) for m in mols]
        sims = [
            s
            for i in range(1, len(fps))
            for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        ]
        out["diversity"] = 1.0 - float(np.mean(sims))
    return out


def _load_train_smiles(path):
    """Read reference SMILES for novelty from a .txt (one per line) or .csv."""
    if path is None:
        return None
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        return set(df[col].dropna().astype(str))
    with open(path) as f:
        return {line.split()[0] for line in f if line.strip()}


def _write_summary(output_path, payload):
    """Write the run summary next to the CSV as JSON."""
    base, _ = os.path.splitext(output_path)
    summary_path = f"{base}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logging.info(f"Summary written to {summary_path}")
    return summary_path


def runner(args):
    split = getattr(args, "split", 1)
    if split < 1:
        raise ValueError(f"--split must be >= 1, got {split}")
    if _IMPORT_ERROR is not None:
        logging.warning(str(_IMPORT_ERROR))
        raise SystemExit(1) from _IMPORT_ERROR
    required_modules = {"torch", "torch_geometric", "ase", "rdkit"}
    if args.metrics in ["all", "posebuster", "geom_revised"]:
        required_modules.update({"openbabel", "posebusters"})
    if args.metrics in ["all", "similarity3d"]:
        required_modules.add("open3d")
    if args.metrics == "sbdd":
        try:
            require_modules("sbdd", {"vina", "meeko"})
        except OptionalDependencyError as exc:
            logging.warning(str(exc))
            raise SystemExit(1) from exc
    try:
        require_modules("analyze", required_modules)
    except OptionalDependencyError as exc:
        logging.warning(str(exc))
        raise SystemExit(1) from exc
    
    metrics_input = _prepare_metrics_input(args.input, args.portion)
    result_tables = []
    xyz_dir = str(metrics_input.xyz_dir)
    recheck_topo = args.recheck_topo
    # the neutrality check shells out to xTB once per molecule; off by default
    check_neutral = getattr(args, "check_neutrality", False)
    # every metric set derives its output paths from args.output, so make the
    # destination once here rather than in each branch
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # check_postbuster = args.check_postbuster # Removed, controlled by --metrics
    skip_idx = args.skip_atoms
    
    if skip_idx is None:
        skip_idx = []

    xyzs = list(metrics_input.xyzs)

    df_res_dict = {
        "file": [],
        "percent_atom_valid": [],
        "valid": [],
        "valid_geom": [],
        "valid_connected": [],
        "num_graphs": [],
        "bad_atom_distort": [],
        "bad_atom_chem": [],
        "n_bad_atom_distort": [],
        "n_bad_atom_chem": [],
        "neutral_molecule": [],
        "smiles": [],
        "num_atoms": [],
    }

    if args.metrics in ["all", "core"]:
        n_failed = 0
        for xyz in tqdm(xyzs, desc="Processing XYZ files", total=len(xyzs)):
            # every per-file value is reset here: a failure below must never
            # inherit the previous molecule's result
            num_atoms = None
            data = None
            coords = None
            smiles = None
            mol = None
            valid_geom = False
            percent_atom_valid = 0.0
            num_components = 100
            bad_atom_chem, bad_atom_distort = [], []

            try:
                coords, atomic_numbers_tensor = read_xyz_file(xyz)
                num_atoms = int(coords.size(0))
                data = create_pyg_graph(coords,
                                            atomic_numbers_tensor,
                                            xyz_filename=xyz,
                                            r=EDGE_THRESHOLD)
                data = correct_edges(data, scale_factor=SCALE_FACTOR)
                (valid_geom, percent_atom_valid, num_components, bad_atom_chem, bad_atom_distort) = \
                    check_validity_v1(data, score_threshold=SCORES_THRESHOLD,
                                      skip_indices=skip_idx,
                                      verbose=False)
            except Exception as e:
                n_failed += 1
                logging.error(f"Error processing {xyz}: {e}")
                if data is not None and hasattr(data, "num_nodes"):
                    bad_atom_chem = list(range(data.num_nodes))
                    bad_atom_distort = list(range(data.num_nodes))

            smiles, mol = _perceive_mol(xyz, args.mol_converter, args.timeout)
            is_valid = _rdkit_valid(mol)

            to_recheck = (
                recheck_topo
                and len(bad_atom_distort) > 0
                and len(bad_atom_chem) == 0
                and mol is not None
                and coords is not None
            )
            neutral_mol = check_neutrality(xyz) if check_neutral else None

            if to_recheck:
                try:
                    (natom_stability_dicts, _, _, _, bad_smiles_chem) = \
                        check_chem_validity([mol], skip_idx=skip_idx)
                    natom_stable = sum(natom_stability_dicts.values())
                    percent_atom_valid = natom_stable / coords.size(0)
                    if len(bad_smiles_chem) == 0:
                        valid_geom = True
                    else:
                        logging.warning(f"Detect bad smiles in {xyz}: {bad_smiles_chem}")
                except Exception as e:
                    logging.error(f"Fail to check on {xyz} due to {e}, assign invalid")
                    valid_geom = False
                    percent_atom_valid = 0.0

            is_valid_connected = bool(is_valid and num_components == 1)

            df_res_dict["smiles"].append(smiles)
            df_res_dict["file"].append(xyz)
            df_res_dict["percent_atom_valid"].append(percent_atom_valid)
            df_res_dict["valid"].append(is_valid)
            df_res_dict["valid_geom"].append(bool(valid_geom))
            df_res_dict["valid_connected"].append(is_valid_connected)
            df_res_dict["neutral_molecule"].append(neutral_mol)
            df_res_dict["num_graphs"].append(num_components)
            df_res_dict["bad_atom_distort"].append(" ".join(str(int(i)) for i in bad_atom_distort))
            df_res_dict["bad_atom_chem"].append(" ".join(str(int(i)) for i in bad_atom_chem))
            df_res_dict["n_bad_atom_distort"].append(len(bad_atom_distort))
            df_res_dict["n_bad_atom_chem"].append(len(bad_atom_chem))
            df_res_dict["num_atoms"].append(num_atoms)

        df = pd.DataFrame(df_res_dict)
        df = df.sort_values(by="file")
        df = _add_db_row_ids(df, metrics_input)
        fully_connected = [1 if num == 1 else 0 for num in df_res_dict["num_graphs"]]

        set_level = _set_level_metrics(
            df.loc[df["valid"], "smiles"].tolist(),
            _load_train_smiles(getattr(args, "train_smiles", None)),
        )

        logging.info(f"{df['percent_atom_valid'].mean() * 100:.2f}% of atoms are stable")
        logging.info(f"{df['valid'].mean() * 100:.2f}% of 3D molecules are valid (RDKit sanitize)")
        logging.info(f"{df['valid_geom'].mean() * 100:.2f}% of 3D molecules are valid (geometry+valency)")
        logging.info(f"{df['valid_connected'].mean() * 100:.2f}% of 3D molecules are valid and fully-connected")
        logging.info(f"{sum(fully_connected) / len(fully_connected) * 100:.2f}% of 3D molecules are fully connected")
        logging.info(f"{n_failed} of {len(xyzs)} files failed to process")
        for name in ("uniqueness", "novelty", "diversity"):
            value = set_level[name]
            logging.info(f"{name.capitalize()}: " + ("n/a" if value is None else f"{value * 100:.2f}%"))
        if split > 1:
            atom_mean, atom_std = _get_split_stats(df["percent_atom_valid"].values, split, scale=100.0)
            valid_mean, valid_std = _get_split_stats(df["valid"].values, split, scale=100.0)
            conn_mean, conn_std = _get_split_stats(df["valid_connected"].values, split, scale=100.0)
            full_mean, full_std = _get_split_stats(fully_connected, split, scale=100.0)
            logging.info(f"Split atoms stable mean ± std: {atom_mean:.2f} ± {atom_std:.2f}%")
            logging.info(f"Split valid mean ± std: {valid_mean:.2f} ± {valid_std:.2f}%")
            logging.info(f"Split valid&connected mean ± std: {conn_mean:.2f} ± {conn_std:.2f}%")
            logging.info(f"Split fully connected mean ± std: {full_mean:.2f} ± {full_std:.2f}%")
        
        logging.info(f"Molecular size mean: {df['num_atoms'].mean():.2f}")
        logging.info(f"Molecular size max: {df['num_atoms'].max()}")
        logging.info(f"Molecular size std: {df['num_atoms'].std():.2f}")

        if args.output is None:
            output_path = _default_metric_path(metrics_input, "output_metrics.csv")
            hist_path = _default_metric_path(metrics_input, "molecular_size_histogram.png")
        else:
            output_path = args.output
            base, _ = os.path.splitext(output_path)
            hist_path = f"{base}_molecular_size_histogram.png"

        plt.figure()
        plt.hist(df['num_atoms'], bins='auto')
        plt.title('Histogram of Molecular Sizes')
        plt.xlabel('Number of Atoms')
        plt.ylabel('Frequency')
        plt.savefig(hist_path)
        plt.close()
        logging.info(f"Molecular size histogram saved to {hist_path}")

        df.to_csv(output_path, index=False)

        summary = {
            "metrics": args.metrics,
            "input": xyz_dir,
            "n_files": len(xyzs),
            "n_failed": n_failed,
            "atom_stability": float(df["percent_atom_valid"].mean()),
            "validity": float(df["valid"].mean()),
            "validity_geom": float(df["valid_geom"].mean()),
            "validity_connected": float(df["valid_connected"].mean()),
            "fully_connected": sum(fully_connected) / len(fully_connected) if fully_connected else 0.0,
            "num_atoms_mean": float(df["num_atoms"].mean()),
            "num_atoms_std": float(df["num_atoms"].std()),
            "num_atoms_max": (None if df["num_atoms"].isna().all() else int(df["num_atoms"].max())),
            **set_level,
        }
        if split > 1:
            summary["split"] = {
                "n_splits": split,
                "atom_stability": [atom_mean, atom_std],
                "validity": [valid_mean, valid_std],
                "validity_connected": [conn_mean, conn_std],
                "fully_connected": [full_mean, full_std],
            }
        _write_summary(output_path, summary)
        result_tables.append(("core", df, output_path))

    if args.metrics in ["all", "posebuster"]:
        mols, xyz_passed = load_molecules_from_xyz(xyz_dir)
        
        if args.portion < 1.0:
             xyzs_set = set(xyzs)
             filtered_data = [(m, x) for m, x in zip(mols, xyz_passed) if x in xyzs_set]
             if filtered_data:
                 mols, xyz_passed = zip(*filtered_data)
                 mols = list(mols)
                 xyz_passed = list(xyz_passed)
             else:
                 mols, xyz_passed = [], []
        
        # one xTB call per molecule -- opt in via --check-neutrality
        if check_neutral:
            neutral_mols = [
                check_neutrality(xyz)
                for xyz in tqdm(xyz_passed, desc="Checking neutrality of molecules",
                                total=len(xyz_passed))
            ]
        else:
            neutral_mols = [None] * len(xyz_passed)


        names = [os.path.basename(xyz) for xyz in xyz_passed]
        # names= keeps a row (tagged posebuster_error) for every molecule
        # PoseBusters fails or times out on, so columns align by filename below
        postbuster_results = run_postbuster(mols, timeout=3000, names=names)
        if postbuster_results is not None:
            num_atoms_by_name = {n: mol.GetNumAtoms() for n, mol in zip(names, mols)}
            postbuster_results['num_atoms'] = postbuster_results['filename'].map(num_atoms_by_name)

            posebuster_checks = [
                'bond_lengths', 'bond_angles', 'internal_steric_clash',
                'aromatic_ring_flatness', 'non-aromatic_ring_non-flatness',
                'double_bond_flatness', 'internal_energy'
            ]
            postbuster_results['valid_posebuster'] = postbuster_results[posebuster_checks].all(axis=1)
            posebuster_checks_connected = posebuster_checks + ['all_atoms_connected']
            postbuster_results['valid_posebuster_connected'] = postbuster_results[posebuster_checks_connected].all(axis=1)
            if 'posebuster_error' in postbuster_results:
                # .all() skips NaN, so a molecule that never ran would pass
                failed = postbuster_results['posebuster_error'].notna()
                postbuster_results.loc[failed, ['valid_posebuster', 'valid_posebuster_connected']] = False
                logging.warning(
                    f"PoseBusters failed or timed out on {int(failed.sum())} of "
                    f"{len(failed)} molecules; kept with valid_posebuster=False"
                )
            if args.output is None:
                postbuster_output_path = _default_metric_path(metrics_input, "postbuster_metrics.csv")
                hist_path = _default_metric_path(metrics_input, "postbuster_molecular_size_histogram.png")
            else:
                base, ext = os.path.splitext(args.output)
                postbuster_output_path = f"{base}_postbuster{ext}"
                hist_path = f"{base}_postbuster_molecular_size_histogram.png"

            postbuster_results['neutral_molecule'] = postbuster_results['filename'].map(dict(zip(names, neutral_mols)))
            # moved last, where it has always been in the CSV
            postbuster_results["filename"] = postbuster_results.pop("filename")
            postbuster_results = _add_db_row_ids(postbuster_results, metrics_input)
            postbuster_results.to_csv(postbuster_output_path, index=False)
            result_tables.append(("posebuster", postbuster_results, postbuster_output_path))

            logging.info(f"Molecular size mean: {postbuster_results['num_atoms'].mean():.2f}")
            logging.info(f"Molecular size max: {postbuster_results['num_atoms'].max()}")
            logging.info(f"Molecular size std: {postbuster_results['num_atoms'].std():.2f}")

            plt.figure()
            plt.hist(postbuster_results['num_atoms'], bins='auto')
            plt.title('Histogram of Molecular Sizes (Posebuster)')
            plt.xlabel('Number of Atoms')
            plt.ylabel('Frequency')
            plt.savefig(hist_path)
            plt.close()
            logging.info(f"Molecular size histogram for posebuster saved to {hist_path}")

            logging.info(f"Sanitization: {postbuster_results['sanitization'].mean() * 100:.2f}%")
            logging.info(f"InChI Convertible: {postbuster_results['inchi_convertible'].mean() * 100:.2f}%")
            logging.info(f"All Atoms Connected: {postbuster_results['all_atoms_connected'].mean() * 100:.2f}%")
            logging.info(f"Bond Lengths: {postbuster_results['bond_lengths'].mean():.2f}")
            logging.info(f"Bond Angles: {postbuster_results['bond_angles'].mean():.2f}")
            logging.info(f"Internal Steric Clash: {postbuster_results['internal_steric_clash'].mean():.2f}")
            logging.info(f"Aromatic Ring Flatness: {postbuster_results['aromatic_ring_flatness'].mean():.2f}")
            logging.info(f"Non-Aromatic Ring Non-Flatness: {postbuster_results['non-aromatic_ring_non-flatness'].mean():.2f}")
            logging.info(f"Double Bond Flatness: {postbuster_results['double_bond_flatness'].mean():.2f}")
            logging.info(f"Internal Energy: {postbuster_results['internal_energy'].mean():.2f}")
            logging.info(f"Valid Posebuster: {postbuster_results['valid_posebuster'].mean() * 100:.2f}%")
            logging.info(f"Valid Posebuster Connected: {postbuster_results['valid_posebuster_connected'].mean() * 100:.2f}%")
            if check_neutral and neutral_mols:
                logging.info(f"Neutral Molecule: {sum(neutral_mols) / len(neutral_mols) * 100:.2f}%")
            if split > 1:
                sanit_mean, sanit_std = _get_split_stats(postbuster_results["sanitization"].values, split, scale=100.0)
                inchi_mean, inchi_std = _get_split_stats(postbuster_results["inchi_convertible"].values, split, scale=100.0)
                conn_mean, conn_std = _get_split_stats(postbuster_results["all_atoms_connected"].values, split, scale=100.0)
                valid_pb_mean, valid_pb_std = _get_split_stats(postbuster_results["valid_posebuster"].values, split, scale=100.0)
                valid_pb_conn_mean, valid_pb_conn_std = _get_split_stats(postbuster_results["valid_posebuster_connected"].values, split, scale=100.0)
                logging.info(f"Split Sanitization mean ± std: {sanit_mean:.2f} ± {sanit_std:.2f}%")
                logging.info(f"Split InChI Convertible mean ± std: {inchi_mean:.2f} ± {inchi_std:.2f}%")
                logging.info(f"Split All Atoms Connected mean ± std: {conn_mean:.2f} ± {conn_std:.2f}%")
                logging.info(f"Split Valid Posebuster mean ± std: {valid_pb_mean:.2f} ± {valid_pb_std:.2f}%")
                logging.info(f"Split Valid Posebuster Connected mean ± std: {valid_pb_conn_mean:.2f} ± {valid_pb_conn_std:.2f}%")
                if check_neutral and neutral_mols:
                    neutral_mean, neutral_std = _get_split_stats(neutral_mols, split, scale=100.0)
                    logging.info(f"Split Neutral Molecule mean ± std: {neutral_mean:.2f} ± {neutral_std:.2f}%")

    # =========================================================================
    # GEOM_REVISED: Aromatic-aware molecule stability
    # =========================================================================
    if args.metrics in ["all", "geom_revised"]:
        from rdkit import Chem
        from MolecularDiffusion.utils.chem.smilify import smilify_xyz2mol
        from MolecularDiffusion.utils.metrics.geom_stability import (
            compute_bond_lengths_diff,
            compute_bond_angles_diff,
            compute_torsion_angles_diff,
            compute_differences,
        )
        from collections import defaultdict

        def run_bond_analysis(pairs, analysis_type="bond_length"):
            accumulated_results = defaultdict(lambda: ([], []))
            if analysis_type == "bond_length":
                results = compute_differences(pairs, compute_bond_lengths_diff)
            elif analysis_type == "bond_angle":
                results = compute_differences(pairs, compute_bond_angles_diff)
            elif analysis_type == "torsion_angle":
                results = compute_differences(pairs, compute_torsion_angles_diff)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
            for key, (avg_diff, std_dev, weight) in results.items():
                accumulated_results[key][0].append(avg_diff)
                accumulated_results[key][1].append(weight)
            return accumulated_results

        def summarize_bond_results(results):
            total_weighted_diffs = []
            total_weight = 0
            for key, (avg_diff_list, weight_list) in results.items():
                weighted_diffs = np.array(avg_diff_list) * np.array(weight_list)
                total_weighted_diffs.append(np.sum(weighted_diffs))
                total_weight += np.sum(weight_list)
            return np.sum(total_weighted_diffs) / total_weight if total_weight > 0 else 0

        def get_split_stats_bond(pairs, analysis_type, n_splits):
            if len(pairs) == 0: return 0.0, 0.0
            if n_splits <= 1:
                results = run_bond_analysis(pairs, analysis_type=analysis_type)
                return summarize_bond_results(results), 0.0
            fold_size = len(pairs) // n_splits
            scores = []
            for i in range(n_splits):
                if i < n_splits - 1:
                    fold_pairs = pairs[i * fold_size: (i + 1) * fold_size]
                else:
                    fold_pairs = pairs[i * fold_size:]
                if not fold_pairs: continue
                results = run_bond_analysis(fold_pairs, analysis_type=analysis_type)
                score = summarize_bond_results(results)
                scores.append(score)
            if not scores: return 0.0, 0.0
            return np.mean(scores), np.std(scores)
        
        logging.info(f"Computing geom_revised metrics (converter: {args.mol_converter})...")
        
        # Load molecules from XYZ
        mols_revised = []
        xyz_files_revised = []
        mol_pairs = []
        mol_pair_mapping = {}
        
        xyzs_to_process = [
            path for path in glob.glob(f"{xyz_dir}/*.xyz")
            if 'opt' not in os.path.basename(path)
        ]
        
        if args.portion < 1.0:
            random.shuffle(xyzs_to_process)
            xyzs_to_process = xyzs_to_process[:int(len(xyzs_to_process) * args.portion)]
        
        for xyz_file in tqdm(xyzs_to_process, desc="Loading molecules for geom_revised"):
            try:
                if args.mol_converter == "xyz2mol":
                    smiles, mol = smilify_xyz2mol(xyz_file, timeout=args.timeout)
                    if mol is not None and mol.GetNumConformers() == 0:
                        try:
                            from rdkit.Geometry import Point3D
                            cart_coords, _ = read_xyz_file(xyz_file)
                            conf = Chem.Conformer(mol.GetNumAtoms())
                            for i in range(mol.GetNumAtoms()):
                                x, y, z = cart_coords[i].tolist()
                                conf.SetAtomPosition(i, Point3D(x, y, z))
                            mol.AddConformer(conf)
                        except Exception as e:
                            logging.debug(f"Failed to add conformer to {xyz_file}: {e}")
                            mol = None
                else:  # openbabel
                    from openbabel import pybel
                    mol_pb = next(pybel.readfile("xyz", xyz_file))
                    mol_sdf = mol_pb.write("sdf")
                    mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False, sanitize=False)
                    if mol is not None:
                        Chem.SanitizeMol(mol)
                
                # Generate optimized molecule using MMFF via OpenBabel
                opt_mol = None
                if mol is not None:
                    try:
                        from openbabel import pybel
                        mol_pb = next(pybel.readfile("xyz", xyz_file))
                        mol_pb.localopt(forcefield="mmff94", steps=1000)
                        
                        # Convert to RDKit mol
                        from rdkit import Chem
                        mol_sdf = mol_pb.write("sdf")
                        opt_mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False, sanitize=False)
                        if opt_mol is not None:
                            Chem.SanitizeMol(opt_mol)
                    except Exception as e:
                        logging.debug(f"OpenBabel MMFF Optimization failed for {xyz_file}: {e}")
                        opt_mol = None

                if mol is not None:
                    mols_revised.append(mol)
                    xyz_files_revised.append(xyz_file)
                    if opt_mol is not None:
                        mol_pairs.append((mol, opt_mol))
                        mol_pair_mapping[xyz_file] = opt_mol

            except Exception as e:
                logging.debug(f"Failed to load {xyz_file}: {e}")
        
        if len(mols_revised) == 0:
            logging.warning("No molecules loaded for geom_revised metrics")
        else:
            # Compute validity metrics first (independent of aromatic mode)
            from rdkit import Chem
            
            valid_list = []  # Sanitization only
            valid_connected_list = []  # Sanitization + single fragment
            
            for mol in mols_revised:
                try:
                    Chem.SanitizeMol(mol)
                    valid_list.append(1)
                    # Check single connected component
                    if len(Chem.GetMolFrags(mol)) == 1:
                        valid_connected_list.append(1)
                    else:
                        valid_connected_list.append(0)
                except:
                    valid_list.append(0)
                    valid_connected_list.append(0)
            
            # Compute stability for both aromatic modes
            # aromatic_true = MS Arom-Dependent Valence (tuple valencies)
            # aromatic_false = MS 1.5 Arom (sum all bonds including aromatic at 1.5)
            results_dict = {
                "file": xyz_files_revised,
                "num_atoms": [mol.GetNumAtoms() for mol in mols_revised],
                "valid": valid_list,
                "valid_connected": valid_connected_list,
            }
            
            # --- Bond Analysis per file ---
            bond_len_diffs = []
            bond_ang_diffs = []
            torsion_diffs = []
            
            for mol, xyz_file in zip(mols_revised, xyz_files_revised):
                opt_mol = mol_pair_mapping.get(xyz_file)
                if opt_mol is not None:
                    len_res = run_bond_analysis([(mol, opt_mol)], "bond_length")
                    ang_res = run_bond_analysis([(mol, opt_mol)], "bond_angle")
                    tor_res = run_bond_analysis([(mol, opt_mol)], "torsion_angle")
                    bond_len_diffs.append(summarize_bond_results(len_res))
                    bond_ang_diffs.append(summarize_bond_results(ang_res))
                    torsion_diffs.append(summarize_bond_results(tor_res))
                else:
                    bond_len_diffs.append(None)
                    bond_ang_diffs.append(None)
                    torsion_diffs.append(None)
            
            results_dict["bond_length_diff"] = bond_len_diffs
            results_dict["bond_angle_diff"] = bond_ang_diffs
            results_dict["torsion_diff"] = torsion_diffs
            
            modes_to_run = [
                ("aromatic_true", True),   # Arom-Dependent Valence
                ("aromatic_false", False), # 1.5 Arom mode
            ]
            
            for mode_name, aromatic_val in modes_to_run:
                try:
                    validity, stability, n_stable_atoms, n_atoms = compute_molecules_stability(
                        mols_revised, aromatic=aromatic_val
                    )
                    results_dict[f"stable_mol_{mode_name}"] = stability.tolist()
                    results_dict[f"n_stable_atoms_{mode_name}"] = n_stable_atoms.tolist()
                    results_dict[f"n_atoms_{mode_name}"] = n_atoms.tolist()
                    results_dict[f"atom_stability_{mode_name}"] = (n_stable_atoms / n_atoms).tolist()
                except Exception as e:
                    logging.error(f"Failed to compute stability for {mode_name}: {e}")
            
            df_revised = pd.DataFrame(results_dict)
            
            # Determine output path
            if args.output is None:
                revised_output_path = _default_metric_path(metrics_input, "geom_revised_metrics.csv")
            else:
                base, ext = os.path.splitext(args.output)
                revised_output_path = f"{base}_geom_revised{ext}"
            
            df_revised = _add_db_row_ids(df_revised, metrics_input)
            df_revised.to_csv(revised_output_path, index=False)
            result_tables.append(("geom_revised", df_revised, revised_output_path))
            logging.info(f"Geom revised metrics saved to {revised_output_path}")
            
            # Print summary statistics
            n_passed = len(mols_revised)
            n_total = len(xyzs_to_process)
            conversion_rate = n_passed / n_total * 100 if n_total > 0 else 0
            
            logging.info("=" * 60)
            logging.info("GEOM_REVISED STABILITY METRICS")
            logging.info("=" * 60)
            logging.info(f"XYZ2Mol Conversion: {n_passed}/{n_total} ({conversion_rate:.2f}%)")
            
            valid_global = (sum(valid_list) / n_passed * 100) if n_passed > 0 else 0.0
            conn_global = (sum(valid_connected_list) / n_passed * 100) if n_passed > 0 else 0.0
            logging.info(f"Valid: {sum(valid_list)}/{n_passed} ({valid_global:.2f}%)")
            logging.info(f"Valid & Connected: {sum(valid_connected_list)}/{n_passed} ({conn_global:.2f}%)")
            if split > 1:
                valid_mean, valid_std = _get_split_stats(valid_list, split, scale=100.0)
                conn_mean, conn_std = _get_split_stats(valid_connected_list, split, scale=100.0)
                logging.info(f"Split Valid mean ± std: {valid_mean:.2f} ± {valid_std:.2f}%")
                logging.info(f"Split Valid & Connected mean ± std: {conn_mean:.2f} ± {conn_std:.2f}%")
            
            for mode_name in ["aromatic_true", "aromatic_false"]:
                if f"stable_mol_{mode_name}" in df_revised.columns:
                    n_stable = int(df_revised[f'stable_mol_{mode_name}'].sum())
                    
                    mol_stab_global = float(df_revised[f'stable_mol_{mode_name}'].mean() * 100)
                    atom_stab_global = float(df_revised[f'atom_stability_{mode_name}'].mean() * 100)
                    
                    logging.info(f"--- Mode: {mode_name} ---")
                    logging.info(f"  Molecule Stability: {n_stable}/{n_passed} ({mol_stab_global:.2f}%)")
                    logging.info(f"  Atom Stability: {atom_stab_global:.2f}%")
                    if split > 1:
                        mol_stab_mean, mol_stab_std = _get_split_stats(df_revised[f'stable_mol_{mode_name}'].values, split, scale=100.0)
                        atom_stab_mean, atom_stab_std = _get_split_stats(df_revised[f'atom_stability_{mode_name}'].values, split, scale=100.0)
                        logging.info(f"  Split Molecule Stability mean ± std: {mol_stab_mean:.2f} ± {mol_stab_std:.2f}%")
                        logging.info(f"  Split Atom Stability mean ± std: {atom_stab_mean:.2f} ± {atom_stab_std:.2f}%")

            if len(mol_pairs) > 0:
                bond_global, _ = get_split_stats_bond(mol_pairs, "bond_length", 1)
                ang_global, _ = get_split_stats_bond(mol_pairs, "bond_angle", 1)
                tor_global, _ = get_split_stats_bond(mol_pairs, "torsion_angle", 1)
                
                logging.info(f"--- Geometry Discrepancies (Bond Mode) ---")
                logging.info(f"  Pairs loaded: {len(mol_pairs)}")
                logging.info(f"  Bond Length: {bond_global:.4f} Å")
                logging.info(f"  Bond Angle: {ang_global:.4f}°")
                logging.info(f"  Torsion Angle: {tor_global:.4f}°")
                if split > 1:
                    bond_len_mean, bond_len_std = get_split_stats_bond(mol_pairs, "bond_length", split)
                    bond_ang_mean, bond_ang_std = get_split_stats_bond(mol_pairs, "bond_angle", split)
                    tor_mean, tor_std = get_split_stats_bond(mol_pairs, "torsion_angle", split)
                    logging.info(f"  Split Bond Length mean ± std: {bond_len_mean:.4f} ± {bond_len_std:.4f} Å")
                    logging.info(f"  Split Bond Angle mean ± std: {bond_ang_mean:.4f} ± {bond_ang_std:.4f}°")
                    logging.info(f"  Split Torsion Angle mean ± std: {tor_mean:.4f} ± {tor_std:.4f}°")

    # =========================================================================
    # DRUGLIKE / SIMILARITY3D: descriptors, and similarity to a reference
    # (these two used to be one `shepherd` set; they have different inputs)
    # =========================================================================
    want_druglike = args.metrics in ["all", "druglike"]
    reference_path = getattr(args, "reference_mol", None)
    want_similarity = args.metrics == "similarity3d" or (
        args.metrics == "all" and reference_path is not None
    )
    if args.metrics == "similarity3d" and reference_path is None:
        raise ValueError(
            "--metrics similarity3d needs --reference-mol (a .pkl or .sdf "
            "holding the molecule to compare against)"
        )
    if args.metrics == "all" and reference_path is None:
        logging.info("Skipping similarity3d metrics: no --reference-mol given")

    if want_druglike or want_similarity:
        import multiprocessing as mp  # noqa: PLC0415

        from rdkit import Chem  # noqa: PLC0415

        from MolecularDiffusion.runmodes.analyze import druglike as druglike_mod  # noqa: PLC0415
        from MolecularDiffusion.runmodes.analyze import similarity3d as sim_mod  # noqa: PLC0415

        item_timeout = getattr(args, "timeout", 10)

        def _perceive_and_druglike(xyz, with_rmsd, n_conf):
            """Perception + scoring for one molecule; run in a subprocess (see
            below) so a molecule that hangs RDKit can be killed without
            losing the rest of the run."""
            mol = xyz_to_rdkit_mol(xyz)
            if mol is None:
                return None, False, None
            return (Chem.MolToSmiles(mol), True,
                    druglike_mod.compute(mol, with_rdkit_rmsd=with_rmsd, n_conf=n_conf))

        def _run_with_timeout(func, args_, timeout):
            """func(*args_) in a forked subprocess; None (and the subprocess
            killed) if it doesn't finish within timeout. A hard kill is
            needed here -- a hang inside RDKit's C++ code doesn't respond to
            signal.alarm in the same process."""
            ctx = mp.get_context("fork")
            q = ctx.Queue()
            p = ctx.Process(target=lambda: q.put(func(*args_)))
            p.start()
            p.join(timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                return None
            return q.get() if not q.empty() else None

        # if the standard list excluded everything (e.g. only *_opt.xyz), fall back
        _xyzs = xyzs if xyzs else glob.glob(f"{xyz_dir}/*.xyz")
        mol_idx = getattr(args, "mol_idx", 0)
        with_rmsd = getattr(args, "rdkit_rmsd", False)
        n_conf = getattr(args, "rmsd_n_conf", 20)

        data_source = None
        fixed_ref_data = None
        if want_similarity and reference_path:
            try:
                data_source = sim_mod.load_reference_source(reference_path)
            except Exception as e:  # noqa: BLE001
                logging.error(f"Failed to load reference data source: {e}")
                data_source = None
            if data_source is not None and mol_idx != -1:
                try:
                    fixed_ref_data = sim_mod.extract_profiles(
                        sim_mod.reference_mol(data_source, mol_idx)
                    )
                    if fixed_ref_data:
                        logging.info(
                            f"Loaded fixed reference mol [{mol_idx}] from {reference_path}: "
                            f"{fixed_ref_data['num_atoms']} atoms, "
                            f"{len(fixed_ref_data['pharm_types'])} pharmacophores"
                        )
                except Exception as e:  # noqa: BLE001
                    logging.error(f"Failed to load fixed reference molecule: {e}")

        druglike_rows, similarity_rows = [], []
        desc = "Computing druglike + similarity3d" if (want_druglike and want_similarity) else (
            "Computing druglike metrics" if want_druglike else "Computing similarity3d metrics"
        )
        for xyz in tqdm(_xyzs, desc=desc):
            name = os.path.basename(xyz)

            if want_druglike and not want_similarity:
                # common case: isolate each molecule with a hard timeout
                # (druglike scoring has hung on a handful-in-thousands
                # pathological structures; skip that one, keep the run going)
                result = _run_with_timeout(
                    _perceive_and_druglike, (xyz, with_rmsd, n_conf), item_timeout,
                )
                if result is None:
                    logging.warning(
                        f"{name}: druglike computation exceeded {item_timeout}s, "
                        "skipping (treated as invalid)"
                    )
                    smiles, mol_ok, dl_row = None, False, None
                else:
                    smiles, mol_ok, dl_row = result
                row = {"file": name, "valid_rdkit": mol_ok, "smiles": smiles}
                if dl_row:
                    row.update(dl_row)
                druglike_rows.append(row)
                continue

            mol = xyz_to_rdkit_mol(xyz)
            smiles = Chem.MolToSmiles(mol) if mol else None

            if want_druglike:
                row = {"file": name, "valid_rdkit": mol is not None, "smiles": smiles}
                if mol:
                    row.update(druglike_mod.compute(
                        mol, with_rdkit_rmsd=with_rmsd, n_conf=n_conf,
                    ))
                druglike_rows.append(row)

            if want_similarity:
                row = {"file": name, "valid_rdkit": mol is not None, "smiles": smiles,
                       "shape_sim": 0.0, "pharm_sim": 0.0, "esp_sim": 0.0}
                if mol:
                    ref_data = fixed_ref_data
                    if mol_idx == -1 and data_source is not None:
                        try:
                            curr_idx = random.randrange(len(data_source))
                            ref_data = sim_mod.extract_profiles(
                                sim_mod.reference_mol(data_source, curr_idx)
                            )
                            logging.info(f"Randomly selected ref mol [{curr_idx}] for {name}")
                        except Exception as e:  # noqa: BLE001
                            logging.warning(f"Failed to load random reference for {name}: {e}")
                            ref_data = None
                    if ref_data:
                        try:
                            row.update(sim_mod.compare(mol, ref_data, xyz_path=xyz))
                        except Exception as e:  # noqa: BLE001
                            logging.warning(f"Failed to compute similarity for {name}: {e}")
                similarity_rows.append(row)

        def _emit(rows, name, filename):
            """Write one of the two tables and log its summary."""
            df_out = pd.DataFrame(rows)
            if args.output is None:
                out_path = _default_metric_path(metrics_input, filename)
            else:
                base, ext = os.path.splitext(args.output)
                out_path = f"{base}_{name}{ext}"
            df_out = _add_db_row_ids(df_out, metrics_input)
            df_out.to_csv(out_path, index=False)
            result_tables.append((name, df_out, out_path))
            logging.info(f"{name} metrics saved to {out_path}")
            return df_out, out_path

        if want_druglike:
            df_druglike, druglike_path = _emit(druglike_rows, "druglike", "druglike_metrics.csv")
            summary = {"metrics": "druglike", "input": xyz_dir, "n_files": len(_xyzs)}
            for col in ("valid_rdkit", "QED", "SA_score", "LogP", "fsp3", "MW", "HBD", "HBA",
                        "lipinski", "pains_pass", "ring_filter_pass",
                        "n_rings", "n_aromatic_rings", "n_aliphatic_rings"):
                if col in df_druglike.columns:
                    value = float(df_druglike[col].mean())
                    summary[f"{col}_mean"] = value
                    logging.info(f"Average {col}: {value:.4f}")
            for size in druglike_mod.RING_SIZES:
                col = f"ring_size_{size}"
                if col in df_druglike.columns:
                    ratio = float(df_druglike[col].mean())
                    summary[col] = ratio
                    logging.info(f"Ring size {size} ratio: {ratio:.3f}")
            for col in ("SCScore", "rdkit_rmsd_min", "rdkit_rmsd_median", "rdkit_rmsd_max"):
                if col in df_druglike.columns and df_druglike[col].notna().any():
                    value = float(df_druglike[col].mean())
                    summary[f"{col}_mean"] = value
                    logging.info(f"Average {col}: {value:.4f}")
            if split > 1 and len(df_druglike) > 0:
                for col in ("valid_rdkit", "SA_score", "QED"):
                    scale = 100.0 if col == "valid_rdkit" else 1.0
                    mean, std = _get_split_stats(df_druglike[col].values, split, scale=scale)
                    summary[f"split_{col}"] = [mean, std]
                    logging.info(f"Split {col} mean ± std: {mean:.4f} ± {std:.4f}")
            _write_summary(druglike_path, summary)

        if want_similarity:
            df_sim, sim_path = _emit(similarity_rows, "similarity3d", "similarity3d_metrics.csv")
            summary = {
                "metrics": "similarity3d", "input": xyz_dir,
                "reference_mol": str(reference_path), "mol_idx": mol_idx,
                "n_files": len(_xyzs),
            }
            for col in ("shape_sim", "esp_sim", "pharm_sim"):
                if col in df_sim.columns:
                    value = float(df_sim[col].mean())
                    summary[f"{col}_mean"] = value
                    logging.info(f"Average {col}: {value:.4f}")
            if split > 1 and len(df_sim) > 0:
                for col in ("shape_sim", "esp_sim", "pharm_sim"):
                    mean, std = _get_split_stats(df_sim[col].values, split, scale=1.0)
                    summary[f"split_{col}"] = [mean, std]
                    logging.info(f"Split {col} mean ± std: {mean:.4f} ± {std:.4f}")
            _write_summary(sim_path, summary)

    # =========================================================================
    # SBDD: AutoDock Vina affinity against the pocket the ligands were made for
    # =========================================================================
    if args.metrics == "sbdd":
        from MolecularDiffusion.runmodes.analyze import docking  # noqa: PLC0415

        receptor = getattr(args, "receptor", None)
        if not receptor:
            raise ValueError(
                "--metrics sbdd needs --receptor (a .pdbqt, or a .pdb that "
                "meeko can prepare)"
            )
        mode = getattr(args, "dock_mode", "dock")
        exhaustiveness = getattr(args, "exhaustiveness", 8)
        ref_ligand = getattr(args, "ref_ligand", None)

        receptor_pdbqt = docking.prepare_receptor(receptor)
        logging.info(f"Receptor: {receptor_pdbqt}")

        ref_scores = None
        if ref_ligand:
            ref_scores = docking.score_reference(
                ref_ligand, receptor_pdbqt, mode=mode,
                exhaustiveness=exhaustiveness,
            )
            logging.info(f"Reference ligand {os.path.basename(ref_ligand)}: " + ", ".join(
                f"{k}={v:.3f}" for k, v in ref_scores.items()))

        rows = []
        for xyz in tqdm(xyzs, desc="Docking", total=len(xyzs)):
            row = {"file": xyz}
            smiles, mol = docking.load_pose(xyz)
            row["smiles"] = smiles
            if mol is None:
                row["error"] = "perception failed or no 3D pose"
                rows.append(row)
                continue
            try:
                row.update(docking.score_pose(
                    mol, receptor_pdbqt, mode=mode,
                    exhaustiveness=exhaustiveness,
                ))
                row.update({
                    k: v for k, v in compute_drug_likeness(mol).items()
                    if k in ("QED", "SA_score")
                })
            except Exception as e:  # noqa: BLE001 -- a failed pose is data
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)

        df_sbdd = pd.DataFrame(rows)
        df_sbdd = _add_db_row_ids(df_sbdd, metrics_input)

        if args.output is None:
            sbdd_output_path = _default_metric_path(metrics_input, "sbdd_metrics.csv")
        else:
            base, ext = os.path.splitext(args.output)
            sbdd_output_path = f"{base}_sbdd{ext}"
        df_sbdd.to_csv(sbdd_output_path, index=False)
        result_tables.append(("sbdd", df_sbdd, sbdd_output_path))
        logging.info(f"SBDD metrics saved to {sbdd_output_path}")

        summary = {
            "metrics": "sbdd",
            "input": xyz_dir,
            "receptor": receptor_pdbqt,
            "dock_mode": mode,
            "exhaustiveness": exhaustiveness,
            "n_files": len(xyzs),
            "n_scored": int(df_sbdd["vina_score"].notna().sum()) if "vina_score" in df_sbdd else 0,
        }
        score_cols = [c for c in ("vina_score", "vina_min", "vina_dock") if c in df_sbdd]
        logging.info(f"Scored {summary['n_scored']}/{len(xyzs)} structures")
        for col in score_cols:
            values = df_sbdd[col].dropna()
            if values.empty:
                continue
            summary[f"{col}_mean"] = float(values.mean())
            summary[f"{col}_median"] = float(values.median())
            summary[f"{col}_best"] = float(values.min())
            logging.info(
                f"{col}: mean {values.mean():.3f}  median {values.median():.3f}  "
                f"best {values.min():.3f}"
            )

        # the affinity column the literature aggregates on
        best_col = score_cols[-1] if score_cols else None
        if best_col and ref_scores:
            reference = ref_scores.get(best_col)
            summary["reference_affinity"] = reference
            scored = df_sbdd[best_col].dropna()
            if reference is not None and not scored.empty:
                high = float((scored < reference).mean())
                summary["high_affinity"] = high
                logging.info(f"High affinity (beats reference {reference:.3f}): {high * 100:.2f}%")

        if best_col and {"QED", "SA_score"}.issubset(df_sbdd.columns):
            # SA_score is the raw 1-10 RDKit score; the papers threshold the
            # normalised (10 - SA) / 9 form
            sa_norm = (10.0 - df_sbdd["SA_score"]) / 9.0
            success = (
                (df_sbdd["QED"] > 0.25) & (sa_norm > 0.59) & (df_sbdd[best_col] < -8.18)
            )
            summary["success_rate"] = float(success.mean())
            logging.info(
                f"Success rate (QED>0.25, SA>0.59, {best_col}<-8.18): "
                f"{success.mean() * 100:.2f}%"
            )

        _write_summary(sbdd_output_path, summary)

    # =========================================================================
    # CONFORMER: paired generated-vs-reference metrics (exclusive, like sbdd)
    # =========================================================================
    if args.metrics == "conformer":
        from MolecularDiffusion.runmodes.analyze import conformer_metrics  # noqa: PLC0415

        df_conf, summary = conformer_metrics.compute_conformer_metrics(
            args.input,
            rmsd_threshold=getattr(args, "rmsd_threshold", 0.5),
            charge=getattr(args, "charge", 0),
            level=getattr(args, "level", "gfn2"),
            timeout=getattr(args, "xtb_timeout", 120),
        )
        df_conf = _add_db_row_ids(df_conf, metrics_input)

        if args.output is None:
            conf_output_path = _default_metric_path(
                metrics_input, "conformer_metrics.csv"
            )
        else:
            base, ext = os.path.splitext(args.output)
            conf_output_path = f"{base}_conformer{ext}"
        df_conf.to_csv(conf_output_path, index=False)
        result_tables.append(("conformer", df_conf, conf_output_path))
        logging.info(f"Conformer metrics saved to {conf_output_path}")

        for key in (
            "rs_score", "ez_score", "rmsd_median", "rmsd_best_per_mol_mean",
            "coverage_at_threshold", "bond_length_mean_mean",
            "bond_angle_mean_mean", "torsion_angle_mean_mean",
            "mmff_strain_kcal_mean", "xtb_strain_kcal_mean",
        ):
            if summary.get(key) is not None:
                logging.info(f"{key}: {summary[key]:.4f}")

        if split > 1:
            for col in ("rmsd", "mmff_strain_kcal"):
                values = df_conf[col].dropna().to_numpy() if col in df_conf else []
                if len(values):
                    mean, std = _get_split_stats(values, split, scale=1.0)
                    summary[f"split_{col}"] = [mean, std]
                    logging.info(
                        f"Split {col} mean \u00b1 std: {mean:.4f} \u00b1 {std:.4f}"
                    )
        _write_summary(conf_output_path, summary)

    try:
        _apply_result_filter(
            result_tables,
            metrics_input,
            getattr(args, "filter", None),
            getattr(args, "filtered_output", None),
        )
    finally:
        metrics_input.cleanup()
