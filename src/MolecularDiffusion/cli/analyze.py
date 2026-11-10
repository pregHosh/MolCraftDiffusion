"""Analyze CLI subcommands for 3D molecule analysis.

Provides subcommands for:
- optimize: xTB geometry optimization
- metrics: Validity/connectivity metrics (incl. the paired `conformer` set)
- xyz2mol: XYZ to SMILES conversion + fingerprints
- featurize: fixed-size feature vectors for XYZ molecules
- xtb-electronic: xTB electronic properties
"""

import os

import click
from MolecularDiffusion.optional import optional_import_error

# Enable -h as alias for --help
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def _load_analyze_module(module_name: str):
    try:
        return __import__(
            f"MolecularDiffusion.runmodes.analyze.{module_name}",
            fromlist=[module_name],
        )
    except ImportError as exc:
        raise click.ClickException(str(optional_import_error("analyze", exc))) from exc


@click.group(context_settings=CONTEXT_SETTINGS)
def analyze():
    """Analyze 3D molecular structures.

    \b
    Subcommands:
      optimize        xTB geometry optimization
      metrics         Validity/connectivity metrics
      dock            Dock a library into one fixed pocket; pose RMSD
                      and actives-vs-decoys enrichment
      bias            Check whether a docking score tracks molecule size,
                      logP or charge (top picks vs the pool)
      xyz2mol         Convert XYZ to SMILES + fingerprints
      featurize       Fixed-size feature vectors for XYZ molecules
      xtb-electronic  xTB electronic properties
    """
    pass


# ============================================================================
# OPTIMIZE: XTB geometry optimization
# ============================================================================

@analyze.command("optimize", context_settings=CONTEXT_SETTINGS)
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output-path", "-o", "--o", default=None, type=click.Path(),
              help="Output directory (for XYZ) or file (for ASE DB)")
@click.option("--charge", "-c", "--c", default=0, type=int,
              help="Molecular charge for xTB (default: 0)")
@click.option("--level", "-l", "--l", default="gfn1", type=click.Choice(["gfn1", "gfn2", "gfn-ff", "mmff94"]),
              help="Optimization level (default: gfn1)")
@click.option("--timeout", "-t", "--t", default=240, type=int,
              help="Timeout per molecule in seconds (default: 240)")
@click.option("--scale-factor", "-s", "--s", default=1.3, type=float,
              help="Scale factor for covalent radii (default: 1.3)")
@click.option("--csv", "csv_path", default=None, type=click.Path(),
              help="CSV file to filter which files to optimize")
@click.option("--filter-column", default=None, type=str,
              help="Column name in CSV to filter by (values must be 1)")
@click.option("--inherit-attributes", "--inherit", is_flag=True, default=True,
              help="Inherit all attributes from the original ASE DB row (default: True)")
def optimize(input_path, output_path, charge, level, timeout, scale_factor, csv_path, filter_column, inherit_attributes):
    """Optimize molecular geometries (XYZ or ASE DB).
    
    If input is a directory, it processes all XYZ files.
    If input is a .db file, it processes all rows in the ASE database.
    
    \b
    Examples:
        MolCraftDiff analyze optimize gen_xyz/
        MolCraftDiff analyze optimize gen_xyz/ --o optimized/ --level gfn2
    """
    get_xtb_optimized_xyz = _load_analyze_module("xtb_optimization").get_xtb_optimized_xyz
    
    output_dir = output_path or os.path.join(input_path, "optimized_xyz")
    
    click.echo(f"Optimizing XYZ files from: {input_path}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"xTB level: {level}, charge: {charge}")
    
    optimized_files = get_xtb_optimized_xyz(
        input_directory=input_path,
        output_directory=output_dir,
        charge=charge,
        level=level,
        timeout=timeout,
        scale_factor=scale_factor,
        csv_path=csv_path,
        filter_column=filter_column,
    )
    
    click.echo(f"\nSuccessfully optimized {len(optimized_files)} files.")


# ============================================================================
# METRICS: Validity/connectivity metrics
# ============================================================================

@analyze.command("metrics", context_settings=CONTEXT_SETTINGS)
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", "-o", "--o", "--output-csv", default=None, type=click.Path(),
              help="Output CSV file for results")
@click.option("--filter", "filter_column", default=None, type=str,
              help="Keep only structures whose column is truthy (any metrics column)")
@click.option("--filtered-output", default=None, type=click.Path(),
              help="Output ASE DB path or XYZ directory for filtered structures")
@click.option("--metrics", "-m", "--m", "metrics_type", default="all",
              type=click.Choice(["all", "core", "posebuster", "geom_revised",
                                 "druglike", "similarity3d", "sbdd",
                                 "conformer"]),
              help="Which metrics to compute (default: all)")
@click.option("--recheck-topo", is_flag=True, default=False,
              help="Recheck topology using RDKit")
@click.option("--check-neutrality", is_flag=True, default=False,
              help="Check electron/spin neutrality with xTB (one call per molecule; slow)")
@click.option("--portion", "-p", "--p", default=1.0, type=float,
              help="Portion of XYZ files to process (default: 1.0 = all)")
@click.option("--mol-converter", default="xyz2mol",
              type=click.Choice(["xyz2mol", "openbabel"]),
              help="XYZ to mol converter (default: xyz2mol)")
@click.option("--skip-atoms", multiple=True, type=int,
              help="Atom indices to skip in validation")
@click.option("--split", "-s", default=1, type=click.IntRange(min=1),
              help="Number of deterministic splits for summary mean±std logging (default: 1, legacy behavior)")
@click.option("--timeout", "-t", "--t", default=10, type=int,
              help="Timeout per xyz2mol conversion in seconds (default: 10)")
@click.option("--reference-mol", "-r", default=None, type=click.Path(),
              help="Reference .pkl or .sdf to compare against (required by --metrics similarity3d)")
@click.option("--mol-idx", default=0, type=int,
              help="Molecule index in reference .pkl (default: 0)")
@click.option("--train-smiles", default=None, type=click.Path(exists=True),
              help="Reference SMILES (.txt one per line, or .csv) to score novelty against")
@click.option("--receptor", default=None, type=click.Path(exists=True),
              help="Protein receptor (.pdbqt, or .pdb to prepare) -- required by --metrics sbdd")
@click.option("--ref-ligand", default=None, type=click.Path(exists=True),
              help="Reference ligand .sdf; its own affinity becomes the bar for high_affinity")
@click.option("--dock-mode", default="dock", type=click.Choice(["score", "min", "dock"]),
              help="sbdd cost/fidelity: score (in place), min (+local opt), dock (full redock, default)")
@click.option("--exhaustiveness", default=8, type=int,
              help="Vina search exhaustiveness for --dock-mode dock (default: 8)")
@click.option("--rdkit-rmsd", is_flag=True, default=False,
              help="druglike: RMSD of the pose vs UFF-optimised RDKit conformers (slow)")
@click.option("--rmsd-n-conf", default=20, type=int,
              help="Conformers embedded per molecule for --rdkit-rmsd (default: 20)")
@click.option("--rmsd-threshold", default=0.5, type=float,
              help="conformer: RMSD cutoff (A) for the coverage metric (default: 0.5)")
@click.option("--charge", "-c", "--c", default=0, type=int,
              help="conformer: molecular charge for xTB strain (default: 0)")
@click.option("--level", "-l", "--l", default="gfn2",
              type=click.Choice(["gfn1", "gfn2", "gfn-ff"]),
              help="conformer: xTB level for strain (default: gfn2)")
@click.option("--xtb-timeout", default=120, type=int,
              help="conformer: timeout per xTB call in seconds (default: 120)")
def metrics(input_path, output, filter_column, filtered_output, metrics_type, recheck_topo, check_neutrality, portion, mol_converter, skip_atoms, split, timeout, reference_mol, mol_idx, train_smiles, receptor, ref_ligand, dock_mode, exhaustiveness, rdkit_rmsd, rmsd_n_conf, rmsd_threshold, charge, level, xtb_timeout):
    """Compute validity and connectivity metrics for XYZ files or ASE DB rows.

    \b
    Metrics types:
      all          Run all metrics (core + posebuster + geom_revised + shepherd)
      core         Validity (RDKit + geometry), connectivity, atom stability,
                   uniqueness/novelty/diversity
      posebuster   PoseBusters checks (bond lengths, angles, clashes)
      geom_revised Aromatic-aware stability metrics
      druglike     Drug-likeness descriptors: QED, SA, LogP, Lipinski, PAINS,
                   ring statistics (+ --rdkit-rmsd)
      similarity3d Shape / ESP / pharmacophore similarity to --reference-mol
      sbdd         AutoDock Vina affinity against a protein pocket
                   (needs --receptor and the [sbdd] extra)
      conformer    Paired generated-vs-reference conformer metrics: stereo
                   preservation (R/S, E/Z), RMSD / coverage, bond-length /
                   angle / torsion deviation, MMFF (and xTB) strain.
                   Exclusive -- not part of 'all'; needs paired input, either
                   conformers.csv + mol_XXXX/*.sdf (conformer generation) or
                   *.xyz + optimized_xyz/<stem>_opt.xyz (analyze optimize).

    \b
    Examples:
        MolCraftDiff analyze metrics gen_xyz/
        MolCraftDiff analyze metrics molecules.db --metrics core
        MolCraftDiff analyze metrics gen_xyz/ --metrics core --filter valid_connected --filtered-output kept/
        MolCraftDiff analyze metrics gen_xyz/ --metrics posebuster
        MolCraftDiff analyze metrics gen_xyz/ --metrics geom_revised --mol-converter openbabel
        MolCraftDiff analyze metrics gen_xyz/ --metrics druglike
        MolCraftDiff analyze metrics gen_xyz/ --split 4
        MolCraftDiff analyze metrics gen_xyz/ --metrics similarity3d -r reference.sdf --mol-idx 0
        MolCraftDiff analyze metrics gen_xyz/ --metrics sbdd --receptor pocket.pdbqt --ref-ligand ref.sdf
        MolCraftDiff analyze metrics generated_conformers/ --metrics conformer
        MolCraftDiff analyze metrics gen_xyz/ --metrics conformer   # after 'analyze optimize'
    """
    import argparse
    runner = _load_analyze_module("compute_metrics").runner
    
    args = argparse.Namespace(
        input=input_path,
        output=output,
        filter=filter_column,
        filtered_output=filtered_output,
        metrics=metrics_type,
        recheck_topo=recheck_topo,
        check_neutrality=check_neutrality,
        portion=portion,
        mol_converter=mol_converter,
        skip_atoms=list(skip_atoms) if skip_atoms else None,
        split=split,
        timeout=timeout,
        reference_mol=reference_mol,
        mol_idx=mol_idx,
        train_smiles=train_smiles,
        receptor=receptor,
        ref_ligand=ref_ligand,
        dock_mode=dock_mode,
        exhaustiveness=exhaustiveness,
        rdkit_rmsd=rdkit_rmsd,
        rmsd_n_conf=rmsd_n_conf,
        rmsd_threshold=rmsd_threshold,
        charge=charge,
        level=level,
        xtb_timeout=xtb_timeout,
    )
    
    click.echo(f"Computing {metrics_type} metrics for: {input_path}")
    try:
        runner(args)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


# ============================================================================
# DOCK: dock an arbitrary library into ONE fixed pocket
# ============================================================================

@analyze.command("dock", context_settings=CONTEXT_SETTINGS)
@click.option("--receptor", required=True, type=click.Path(exists=True),
              help="Receptor .pdbqt, or a .pdb to prepare via meeko")
@click.option("--ligands", multiple=True, type=click.Path(exists=True),
              help="Unlabelled library (.sdf/.smi/.txt/.smiles/.ism/.csv)")
@click.option("--actives", multiple=True, type=click.Path(exists=True),
              help="Known actives -> label 1 (enables enrichment metrics)")
@click.option("--decoys", multiple=True, type=click.Path(exists=True),
              help="Known inactives -> label 0 (enables enrichment metrics)")
@click.option("--box-ref-ligand", default=None, type=click.Path(exists=True),
              help="Crystal-ligand SDF defining the pocket box (recommended)")
@click.option("--box-center", nargs=3, type=float, default=None,
              help="Box centre 'X Y Z', as an alternative to a ref ligand")
@click.option("--box-size", nargs=3, type=float, default=None,
              help="Box size 'X Y Z' in angstrom")
@click.option("--mode", default="dock",
              type=click.Choice(["score", "min", "dock"]),
              help="Cost/fidelity: score (in place), min, dock (default)")
@click.option("--exhaustiveness", default=8, type=int,
              help="Vina search exhaustiveness (default: 8)")
@click.option("--n-poses", default=1, type=int,
              help="Poses to keep per molecule; >1 forces a full dock")
@click.option("--rmsd-ref", default=None, type=click.Path(exists=True),
              help="Crystal pose SDF to measure pose recovery against")
@click.option("--save-poses", default=None, type=click.Path(),
              help="Write docked poses to this SDF (tagged rank/score/RMSD)")
@click.option("--seed", default=42, type=int, help="Vina seed (default: 42)")
@click.option("--allow-bad-res", is_flag=True,
              help="Let meeko accept incomplete residues in the receptor")
@click.option("--default-altloc", default=None,
              help="Altloc to keep, e.g. A, when the PDB has alternates")
@click.option("--out-dir", default=None, type=click.Path(),
              help="Where to write the prepared receptor .pdbqt")
@click.option("--output", "-o", "--o", default=None, type=click.Path(),
              help="Output CSV (default: dock_library.csv)")
def dock(receptor, ligands, actives, decoys, box_ref_ligand, box_center,
         box_size, mode, exhaustiveness, n_poses, rmsd_ref, save_poses, seed,
         allow_bad_res, default_altloc, out_dir, output):
    """Dock a library of molecules into one fixed pocket and rank them.

    \b
    Unlike `metrics --metrics sbdd`, which scores generated .xyz files each
    in its own box, this scores an arbitrary library (SDF, or SMILES
    embedded on the fly) inside a single shared pocket box -- what a
    retrospective benchmark needs. Needs the [sbdd] extra.

    \b
    Pose recovery (is the top-scored pose the crystal pose?):
      MolCraftDiff analyze dock --receptor receptor.pdb \\
        --ligands ligand.sdf --rmsd-ref ligand.sdf --n-poses 5
      -> rmsd_top, rmsd_best, rmsd_best_rank; under 2 A is the usual bar

    \b
    Enrichment (do known actives rank above decoys?):
      MolCraftDiff analyze dock --receptor receptor.pdb \\
        --box-ref-ligand crystal.sdf \\
        --actives actives.smi --decoys decoys.smi
      -> roc_auc, ef_1pct, ef_5pct

    \b
    Without --box-ref-ligand or --box-center each molecule is boxed on its
    OWN coordinates, which is only correct for molecules already posed in
    the pocket frame. A warning is printed in that case.
    """
    import argparse
    runner = _load_analyze_module("dock_library").runner

    args = argparse.Namespace(
        receptor=receptor,
        ligands=list(ligands) if ligands else None,
        actives=list(actives) if actives else None,
        decoys=list(decoys) if decoys else None,
        box_ref_ligand=box_ref_ligand,
        box_center=list(box_center) if box_center else None,
        box_size=list(box_size) if box_size else None,
        mode=mode,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        rmsd_ref=rmsd_ref,
        save_poses=save_poses,
        seed=seed,
        allow_bad_res=allow_bad_res,
        default_altloc=default_altloc,
        out_dir=out_dir,
        output=output,
    )

    click.echo(f"Docking into {receptor}")
    try:
        runner(args)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


# ============================================================================
# BIAS: is the docking score just measuring size?
# ============================================================================

@analyze.command("bias", context_settings=CONTEXT_SETTINGS)
@click.argument("inputs", nargs=-1, required=True,
                type=click.Path(exists=True))
@click.option("--labels", default=None, type=click.Path(exists=True),
              help="CSV of known hits; adds a hits-vs-pool section")
@click.option("--id-col", default="name",
              help="--labels column holding the molecule name (default: name)")
@click.option("--label-col", default="label",
              help="--labels column, > 0 means hit (default: label)")
@click.option("--score-col", default="vina_dock",
              help="Score column, lower is better (default: vina_dock)")
@click.option("--top", multiple=True, type=click.IntRange(min=1),
              help="Top-k to test; repeatable (default: 10 and 1% of pool)")
@click.option("--sd-flag", default=0.5, type=float,
              help="Warn when the top-k mean is this many pool SDs above "
                   "the pool mean (default: 0.5)")
@click.option("--out-dir", default=None, type=click.Path(),
              help="Write property_bias_*.csv here (none written otherwise)")
def bias(inputs, labels, id_col, label_col, score_col, top, sd_flag, out_dir):
    """Check whether a docking score is really measuring molecule size.

    \b
    Reads the CSVs written by `analyze dock` (columns name, smiles and the
    score). Reports how the score correlates with heavy atoms, logP and
    formal charge, and whether the top-ranked molecules differ from the whole
    pool. It is a diagnostic: it warns, and never exits non-zero.

    \b
    Examples:
      MolCraftDiff analyze bias dock_results.csv
      MolCraftDiff analyze bias out/full_0.csv out/full_1.csv --top 10 --top 174
      MolCraftDiff analyze bias out.csv --labels master.csv \\
        --id-col cache_id --label-col hit29 --out-dir bias/
    """
    import argparse
    runner = _load_analyze_module("property_bias").runner

    args = argparse.Namespace(
        inputs=list(inputs),
        labels=labels,
        id_col=id_col,
        label_col=label_col,
        score_col=score_col,
        top=list(top) if top else None,
        sd_flag=sd_flag,
        out_dir=out_dir,
    )
    try:
        runner(args)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


# ============================================================================
# XYZ2MOL: Convert XYZ to SMILES + fingerprints
# ============================================================================

@analyze.command("xyz2mol", context_settings=CONTEXT_SETTINGS)
@click.argument("xyz_dir", type=click.Path(exists=True))
@click.option("--input-csv", "-i", "--i", default=None, type=click.Path(),
              help="Optional input CSV with xyz file list")
@click.option("--label", "-l", "--l", default=None, type=str,
              help="Label for processed files")
@click.option("--timeout", "-t", "--t", default=30, type=int,
              help="Timeout per conversion in seconds (default: 30)")
@click.option("--bits", "-b", "--b", default=2048, type=int,
              help="Number of bits for Morgan fingerprint (default: 2048)")
@click.option("--verbose", "-v", "--v", is_flag=True,
              help="Enable verbose output")
def xyz2mol(xyz_dir, input_csv, label, timeout, bits, verbose):
    """Convert XYZ files to SMILES and extract fingerprints/scaffolds.
    
    Outputs are saved to xyz_dir/2d_reprs/:
      - smiles_processed.csv
      - fingerprints.npy
      - scaffolds.txt
      - substructures.json
    
    \b
    Examples:
        MolCraftDiff analyze xyz2mol gen_xyz/
        MolCraftDiff analyze xyz2mol gen_xyz/ --bits 1024 -v
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import json
    import logging
    
    xyz2mol_module = _load_analyze_module("xyz2mol")
    load_file_list_from_dir = xyz2mol_module.load_file_list_from_dir
    run_processing = xyz2mol_module.run_processing
    extract_scaffold_and_fingerprints = xyz2mol_module.extract_scaffold_and_fingerprints
    
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    xyz_dir = Path(xyz_dir)
    two_d_reprs_dir = xyz_dir / "2d_reprs"
    two_d_reprs_dir.mkdir(parents=True, exist_ok=True)
    
    smiles_csv_output = two_d_reprs_dir / "smiles_processed.csv"
    
    click.echo(f"Processing XYZ files from: {xyz_dir}")
    click.echo(f"Output directory: {two_d_reprs_dir}")
    
    # Load file list
    if input_csv:
        df = pd.read_csv(input_csv)
    else:
        df = load_file_list_from_dir(str(xyz_dir))
    
    # Generate SMILES
    df_smiles = run_processing(df, str(xyz_dir), label, smiles_csv_output, timeout=timeout, verbose=verbose)
    
    if df_smiles is None or 'smiles' not in df_smiles.columns or df_smiles['smiles'].isnull().all():
        click.echo("No valid SMILES generated.", err=True)
        return
    
    # Extract fingerprints and scaffolds
    click.echo("\nExtracting fingerprints and scaffolds...")
    fps, scaffolds, clean_smiles, n_fail, substruct_counts = \
        extract_scaffold_and_fingerprints(df_smiles["smiles"].dropna().values, fp_bits=bits)
    
    np.save(two_d_reprs_dir / "fingerprints.npy", fps)
    with open(two_d_reprs_dir / "scaffolds.txt", "w") as f:
        f.write("\n".join(scaffolds))
    with open(two_d_reprs_dir / "smiles_cleaned.txt", "w") as f:
        f.write("\n".join(clean_smiles))
    with open(two_d_reprs_dir / "substructures.json", "w") as f:
        json.dump(substruct_counts, f, indent=2)
    
    total = len(df_smiles["smiles"].dropna())
    click.echo(f"\n--- Summary ---")
    click.echo(f"Total SMILES: {total}")
    click.echo(f"Failed FP extraction: {n_fail}")
    click.echo(f"Unique substructures: {len(substruct_counts)}")
    click.echo(f"Outputs saved to: {two_d_reprs_dir}")


# ============================================================================
# FEATURIZE: Fixed-size molecular feature vectors
# ============================================================================

@analyze.command("featurize", context_settings=CONTEXT_SETTINGS)
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--backend", "-b", default="soap",
              type=click.Choice(["soap", "uma", "ssl3d"]),
              help="Featurization backend (default: soap)")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output stem (default: input_dir/features). .npy/.csv/_meta.json appended.")
@click.option("--recursive", "-r", is_flag=True, default=False,
              help="Search subdirectories for structure files")
# --- SOAP options ---
@click.option("--r-cut", default=6.0, type=float, show_default=True,
              help="SOAP cutoff radius in Å")
@click.option("--n-max", default=8, type=int, show_default=True,
              help="SOAP radial basis functions")
@click.option("--l-max", default=6, type=int, show_default=True,
              help="SOAP angular basis functions")
@click.option("--sigma", default=0.1, type=float, show_default=True,
              help="SOAP Gaussian smearing width")
@click.option("--autodetect", "autodetect_species", is_flag=True, default=False,
              help="SOAP: auto-detect element species from files (overrides --species)")
@click.option("--species", multiple=True, type=str,
              help="SOAP: element symbols to use (default: H B C N O F Al Si P S Cl As Se Br I Hg Bi)")
@click.option("--pooling", default="mean", type=click.Choice(["mean", "sum"]), show_default=True,
              help="Atom pooling mode")
@click.option("--soap-jobs", default=1, type=int, show_default=True,
              help="Parallel workers for SOAP")
# --- UMA options ---
@click.option("--checkpoint", default="training_outputs/uma-s-1p2.pt", type=click.Path(),
              show_default=True, help="Path to UMA checkpoint (.pt)")
@click.option("--task-name", default="omol", type=str, show_default=True,
              help="UMA task name")
@click.option("--device", default=None, type=click.Choice(["cuda", "cpu"]),
              help="Device for UMA inference (default: auto)")
@click.option("--batch-size", default=8, type=int, show_default=True,
              help="Molecules per UMA batch")
@click.option("--all-components", is_flag=True, default=False,
              help="UMA: use all spherical components instead of L=0 scalars only")
@click.option("--charge", default=0, type=int, show_default=True,
              help="UMA: total molecular charge (default: 0)")
@click.option("--spin", default=1, type=int, show_default=True,
              help="UMA: spin multiplicity (default: 1)")
# --- SSL3D options ---
@click.option("--ssl3d-checkpoint", default=None, type=click.Path(),
              help="SSL3D: path to trained .ckpt or .pkl checkpoint (required for --backend ssl3d)")
@click.option("--edge-radius", default=5.0, type=float, show_default=True,
              help="SSL3D: radius graph cutoff in Å for graph construction")
def featurize(input_dir, backend, output, recursive,
              r_cut, n_max, l_max, sigma, autodetect_species, species, pooling, soap_jobs,
              checkpoint, task_name, device, batch_size, all_components, charge, spin,
              ssl3d_checkpoint, edge_radius):
    """Featurize 3D XYZ molecules into fixed-size feature vectors.

    \b
    Backends:
      soap   SOAP descriptor via dscribe — no GPU required
      uma    UMA backbone embeddings — requires vendored fairchem/src + checkpoint
      ssl3d  SSL3D backbone embeddings — requires a trained SSL3D .ckpt or .pkl

    \b
    Examples:
        MolCraftDiff analyze featurize gen_xyz/
        MolCraftDiff analyze featurize gen_xyz/ --autodetect
        MolCraftDiff analyze featurize gen_xyz/ --species C --species H --species N --species O
        MolCraftDiff analyze featurize gen_xyz/ --backend soap --n-max 12 --l-max 9
        MolCraftDiff analyze featurize gen_xyz/ --backend uma --device cuda
        MolCraftDiff analyze featurize gen_xyz/ --backend ssl3d --ssl3d-checkpoint runs/last.ckpt
        MolCraftDiff analyze featurize gen_xyz/ --backend ssl3d --ssl3d-checkpoint runs/last.ckpt --device cuda
    """
    if backend == "ssl3d" and ssl3d_checkpoint is None:
        raise click.UsageError("--ssl3d-checkpoint is required when --backend ssl3d")

    run_featurize = _load_analyze_module("featurize").run_featurize

    run_featurize(
        input_dir=input_dir,
        backend=backend,
        output_path=output,
        recursive=recursive,
        # SOAP
        autodetect_species=autodetect_species,
        species=list(species) if species else None,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        pooling=pooling,
        n_jobs=soap_jobs,
        # UMA
        checkpoint=checkpoint,
        task_name=task_name,
        device=device,
        batch_size=batch_size,
        scalar_only=not all_components,
        charge=charge,
        spin=spin,
        # SSL3D
        ssl3d_checkpoint=ssl3d_checkpoint,
        ssl3d_edge_radius=edge_radius,
    )


# ============================================================================
# XTB-ELECTRONIC: Compute XTB electronic properties
# ============================================================================

@analyze.command("xtb-electronic", context_settings=CONTEXT_SETTINGS)
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", "--o", "-o", default=None, type=click.Path(),
              help="Output file path (without extension for 'all' format)")
@click.option("--method", "--m", "-m", default="2", type=click.Choice(["1", "2", "ptb"]),
              help="XTB method: 1=GFN1, 2=GFN2, ptb=PTB (default: 2)")
@click.option("--charge", "--c", "-c", default=0, type=int,
              help="Molecular charge (default: 0)")
@click.option("--n-unpaired", "--unpaired", default=0, type=int,
              help="Number of unpaired electrons (default: 0)")
@click.option("--auto-charge", is_flag=True, default=False,
              help="For PTB neutral singlets with odd electron count, infer +1/-1 charge from XYZ chemistry")
@click.option("--solvent", "--s", "-s", default=None, type=str,
              help="Solvent for solvation calculations (e.g., 'water', 'thf', 'chcl3')")
@click.option("--properties", "--prop", "-p", multiple=True, 
              type=click.Choice(["energy", "dipole", "reactivity", "global", 
                                  "charges", "fukui", "bond_orders", "all"]),
              help="Property groups to compute (default: energy)")
@click.option("--corrected/--no-corrected", default=True,
              help="Apply empirical IP/EA correction (default: True)")
@click.option("--timeout", "--t", "-t", default=120, type=int,
              help="Timeout per molecule in seconds (default: 120)")
@click.option("--n-jobs", "--jobs", "-j", default=1, type=int,
              help="Number of parallel jobs (default: 1)")
@click.option("--format", "--fmt", "-f", "output_format", default="csv", 
              type=click.Choice(["csv", "json", "ase", "all"]),
              help="Output format: csv, json, ase (.db), or all (default: csv)")
@click.option("--annotate-db", is_flag=True, default=False,
              help="For ASE .db input, annotate the input rows in place with xtb_* results")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Log mean/min/max/std statistics for each computed scalar property")
def xtb_electronic(input_path, output, method, charge, n_unpaired,
                   auto_charge, solvent, properties, corrected, timeout,
                   n_jobs, output_format, annotate_db, verbose):
    """Compute XTB electronic properties for XYZ files or ASE DB rows.
    
    Uses morfeus to calculate quantum-chemical descriptors at the GFN-xTB level.
    
    \b
    Property groups (molecular-level):
      energy      Total energy, HOMO, LUMO, gap, Fermi level
      dipole      Dipole moment and vector
      reactivity  IP, EA, electronegativity, hardness, softness
      global      Electrophilicity, nucleophilicity, fugalities
      solvation   Solvation energy, H-bond correction (requires --solvent)
    
    \b
    Property groups (atomic-level):
      charges     Atomic charges (Mulliken)
      fukui       Fukui indices (f+, f-, f, dual)
      bond_orders Bond orders between atom pairs
    
    \b
    Output formats:
      csv   Molecular-level properties only (one row per molecule)
      json  Full data including atomic-level properties
      ase   ASE database with properties in atoms.info/arrays
      all   Generate all three formats
    
    \b
    Examples:
        MolCraftDiff analyze xtb-electronic gen_xyz/
        MolCraftDiff analyze xtb-electronic molecules.db -p all
        MolCraftDiff analyze xtb-electronic molecules.db -p all --annotate-db
        MolCraftDiff analyze xtb-electronic gen_xyz/ -p energy -p reactivity
        MolCraftDiff analyze xtb-electronic gen_xyz/ -s water -p solvation
        MolCraftDiff analyze xtb-electronic gen_xyz/ --method ptb --auto-charge
        MolCraftDiff analyze xtb-electronic gen_xyz/ -p all -f ase -o results.db
    """
    batch_xtb_electronic = _load_analyze_module("xtb_electronic").batch_xtb_electronic
    
    # Parse method
    if method in ["1", "2"]:
        method = int(method)
    
    # Default properties
    if not properties:
        properties = ["energy"]
    
    # Default output path
    if annotate_db and os.path.splitext(input_path)[1].lower() != ".db":
        raise click.UsageError("--annotate-db requires an ASE .db input file")

    if output is None and not annotate_db:
        if os.path.isfile(input_path):
            root, _ = os.path.splitext(input_path)
            output = f"{root}_xtb_electronic"
        else:
            output = os.path.join(input_path, "xtb_electronic")
    
    click.echo(f"Computing XTB electronic properties for: {input_path}")
    click.echo(f"Method: GFN{method}-xTB" if method != "ptb" else "Method: PTB")
    click.echo(f"Charge: {charge}, Unpaired: {n_unpaired}")
    if auto_charge:
        click.echo("Auto charge: enabled")
    if solvent:
        click.echo(f"Solvent: {solvent}")
    click.echo(f"Properties: {', '.join(properties)}")
    click.echo(f"Output format: {output_format}")
    if annotate_db:
        click.echo("Annotate DB: enabled")
    
    df = batch_xtb_electronic(
        input_dir=input_path,
        output_path=output,
        output_format=output_format,
        method=method,
        charge=charge,
        n_unpaired=n_unpaired,
        solvent=solvent,
        properties=list(properties),
        corrected=corrected,
        timeout=timeout,
        n_jobs=n_jobs,
        auto_charge=auto_charge,
        annotate_db=annotate_db,
    )
    
    import pandas as pd

    n_success = df["success"].sum() if "success" in df.columns else len(df)
    n_total = len(df)

    click.echo(f"\n--- Summary ---")
    click.echo(f"Processed: {n_total} molecules")
    click.echo(f"Successful: {n_success}")
    click.echo(f"Failed: {n_total - n_success}")
    if annotate_db:
        click.echo(f"Annotated DB: {input_path}")
    if output:
        click.echo(f"Output saved to: {output}")

    if verbose and not df.empty:
        _META_COLS = {
            "filename", "n_atoms", "method", "charge", "solvent",
            "success", "error", "ase_db_row_id",
            "requested_charge", "auto_charge_applied",
            "auto_charge_reason", "auto_charge_candidates",
        }
        successful = df[df["success"] == True] if "success" in df.columns else df
        numeric_cols = [
            c for c in df.columns
            if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not successful.empty and numeric_cols:
            click.echo("\n--- Property Statistics (successful molecules) ---")
            for col in numeric_cols:
                vals = successful[col].dropna()
                if len(vals) > 0:
                    click.echo(
                        f"  {col:30s}  mean={vals.mean():.4g}"
                        f"  min={vals.min():.4g}"
                        f"  max={vals.max():.4g}"
                        f"  std={vals.std():.4g}"
                    )
