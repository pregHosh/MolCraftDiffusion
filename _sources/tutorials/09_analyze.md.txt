# Tutorial 9: Analyze Module - 3D Molecular Structure Analysis

This tutorial covers the **analyze** module, which provides tools for post-generation analysis and validation of 3D molecular structures.

---

## Overview

The analyze module includes five subcommands:

| Command | Description |
|---------|-------------|
| `optimize` | XTB geometry optimization |
| `metrics` | Validity/connectivity metrics |
| `compare` | RMSD and energy comparison |
| `xyz2mol` | XYZ to SMILES + fingerprints |
| `xtb-electronic` | XTB electronic properties |

Access the CLI with:
```bash
MolCraftDiff analyze --help
```

---

## Part 1: XTB Geometry Optimization

Optimize generated structures using xTB (GFN1, GFN2, GFN-FF) or MMFF94.

### Usage

```bash
MolCraftDiff analyze optimize gen_xyz/ --level gfn2 --charge 0
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output-dir` | `input_dir/optimized_xyz` | Output directory |
| `-l, --level` | `gfn1` | Optimization level: gfn1, gfn2, gfn-ff, mmff94 |
| `-c, --charge` | `0` | Molecular charge |
| `-t, --timeout` | `240` | Timeout per molecule (seconds) |
| `-s, --scale-factor` | `1.3` | Covalent radii scale factor |

### Output

Optimized XYZ files saved to `output_dir/` with same filenames.

---

## Part 2: Validity & Connectivity Metrics

Compute structural validation metrics for generated molecules.

### Usage

```bash
MolCraftDiff analyze metrics gen_xyz/ --metrics all
```

### Metric Types

| Type | Description |
|------|-------------|
| `core` | Basic validity (connectivity, atom stability) |
| `posebuster` | Bond lengths, angles, clashes |
| `geom_revised` | Aromatic-aware stability metrics |
| `all` | All of the above |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | None | Output CSV file |
| `-m, --metrics` | `all` | Metric type to compute |
| `--recheck-topo` | False | Recheck topology using RDKit |
| `--check-strain` | False | Check strain via XTB optimization |
| `--mol-converter` | `xyz2mol` | XYZ to mol converter |

---

## Part 3: Compare to Optimized Geometries

Compare generated structures with their optimized counterparts.

### Prerequisites

Run optimization first to create `optimized_xyz/` subdirectory:
```bash
MolCraftDiff analyze optimize gen_xyz/
```

### Usage

```bash
MolCraftDiff analyze compare gen_xyz/ --level gfn2
```

### Computed Metrics

- **RMSD**: Root Mean Square Deviation between original and optimized
- **Energy Difference**: xTB energy change
- **Bond Geometry**: Bond length and angle deviations

### Output

Results saved to CSV with per-molecule metrics.

---

## Part 4: XYZ to SMILES Conversion

Convert 3D XYZ files to 2D SMILES and extract molecular fingerprints.

### Usage

```bash
MolCraftDiff analyze xyz2mol gen_xyz/ --bits 2048
```

### Output Files (in `xyz_dir/2d_reprs/`)

| File | Description |
|------|-------------|
| `smiles_processed.csv` | Filename → SMILES mapping |
| `fingerprints.npy` | Morgan fingerprints array |
| `scaffolds.txt` | Murcko scaffolds |
| `substructures.json` | Substructure counts |

---

## Part 5: XTB Electronic Properties

Compute quantum-chemical descriptors at GFN-xTB level using morfeus.

### Usage

```bash
# Basic energy properties
MolCraftDiff analyze xtb-electronic gen_xyz/ -p energy

# All properties with JSON output
MolCraftDiff analyze xtb-electronic gen_xyz/ -p all -f json -o results.json

# ASE database for downstream analysis
MolCraftDiff analyze xtb-electronic gen_xyz/ -p all -f ase -o results.db
```

### Property Groups

**Molecular-level:**
| Group | Properties |
|-------|------------|
| `energy` | HOMO, LUMO, HOMO-LUMO gap |
| `dipole` | Dipole vector and magnitude |
| `reactivity` | Ionization potential, electron affinity |
| `global` | Electrophilicity, nucleophilicity, fugalities |

**Atomic-level:**
| Group | Properties |
|-------|------------|
| `charges` | Mulliken atomic charges |
| `fukui` | Fukui indices (f⁺, f⁻, radical, dual) |
| `bond_orders` | Wiberg bond orders |

### Output Formats

| Format | Description |
|--------|-------------|
| `csv` | Molecular-level properties (one row per molecule) |
| `json` | Full data including atomic-level properties |
| `ase` | ASE database with properties in atoms.info/arrays |
| `all` | Generate all three formats |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --method` | `2` | XTB method: 1=GFN1, 2=GFN2, ptb=PTB |
| `-c, --charge` | `0` | Molecular charge |
| `-p, --properties` | `energy` | Property groups to compute |
| `-f, --format` | `csv` | Output format |
| `--corrected/--no-corrected` | True | Apply empirical IP/EA correction |
| `-j, --n-jobs` | `1` | Parallel jobs |

---

## Example Workflow

A typical post-generation analysis workflow:

```bash
# 1. Generate molecules
MolCraftDiff generate gen_config.yaml

# 2. Optimize geometries
MolCraftDiff analyze optimize gen_xyz/ -l gfn2 -o gen_xyz/optimized_xyz

# 3. Compute validity metrics
MolCraftDiff analyze metrics gen_xyz/optimized_xyz -o metrics.csv

# 4. Compare to optimized structures
MolCraftDiff analyze compare gen_xyz/

# 5. Convert to SMILES for downstream analysis
MolCraftDiff analyze xyz2mol gen_xyz/optimized_xyz

# 6. Compute electronic properties
MolCraftDiff analyze xtb-electronic gen_xyz/optimized_xyz -p all -f ase -o electronic.db
```

---

## Python API

All analyze functions are also available programmatically:

```python
from MolecularDiffusion.runmodes.analyze import (
    optimize_molecule,
    get_xtb_optimized_xyz,
    compute_xtb_electronic,
    batch_xtb_electronic,
    run_compare_analysis,
    run_xyz2mol,
)

# Compute electronic properties for single file
result = compute_xtb_electronic(
    "molecule.xyz",
    method=2,
    properties=["energy", "charges"]
)
print(result["homo"], result["lumo"])

# Batch processing
df = batch_xtb_electronic(
    input_dir="gen_xyz/",
    output_path="results.csv",
    output_format="csv",
    properties=["energy", "reactivity"],
)
```
