# Tutorial 9: Analyze Module - 3D Molecular Structure Analysis

This tutorial covers the **analyze** module, which provides tools for post-generation analysis and validation of 3D molecular structures.

---

## Overview

The analyze module includes six subcommands:

| Command | Description |
|---------|-------------|
| `optimize` | XTB geometry optimization |
| `metrics` | Validity/connectivity metrics |
| `compare` | RMSD and energy comparison |
| `xyz2mol` | XYZ to SMILES + fingerprints |
| `xtb-electronic` | XTB electronic properties |
| `featurize` | Fixed-size molecular feature vectors (SOAP / UMA) |

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

## Part 6: Featurize — Fixed-Size Molecular Vectors

Convert a directory of XYZ files into a fixed-size feature matrix for downstream
machine learning (clustering, regression, dimensionality reduction, etc.).

Two backends are available:

| Backend | Description | GPU needed |
|---------|-------------|-----------|
| `soap` | SOAP descriptor via dscribe | No |
| `uma` | UMA backbone embeddings from pretrained fairchem model | Optional |

### Usage

```bash
# SOAP (default) — uses the built-in species list
MolCraftDiff analyze featurize gen_xyz/

# SOAP — auto-detect species from the files
MolCraftDiff analyze featurize gen_xyz/ --autodetect

# SOAP — specify species explicitly
MolCraftDiff analyze featurize gen_xyz/ --species C --species H --species N --species O

# SOAP — custom descriptor parameters
MolCraftDiff analyze featurize gen_xyz/ --n-max 12 --l-max 9 --r-cut 8.0

# UMA — CPU inference
MolCraftDiff analyze featurize gen_xyz/ --backend uma --device cpu

# UMA — GPU inference with custom checkpoint
MolCraftDiff analyze featurize gen_xyz/ --backend uma --device cuda \
    --checkpoint training_outputs/uma-s-1p2.pt

# UMA — all spherical components (higher-dimensional embedding)
MolCraftDiff analyze featurize gen_xyz/ --backend uma --all-components
```

### SOAP Options

| Option | Default | Description |
|--------|---------|-------------|
| `--autodetect` | False | Detect element species from files (overrides `--species`) |
| `--species` | See below | Element symbols; repeatable: `--species C --species H` |
| `--r-cut` | `6.0` | Cutoff radius in Å |
| `--n-max` | `8` | Radial basis functions |
| `--l-max` | `6` | Angular basis functions |
| `--sigma` | `0.1` | Gaussian smearing width |
| `--pooling` | `mean` | Atom pooling mode: `mean` or `sum` |
| `--soap-jobs` | `1` | Parallel workers |

**Default species list:**
`H B C N O F Al Si P S Cl As Se Br I Hg Bi`

Elements found in the files that are not in the species list are added automatically
with a warning, so the run never fails silently on unseen atoms.

### UMA Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | `training_outputs/uma-s-1p2.pt` | Path to UMA checkpoint `.pt` file |
| `--task-name` | `omol` | UMA task name |
| `--device` | auto | `cuda` or `cpu` |
| `--batch-size` | `8` | Molecules per UMA forward pass |
| `--charge` | `0` | Total molecular charge applied to all structures |
| `--spin` | `1` | Spin multiplicity applied to all structures |
| `--all-components` | False | Use all spherical components instead of L=0 scalars only |
| `--pooling` | `mean` | Atom pooling mode: `mean` or `sum` |

The UMA backend requires the vendored fairchem source tree at `<repo_root>/fairchem/src`.
If it is not found, clone it with:

```bash
git clone https://github.com/pregHosh/fairchem fairchem
```

Then run from the repository root, or set:
```bash
export MOLCRAFT_REPO_ROOT=/path/to/MolCraftDiffusion
```

### Output Files

Three files are written to the output stem (default: `input_dir/features`):

| File | Content |
|------|---------|
| `features.npy` | `(N, D)` float32 feature matrix |
| `features.csv` | Row index → source file + frame mapping |
| `features_meta.json` | Backend, all parameters, feature dim, timestamp |

```bash
# Custom output stem
MolCraftDiff analyze featurize gen_xyz/ -o results/soap_features
# writes: results/soap_features.npy / .csv / _meta.json
```

### Loading the Output

```python
import numpy as np
import pandas as pd

features = np.load("gen_xyz/features.npy")        # (N, D) float32
index    = pd.read_csv("gen_xyz/features.csv")    # maps row → file/frame

print(features.shape)   # e.g. (160, 81396) for SOAP, (160, 128) for UMA
```

### Python API

```python
from MolecularDiffusion.runmodes.analyze.featurize import run_featurize

# SOAP
features = run_featurize(
    input_dir="gen_xyz/",
    backend="soap",
    output_path="gen_xyz/soap_features",
    n_max=12,
    l_max=9,
)

# UMA — pass a pre-loaded list of ASE Atoms directly
from MolecularDiffusion.runmodes.analyze.uma_embeddings import get_uma_molecule_embeddings
from ase.io import read

atoms_list = [read("gen_xyz/molecule_0000.xyz"), read("gen_xyz/molecule_0001.xyz")]
results = get_uma_molecule_embeddings(
    source=atoms_list,
    checkpoint_path="training_outputs/uma-s-1p2.pt",
    device="cpu",
    charge=0,
    spin=1,
)
mol_emb = results[0]["molecule_embedding"]   # torch.Tensor, shape (128,)
node_emb = results[0]["node_embedding"]      # torch.Tensor, shape (n_atoms, 128)
```

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

# 7. Featurize for downstream ML (SOAP)
MolCraftDiff analyze featurize gen_xyz/optimized_xyz -o gen_xyz/soap_features

# 7b. Or use UMA embeddings (requires fairchem checkout + checkpoint)
MolCraftDiff analyze featurize gen_xyz/optimized_xyz --backend uma --device cuda \
    -o gen_xyz/uma_features
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
from MolecularDiffusion.runmodes.analyze.featurize import run_featurize
from MolecularDiffusion.runmodes.analyze.uma_embeddings import get_uma_molecule_embeddings

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
