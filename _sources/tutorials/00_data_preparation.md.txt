# Tutorial 0: Data Preparation & Management

This tutorial covers the complete data pipeline in `MolecularDiffusion`—from raw 3D structures to training-ready PyTorch datasets. It is divided into two main parts:

1. **The CLI Data Operations (`MolCraftDiff data`)**: How to compile, augment, and featurize your datasets into unified ASE databases.
2. **The `DataModule` Configuration**: How to load those databases into the training engine with on-the-fly preprocessing.

---

## Part 1: Data Preparation via CLI

The `MolCraftDiff data` command group provides utilities for managing raw `.xyz` files and converting them into unified `.db` (ASE Database) files.

### 1. Compilation & Annotation

First, compile your raw XYZ files or NumPy arrays into a single ASE database:

```bash
# From a directory of .xyz files
MolCraftDiff data prepare compile -s xyz_dir/ -d dataset.db

# From NumPy arrays with a metadata CSV
MolCraftDiff data prepare compile -s coords.npy -n natoms.npy -c metadata.csv -d dataset.db
```

Add custom metadata tags (like project or subset labels) to your new database:

```bash
MolCraftDiff data prepare annotate -d dataset.db -t group -v training
```

### 2. Featurization

Convert 3D structures into fixed-length vectors for downstream property guidance or evaluation tasks:

```bash
# Generate Morgan fingerprints (requires RDKit SMILES generation)
MolCraftDiff data featurize -m morgan -i dataset.db -o features/ --radius 2 --nbits 2048

# Compute SOAP descriptors
MolCraftDiff data featurize -m soap -i xyz_dir/ -o features/ --rcut 5.0 --nmax 8 --lmax 6
```


### 3. Data Augmentation

Increase dataset diversity via structural transformations:

```bash
# Charge Augmentation: Randomly add/remove hydrogens
MolCraftDiff data augment charge -i dataset.db -o augmented.db --max-h 1 --db

# Coordinate Distortion: Apply random Gaussian noise to atomic coordinates
MolCraftDiff data augment distortion -i xyz_dir/ -o noisy_xyz/ --sigma 0.1
```

### 4. ASE Database Operations

Manage and inspect your compiled datasets:

```bash
# Merge multiple DBs
MolCraftDiff data ase-ops merge -i db_dir/ -o merged.db

# Sample 10% of entries for a quick test
MolCraftDiff data ase-ops sample -i dataset.db -o subset.db --fraction 0.1

# Inspect property distributions and generate plots
MolCraftDiff data ase-ops inspect -d dataset.db --limit 10 --output plots/
```

---

## Part 2: Loading Data for Training (`DataModule`)

Once you have your compiled `dataset.db` (or just your raw CSV/XYZ files), you use the `DataModule` in your YAML configuration to load them into the training engine.

### 1. Supported Input Formats

The engine supports three primary inputs:

1.  **ASE Database** (Recommended): `load_db` - Reads the compiled `.db` files from Part 1.
2.  **CSV + XYZ**: `load_csv` - Reads a CSV metadata file mapping to a directory of `.xyz` files.
3.  **CSV + NPY**: `load_csv_npy` - Reads coordinates and atom counts from `.npy` files.

### 2. Data Types (`pointcloud` vs `pyg`)

Two molecular tensor representations are supported, set via the `data_type` config:

| Type | Class | Description |
|------|-------|-------------|
| `pointcloud` | `PointCloudDataset` | Dense tensor format with padding, used by standard EDM models. |
| `pyg` | `GraphDataset` | PyTorch Geometric graph format with edge indices, used by architectures like EGCL. |

### 3. Node Featurization

You can compute graph or geometric features on-the-fly via `node_feature_choice`:

*   **Geometric Features** (`atom_topological`, `atom_geom`, `atom_geom_v2`, `atom_geom_v2_trun`, `atom_geom_opt`): Computed directly from 3D coordinates. Works with all loader types.
*   **RDKit Scalar Features** (`['degree', 'formal_charge', 'hybridization', 'is_aromatic']`): Requires RDKit mol blocks, meaning it *only works if you compiled an ASE Database* (via `load_db`).

One-hot encoding of atom types is controlled separately via `use_ohe_feature: true`.

### 4. Example Data Configuration

Here is how you wire everything together in your experiment's YAML file to train on the `.db` you created in Part 1:

```yaml
data:
  _target_: MolecularDiffusion.runmodes.train.DataModule
  
  # 1. Point to your compiled database
  ase_db_path: data/augmented.db
  dataset_name: my_augmented_dataset  # Used to name the cached .pt file
  
  # 2. Representation
  data_type: pointcloud
  
  # 3. Features & Constraints
  use_ohe_feature: true
  node_feature_choice: geometric_fast
  atom_vocab: [H, C, N, O, F]
  max_atom: 29
  
  # 4. Training splits
  batch_size: 64
  train_ratio: 0.8
```

> **Note on Caching**: The first time the engine runs, it parses your DB and caches the PyTorch tensors as `data/processed_data_my_augmented_dataset.pt`. To force a re-parse, simply delete this `.pt` file.
