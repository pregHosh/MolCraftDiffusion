# Installation

## Prerequisites

- Python ≥3.10, <3.14 (3.11 recommended)
- A CUDA-capable GPU is recommended for training

## Step-by-step

```bash
# 1. Create and activate a new environment
conda create -n molcraft python=3.11 -y
conda activate molcraft

# 2. Install MolCraftDiffusion with a compute backend
# GPU/CUDA:
pip install molcraftdiffusion[gpu] \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html

# CPU-only:
pip install molcraftdiffusion[cpu] \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

The base package does not install every data-processing or analysis dependency. Add the feature groups you need:

```bash
# Data preparation, augmentation, and featurisation commands
pip install 'molcraftdiffusion[data]'

# Analysis and post-processing commands (metrics, compare, xyz2mol, xtb-electronic, featurise SOAP)
pip install 'molcraftdiffusion[analyze]'

# Backbone-specific groups
pip install 'molcraftdiffusion[bio]'      # DiffPharma: build a pocket + pharmacophore
                                          # particles from a raw PDB+SDF pair. Not needed
                                          # to train or generate from converted ASE dbs.
pip install 'molcraftdiffusion[shape]'    # DiffSMol: offline shape-cache precompute only
pip install 'molcraftdiffusion[flowmol]'  # FlowMol: DGL (install the CUDA build matching
                                          # your torch, see pyproject.toml)

# xTB is used by optimise, compare, and xtb-electronic — best installed from conda-forge:
conda install -c conda-forge xtb==6.7.1 -y
conda install xtb-python -y
```

If an optional command is called without its dependencies, MolCraftDiffusion exits with a warning and an install hint such as `pip install 'molcraftdiffusion[analyze]'`.

### Development / editable install

```bash
git clone https://github.com/pregHosh/MolCraftDiffusion
cd MolCraftDiffusion
pip install -e .[gpu] \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Add optional groups for editable development when needed:
pip install -e '.[data]'
pip install -e '.[analyze]'
```

## Which extras do I need?

Use this table to decide which optional groups to install before you start:

| What you want to do | Install |
| :--- | :--- |
| Train or generate only (no data prep, no analysis) | *(base `[gpu]` or `[cpu]` is enough)* |
| Compile raw XYZ files into an ASE database | `[data]` |
| Featurise molecules with SOAP descriptors | `[data]` |
| Run validity/connectivity metrics, xyz2mol, RMSD compare | `[analyze]` |
| Run `analyze optimize` or `xtb-electronic` | `[analyze]` + conda `xtb` |
| Run geometric-shape metrics (`--metrics geom_revised` or `all`) | `[analyze]` + `pip install cosymlib` |
| Featurise with UMA neural-network embeddings | `[analyze]` + fairchem clone (see below) |
| Pharmacophore-conditioned training/generation | `pip install open3d` |

## Optional dependencies

```bash
# Data utilities (includes dscribe for SOAP featurisation)
pip install 'molcraftdiffusion[data]'

# Analysis utilities (PoseBusters/RDKit/OpenBabel Python bindings)
pip install 'molcraftdiffusion[analyze]'

# DiffPharma novel-pocket preprocessing (Biopython + ODDT)
pip install 'molcraftdiffusion[bio]'

# Optional: needed for geometric-shape metrics in
# `MolCraftDiff analyze metrics --metrics {core,geom_revised,all}`
pip install cosymlib

# xTB executable for xTB-backed analysis
conda install -c conda-forge xtb==6.7.1 -y
```

### UMA featurisation backend

The `featurize --backend uma` command uses a pretrained UMA model from fairchem.
fairchem is **not** installed as a pip package — the source tree is vendored into
the repository and loaded at runtime.

Clone it into the repo root before using the UMA backend:

```bash
# from the MolCraftDiffusion repo root
git clone https://github.com/pregHosh/fairchem fairchem
```

A pretrained UMA checkpoint is also required. Download `uma-s-1p2.pt` from
[Hugging Face](https://huggingface.co/pregH/MolecularDiffusion) and place it at:

```
training_outputs/uma-s-1p2.pt
```

or pass a custom path with `--checkpoint /path/to/checkpoint.pt`.

If the fairchem source tree is not found at runtime, MolCraftDiffusion will print
an explicit error with the clone instruction above. You can also set:

```bash
export MOLCRAFT_REPO_ROOT=/path/to/MolCraftDiffusion
```

to point to the repo root when running from a different working directory.

## Verifying the installation

```bash
MolCraftDiff --help
```

You should see a list of all available commands: `train`, `generate`, `generate-sweep`, `predict`, `eval-predict`, `analyze`, `data`.

## Pre-trained models

Pre-trained checkpoints are available on [Hugging Face](https://huggingface.co/pregH/MolecularDiffusion).
We recommend starting from these for any downstream application.

---

## Troubleshooting

### `torch_scatter` / `torch_sparse` import errors

PyTorch Geometric sparse extensions must be compiled against your **exact** PyTorch + CUDA version. If you see `ImportError: … torch_scatter`, rebuild from the correct wheel:

```bash
# Check your torch version first
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Then install matching wheels (replace cu124/torch-2.6.0 as needed)
pip install torch-scatter torch-sparse \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

If the PyG wheel server does not have a prebuilt wheel for your exact version, you may need to build from source or use a matching Docker image.

---

### xTB or OpenBabel not found at runtime

`xtb` and `openbabel` are **not pip-installable** in a way that exposes the executables and shared libraries MolCraftDiffusion relies on. Installing them via pip creates a broken partial install that fails silently during optimisation or xyz2mol conversion.

Always install from conda-forge **before** the pip install:

```bash
conda install -c conda-forge xtb==6.7.1 openbabel -y
conda install xtb-python -y
```

If you already have a broken pip install, uninstall it first: `pip uninstall xtb openbabel`.

---

### `open3d` import errors on headless servers

`open3d` requires an OpenGL context for some of its initialisation paths. On headless HPC nodes you may see errors like `libGL.so.1: cannot open shared object file`.

Install the headless variant:

```bash
pip install open3d-cpu
```

or set the environment variable before running:

```bash
export OPEN3D_CPU_RENDERING=true
```

---

### `MolCraftDiff` command not found

Ensure the package is installed in the active conda environment and the environment is activated:

```bash
conda activate molcraft
MolCraftDiff --help
```

If installed in editable mode, run from the repository root where `.project-root` is visible.
