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

# Analysis and post-processing commands (metrics, xyz2mol, xtb-electronic, featurise SOAP)
pip install 'molcraftdiffusion[analyze]'

# Backbone-specific groups
pip install 'molcraftdiffusion[bio]'      # DiffPharma: build a pocket + pharmacophore
                                          # particles from a raw PDB+SDF pair. Not needed
                                          # to train or generate from converted ASE dbs.
pip install 'molcraftdiffusion[sbdd]'     # AutoDock Vina scoring of generated ligands
                                          # against a protein pocket. Pure pip: no conda,
                                          # no external binaries.
pip install 'molcraftdiffusion[shape]'    # DiffSMol: offline shape-cache precompute only
pip install 'molcraftdiffusion[flowmol]'  # FlowMol: DGL (install the CUDA build matching
                                          # your torch, see pyproject.toml)

# xTB is used by optimise, xtb-electronic, and the `conformer` strain column — best installed from conda-forge:
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

## macOS (Apple Silicon)

On a Mac with an M-series chip, MolCraftDiffusion trains and generates on the
Apple GPU through PyTorch's **MPS** backend. CUDA does not exist on macOS, so the
`[gpu]`/`[cpu]` commands above do not apply; use this route instead. The Linux /
WSL instructions above are unchanged.

```bash
# 1. Environment (conda-forge only)
conda create -n molcraft -c conda-forge --override-channels python=3.11 -y
conda activate molcraft

# 2. Optional: xTB / OpenBabel for analysis (Apple Silicon builds exist on conda-forge)
conda install -c conda-forge --override-channels xtb==6.7.1 xtb-python openbabel -y

# 3. MolCraftDiffusion from the repository (PyPI lags behind this version)
git clone https://github.com/pregHosh/MolCraftDiffusion
cd MolCraftDiffusion
pip install -e '.[mac]'
pip install ./mac_shims

# 4. Feature groups as needed, same as on Linux
pip install -e '.[data]'
pip install -e '.[analyze]'

# 5. Check the GPU is visible
python -c "import torch; print(torch.backends.mps.is_available())"   # True
```

What the Mac route changes:

- **`[mac]`** installs plain PyPI `torch==2.6.0` (which includes MPS) and
  `torch_geometric<2.8`, without the PyG C++ extensions (`torch_scatter`,
  `torch_sparse`, `torch_cluster`, `torch_spline_conv`), which have no MPS support.
- **`./mac_shims`** provides pure-torch `torch_scatter` and `torch_cluster` with
  the same functions and results, so every model runs on the GPU unchanged. Never
  install it on Linux next to the real extensions.
- **Device selection is automatic**: CUDA, then MPS, then CPU. Force a device
  with `MOLCRAFT_DEVICE`, e.g. `MOLCRAFT_DEVICE=cpu MolCraftDiff train ...`.
- Operations that MPS lacks fall back to the CPU automatically
  (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set for you on macOS).
- Keep `trainer.precision: 32`; mixed precision on MPS is limited.
- With `engine: lightning`, DataLoader workers are turned off automatically when
  the collate function cannot be sent to worker processes (macOS starts workers
  with `spawn`, Linux with `fork`).
- **Not accelerated on Mac**: FlowMol (`[flowmol]`, DGL has no MPS backend; its
  macOS wheel is CPU-only) and the UMA featurisation backend run on CPU.

Two extras need one extra step on Apple Silicon:

```bash
# [sbdd]: vina has no macOS arm64 wheel on PyPI; take it from conda-forge first
conda install -c conda-forge --override-channels vina -y
pip install -e '.[sbdd]'

# [bio]: oddt is source-only and its build needs `six` outside pip's build sandbox
pip install six
pip install --no-build-isolation oddt
pip install -e '.[bio]'
```

## Which extras do I need?

Use this table to decide which optional groups to install before you start:

| What you want to do | Install |
| :--- | :--- |
| Train or generate only (no data prep, no analysis) | *(base `[gpu]` or `[cpu]` is enough)* |
| Compile raw XYZ files into an ASE database | `[data]` |
| Featurise molecules with SOAP descriptors | `[data]` |
| Run validity/connectivity metrics, xyz2mol, paired `conformer` metrics | `[analyze]` |
| Run `analyze optimize` or `xtb-electronic` | `[analyze]` + conda `xtb` |
| Run geometric-shape metrics (`--metrics geom_revised` or `all`) | `[analyze]` |
| Score generated ligands against a protein pocket (`--metrics sbdd`) | `[analyze]` + `[sbdd]` |
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

# AutoDock Vina scoring of generated ligands against a protein pocket
# (`MolCraftDiff analyze metrics ... --metrics sbdd`)
pip install 'molcraftdiffusion[sbdd]'

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

### `ModuleNotFoundError: No module named 'gemmi'` from meeko

`meeko` declares no dependencies in its wheel metadata, but `import meeko`
pulls in `gemmi` (via `meeko.polymer` -> `meeko.chemtempgen`). The `[sbdd]`
extra lists `gemmi` explicitly, so the normal install is fine — you only hit
this if you installed meeko with `--no-deps`:

```bash
pip install gemmi
```

Neither `vina`, `meeko` nor `gemmi` touches an existing numpy/pandas install;
they add only themselves.

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
