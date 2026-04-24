# Installation

## Prerequisites

- Python 3.11
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
# Data preparation, augmentation, and featurization commands
pip install 'molcraftdiffusion[data]'

# Analysis and post-processing commands
pip install 'molcraftdiffusion[analyze]'

# Some analyze commands shell out to xTB.
conda install -c conda-forge xtb==6.7.1 -y
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

## Optional dependencies

```bash
# Data utilities
pip install 'molcraftdiffusion[data]'

# Analyze utilities, including PoseBusters/RDKit/OpenBabel Python bindings/cosymlib
pip install 'molcraftdiffusion[analyze]'

# xTB executable for xTB-backed analysis
conda install -c conda-forge xtb==6.7.1 -y
```

## Verifying the installation

```bash
MolCraftDiff --help
```

You should see a list of all available commands: `train`, `generate`, `predict`, `eval_predict`, `analyze`, `data`.

## Pre-trained models

Pre-trained checkpoints are available on [Hugging Face](https://huggingface.co/pregH/MolecularDiffusion).
We recommend starting from these for any downstream application.
