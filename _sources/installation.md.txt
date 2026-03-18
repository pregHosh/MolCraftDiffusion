# Installation

## Prerequisites

- Python 3.11
- A CUDA-capable GPU is recommended for training

## Step-by-step

```bash
# 1. Create and activate a new environment
conda create -n molcraft python=3.11 -y
conda activate molcraft

# 2. Install conda-only tools (xtb, openbabel)
conda install -c conda-forge xtb==6.7.1 openbabel -y

# 3. Install MolCraftDiffusion with PyTorch + PyG + sparse extensions
pip install molcraftdiffusion[gpu] \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html

# or CPU-only:
pip install molcraftdiffusion[cpu] \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cpu.html

# Optional: xyz-to-SMILES conversion via cell2mol
git clone https://github.com/lcmd-epfl/cell2mol
cd cell2mol && python setup.py install && cd .. && rm -rf cell2mol
```

### Development / editable install

```bash
git clone https://github.com/pregHosh/MolCraftDiffusion
cd MolCraftDiffusion
conda install -c conda-forge xtb==6.7.1 openbabel -y
pip install -e .[gpu] \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

## Optional dependencies

```bash
# Symmetry analysis (requires numpy==1.24.*)
pip install cosymlib
```

## Verifying the installation

```bash
MolCraftDiff --help
```

You should see a list of all available commands: `train`, `generate`, `predict`, `eval_predict`, `analyze`, `data`.

## Pre-trained models

Pre-trained checkpoints are available on [Hugging Face](https://huggingface.co/pregH/MolecularDiffusion).
We recommend starting from these for any downstream application.
