# Installation

## Prerequisites

- Python 3.11
- A CUDA-capable GPU is recommended for training

## Step-by-step

```bash
# Create and activate a new environment
conda create -n moleculardiffusion python=3.11 -c defaults
conda activate moleculardiffusion

# Install PyTorch (adjust CUDA version for your system)
# https://pytorch.org/get-started/
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Geometric
# https://pytorch-geometric.readthedocs.io/
pip install torch_geometric

# Optional PyG extensions
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# System-level chemistry tools
conda install conda-forge::openbabel
conda install xtb==6.7.1

# Python dependencies
pip install fire seaborn decorator numpy scipy rdkit-pypi posebusters==0.5.1 \
    networkx matplotlib pandas scikit-learn tqdm pyyaml omegaconf ase \
    morfeus-ml wandb rmsd

pip install hydra-core==1.* hydra-colorlog rootutils

# Install cell2mol
git clone https://github.com/lcmd-epfl/cell2mol
cd cell2mol && python setup.py install && cd .. && rm -rf cell2mol

# Install MolCraftDiffusion in editable mode (makes MolCraftDiff CLI available)
pip install -e .
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
