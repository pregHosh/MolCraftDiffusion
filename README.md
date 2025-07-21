MolecularDiffusion
==================

A molecular diffusion framework for machine learning applications.

Installation
-----------

### Detailed Installation Guide

For a more detailed installation, including setting up a conda environment and installing necessary packages, follow these steps:

    # create new python environment
    conda create -n moleculardiffusion python=3.10 -c defaults
    conda activate moleculardiffusion

    # install pytorch according to instructions (use CUDA version for your system)
    # https://pytorch.org/get-started/
    conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    # install pytorch geometric (use CUDA version for your system)
    # https://pytorch-geometric.readthedocs.io/
    pip install torch_geometric

    # install other libraries
    pip install numpy scipy rdkit-pypi networkx matplotlib pandas scikit-learn tqdm pyyaml omegaconf ase morfeus cosymlib wandb torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

    # install cell2mol
    git clone https://github.com/lcmd-epfl/cell2mol
    cd cell2mol
    python setup.py install
    cd ..
    rm -rf cell2mol


For development installation:


    pip install -e ".[dev]"

Usage
-----

Basic usage:


    import MolecularDiffusion
    
    # Your code here

Command-line interface for training

    python scripts/train.py tasks=[TASK]

where TASK is one of the following: diffusion, guidance, regression

Command-line interface for generation

    python scripts/generate.py interference=[INTERFERENCE]

where INTERFERENCE is one of the following: gen_cfg, gen_cfggg, gen_conditional, gen

Command-line interface for prediction

    python scripts/predict.py



Documentation
------------

For more information, visit: https://moleculardiffusion.readthedocs.io


Project Structure
-----------------

```
├── .project-root
├── justfile
├── pyproject.toml
├── README.md
├── setup.py
├── configs
│   ├── generate.yaml
│   ├── predict.yaml
│   ├── train.yaml
│   ├── data
│   │   └── mol_dataset.yaml
│   ├── hydra
│   │   └── default.yaml
│   ├── interference
│   │   ├── gen_cfg.yaml
│   │   ├── gen_cfggg.yaml
│   │   ├── gen_conditional.yaml
│   │   ├── gen_gg.yaml
│   │   ├── gen_inpaint.yaml
│   │   ├── gen_outpaint.yaml
│   │   ├── gen_outpaintft.yaml
│   │   ├── gen_unconditional.yaml
│   │   └── prediction.yaml
│   ├── logger
│   │   └── default.yaml
│   ├── tasks
│   │   ├── diffusion.yaml
│   │   ├── guidance.yaml
│   │   └── regression.yaml
│   └── trainer
│       ├── default.yaml
│       └── regression.yaml
├── data
│   └── template_structures
├── scripts
│   ├── generate.py
│   ├── predict.py
│   ├── train.py
│   └── gradient_guidance
│       ├── scheduler.py
│       └── sf_energy_score.py
├── src
│   └── MolecularDiffusion
│       ├── __init__.py
│       ├── _version.py
│       ├── cli.py
│       ├── callbacks
│       │   ├── __init__.py
│       │   └── train_helper.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── core.py
│       │   ├── engine.py
│       │   ├── logger.py
│       │   └── meter.py
│       ├── data
│       │   ├── __init__.py
│       │   ├── dataloader.py
│       │   ├── dataset.py
│       │   └── component
│       │       ├── __init__.py
│       │       ├── dataset.py
│       │       ├── feature.py
│       │       └── pointcloud.py
│       ├── modules
│       │   ├── layers
│       │   │   ├── common.py
│       │   │   ├── conv.py
│       │   │   └── functional.py
│       │   ├── models
│       │   │   ├── __init__.py
│       │   │   ├── egcl.py
│       │   │   ├── egt.py
│       │   │   ├── en_diffusion.py
│       │   │   └── noisemodel.py
│       │   └── tasks
│       │       ├── __init__.py
│       │       ├── diffusion.py
│       │       ├── metrics.py
│       │       ├── regression.py
│       │       └── task.py
│       ├── runmodes
│       │   ├── __init__.py
│       │   ├── handler.py
│       │   ├── generate
│       │   │   ├── __init__.py
│       │   │   └── tasks_generate.py
│       │   └── train
│       │       ├── __init__.py
│       │       ├── data.py
│       │       ├── eval.py
│       │       ├── logger.py
│       │       ├── tasks_egcl.py
│       │       ├── tasks_egt.py
│       │       └── trainer.py
│       └── utils
│           ├── __init__.py
│           ├── comm.py
│           ├── diffusion_utils.py
│           ├── file.py
│           ├── geom_analyzer.py
│           ├── geom_constant.py
│           ├── geom_constraint.py
│           ├── geom_metrics.py
│           ├── geom_utils.py
│           ├── io.py
│           ├── molgraph_utils.py
│           ├── plot_function.py
│           ├── pretty.py
│           ├── sascore.py
│           ├── smilify.py
│           └── torch.py
```


License
-------

This project is licensed under the MIT License.


Citation
--------



ArXiv link: [*XXXXXX*](XXXXX)

```

```