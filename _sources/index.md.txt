# MolCraftDiffusion

**A unified generative-AI framework for 3D molecular design.**

MolCraftDiffusion is an open-source framework for **3D molecular generation using diffusion models**, designed for **data-driven molecular design and computational chemistry**. The framework enables researchers to train generative models that produce chemically meaningful 3D molecular structures while supporting property optimization, scaffold modification, and exploration of chemical space.

By combining modular training pipelines, flexible guidance strategies, and integrated analysis tools, MolCraftDiffusion provides a complete workflow for developing and deploying **molecular diffusion models** in research applications such as drug discovery, catalyst discovery, materials design, and molecular property optimization.

![Workflow overview](../images/overview.png)

[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/pregHosh/MolCraftDiffusion)
[![PyPI](https://img.shields.io/pypi/v/molcraftdiffusion)](https://pypi.org/project/molcraftdiffusion/)
[![arXiv](https://img.shields.io/badge/PDF-arXiv-blue)](https://chemrxiv.org/engage/chemrxiv/article-details/6909e50fef936fb4a23df237)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19511401.svg)](https://zenodo.org/records/19511401)
[![Weights](https://img.shields.io/badge/Weights-HuggingFace-yellow)](https://huggingface.co/pregH/MolecularDiffusion)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/pregH/MolecularDiffusion)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/pregH/MolCraftDiffusion-demo)

---

## Key Features

MolCraftDiffusion is built with **modularity** at its core, offering an all-in-one, systematic workflow entirely driven by a unified CLI and YAML configuration files.

- **Data Module** — Preprocess, compile, and manage raw `.xyz` files into unified `.db` (ASE Database) pipelines, and annotate properties.
- **Training & Fine-Tuning Module** — Flexibly train (or fine-tune) diffusion models, property regressors, and time-aware guidance models. 
- **Generation & Guidance Module** — Generate 3D molecules using a variety of guidance mechanisms:
  - *Unconditional Generation*: Generate 3D molecules without any specific constraints or guidance.
  - *Property-Targeted Guidance*: Steer generation toward desired properties using Classifier-Free Guidance (CFG), Gradient Guidance (GG), or a hybrid approach.
  - *Structure-Guided Generation*: Perform inpainting (scaffold decoration) and outpainting (fragment extension) with precise 3D geometric constraints.
- **Analysis & Evaluation Module** — Assess the quality of generated 3D molecules. Includes tools for structural validity metrics, xTB geometry optimization, RMSD comparisons, and quantum-chemical property calculation/prediction.

---

## Quick Start

```bash
# Train a diffusion model
MolCraftDiff train my_config

# Generate molecules
MolCraftDiff generate my_gen_config

# Analyse outputs
MolCraftDiff analyze metrics generated_molecules/
```

Ready-to-use template configuration files for common workflows are listed in [Configuration Templates](config_templates.md), with the full packaged Hydra defaults under `src/MolecularDiffusion/configs/`. Copy the relevant file, fill in your paths, and run:

```bash
# Example: unconditional generation with the template
cp docs/cfg_examples/gen_unconditional.yaml my_gen.yaml
# edit my_gen.yaml → set chkpt_directory
MolCraftDiff generate my_gen
```

---

## Contents

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/00_data_preparation
tutorials/01_training_diffusion
tutorials/02_training_regressor
tutorials/03_training_guidance
tutorials/04_finetuning
tutorials/05_generation_overview
tutorials/06_structure_guided
tutorials/07_property_directed
tutorials/08_eval_predict
tutorials/09_analyze
```

```{toctree}
:maxdepth: 1
:caption: Applications

applications/index
applications/local_chemical_space
applications/library_design
applications/inverse_design
```

```{toctree}
:maxdepth: 1
:caption: Workflows

workflows/index
workflows/generate_and_filter
workflows/transfer_learning
workflows/conditioned_generation
workflows/end_to_end
workflows/visualization
```

```{toctree}
:maxdepth: 1
:caption: Configuration Templates

config_templates
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api
```
