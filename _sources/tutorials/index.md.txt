# Tutorials

A hands-on walkthrough of MolCraftDiffusion, ordered as a learning path: **prepare data → train → generate → evaluate**. Each tutorial is self-contained, but they build on one another, so first-time users should follow them in order.

:::{tip}
New here? Start with [Installation](../installation.md), then work through **Data Preparation → Training a Diffusion Model → Generation Overview**. The rest are optional deep-dives you can reach for when you need them.
:::

## Start Here

::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} Quickstart · Generate Your First Molecules
:link: quickstart_model_zoo
:link-type: doc

No data, no training. Fetch a pretrained model from the zoo and generate 3D
molecules in three commands — then come back and train your own.
:::

::::

## 1 · Prepare

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 0 · Data Preparation
:link: 00_data_preparation
:link-type: doc

Compile raw `.xyz` files into ASE databases, featurise, augment, and wire them into the `DataModule`.
:::

::::

## 2 · Train

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 1 · Diffusion Model
:link: 01_training_diffusion
:link-type: doc

Train a 3D molecular diffusion model from scratch with the override-only config workflow.
:::

:::{grid-item-card} 2 · Regressor
:link: 02_training_regressor
:link-type: doc

Train a property predictor — a standalone model and the basis for gradient guidance.
:::

:::{grid-item-card} 3 · Guidance Model
:link: 03_training_guidance
:link-type: doc

Train a time-aware regressor on noisy data to steer generation towards target properties.
:::

:::{grid-item-card} 4 · Fine-Tuning
:link: 04_finetuning
:link-type: doc

Adapt a pretrained model to a new chemical space, add conditions, or specialise for outpainting.
:::

::::

## 3 · Generate

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 5 · Generation Overview
:link: 05_generation_overview
:link-type: doc

Unconditional sampling and the three generation modes at a glance.
:::

:::{grid-item-card} 6 · Structure-Guided
:link: 06_structure_guided
:link-type: doc

Inpainting, outpainting and SILVR with 3D geometric constraints, plus a full parameter-tuning guide.
:::

:::{grid-item-card} 7 · Property-Directed
:link: 07_property_directed
:link-type: doc

Steer generation with Classifier-Free Guidance, Gradient Guidance, or a hybrid of both.
:::

::::

## 4 · Evaluate & Scale

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 8 · Predict & Evaluate
:link: 08_eval_predict
:link-type: doc

Predict properties for new molecules and benchmark a model against a labelled set.
:::

:::{grid-item-card} 9 · Analyse
:link: 09_analyze
:link-type: doc

Post-generation analysis: xTB optimisation, validity metrics, RMSD, electronic properties, featurisation.
:::

:::{grid-item-card} 10 · Generation Sweeps
:link: 10_generation_sweeps
:link-type: doc

Grid and Bayesian sweeps over controlled-generation parameters with automatic metric collection.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

quickstart_model_zoo
00_data_preparation
01_training_diffusion
02_training_regressor
03_training_guidance
04_finetuning
05_generation_overview
06_structure_guided
07_property_directed
08_eval_predict
09_analyze
10_generation_sweeps
```
