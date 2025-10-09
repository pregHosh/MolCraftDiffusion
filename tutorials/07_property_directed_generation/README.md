# Tutorial 7: Property-Directed Generation

This tutorial covers advanced generation techniques that steer the process towards desired chemical properties.

## Contents

1.  **Introduction**: The concept of directing generation with external models or guidance schemes.
2.  **Classifier-Free Guidance (CFG)**: How to use CFG to amplify the effect of training conditions.
3.  **Gradient Guidance (GG)**: How to use a trained regressor model (from Tutorial 2) to guide generation towards a specific property value.
4.  **Hybrid CFG/GG Guidance**: How to combine both CFG and GG for multi-objective guidance.

---

## 1. Introduction

Property-directed generation allows you to guide the diffusion model to generate molecules with specific desired properties. This is achieved by providing an additional signal to the model during the sampling process. This tutorial covers three main techniques for property-directed generation.

## 2. Classifier-Free Guidance (CFG)

Classifier-Free Guidance is a technique that amplifies the learned conditional distribution of the diffusion model. It uses two forward passes of the model: one with the condition and one without. The difference between the two outputs is then used to guide the generation process.

### Configuration

The configuration for CFG is in `configs/interference/gen_cfg.yaml`.

| Parameter | Description |
| :--- | :--- |
| `task_type` | Must be set to `conditional`. |
| `target_values` | A list of target values for the properties specified in `property_names`. |
| `property_names`| A list of property names that the model was trained on. |
| `cfg_scale` | A scaling factor that controls the strength of the guidance. A higher value will result in a stronger push towards the target properties. |

### Example `gen_cfg.yaml`

```yaml
_target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
task_type: conditional
sampling_mode: "ddpm"
num_generate: 100
mol_size:  [0,0]
target_values: [3,1.5]
property_names: ["S1_exc", "T1_exc"]
batch_size: 1
seed: 86
visualize_trajectory: False
output_path: generated_mol
condition_configs:
  cfg_scale: 1
```

### Running CFG Generation

```bash
MolCraftDiff generate interference=gen_cfg
```

## 3. Gradient Guidance (GG)

Gradient Guidance uses a separate, pre-trained regressor model (like the one from Tutorial 2) to estimate the gradient of a desired property with respect to the molecule's latent representation. This gradient is then used to guide the diffusion process towards molecules with the desired property value.

### Configuration

The configuration for GG is in `configs/interference/gen_gg.yaml`.

| Parameter | Description |
| :--- | :--- |
| `task_type` | Must be set to `gradient_guidance`. |
| `target_function` | A dictionary that specifies the pre-trained guidance model to use. |
| `gg_scale` | A scaling factor for the gradient. |
| `max_norm` | The maximum norm of the gradient to prevent exploding gradients. |
| `scheduler` | A learning rate scheduler for the guidance. |
| `guidance_ver` | The version of the guidance implementation to use. |
| `guidance_at` | The timestep at which to start applying the guidance. |
| `guidance_stop`| The timestep at which to stop applying the guidance. |
| `n_backwards` | The number of backward steps to take for the guidance. |

### Example `gen_gg.yaml`

```yaml
_target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
task_type: gradient_guidance # gg
sampling_mode: "ddpm"
num_generate: 100
mol_size:  [0,0]
target_values: []
property_names: []
batch_size: 1
seed: 86
visualize_trajectory: False
output_path: generated_mol
condition_configs:
  cfg_scale: 0
  target_function:
    _target_: scripts.gradient_guidance.sf_energy_score.SFEnergyScore
    _partial_: true
    chkpt_directory: trained_models/egcl_guidance_s1t1.ckpt
  gg_scale: 1e-3
  max_norm: 1e-3
  scheduler:
    _target_: scripts.gradient_guidance.scheduler.CosineAnnealing
    _partial_: true
    T_max: 1000
    eta_min: 0
  guidance_ver: 2
  guidance_at: 1
  guidance_stop: 0
  n_backwards: 0
```

### Running GG Generation

```bash
MolCraftDiff generate interference=gen_gg
```

## 4. Hybrid CFG/GG Guidance

It is also possible to combine CFG and GG to guide the generation with both the internal conditional model and an external guidance model.

### Configuration

The configuration for hybrid CFG/GG is in `configs/interference/gen_cfggg.yaml`. It combines the parameters from both CFG and GG.

### Example `gen_cfggg.yaml`

```yaml
_target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
task_type: gradient_guidance # cfggg
sampling_mode: "ddpm"
num_generate: 100
mol_size:  [0,0]
target_values: [3,1.5]
property_names: ["S1_exc", "T1_exc"]
batch_size: 1
seed: 86
visualize_trajectory: False
output_path: generated_mol
condition_configs:
  cfg_scale: 1
  target_function:
    _target_: scripts.gradient_guidance.sf_energy_score.SFEnergyScore
    _partial_: true
    chkpt_directory: trained_models/egcl_guidance_s1t1.ckpt
  gg_scale: 1e-3
  max_norm: 1e-3
  scheduler:
    _target_: scripts.gradient_guidance.scheduler.CosineAnnealing
    _partial_: true
    T_max: 1000
    eta_min: 0
  guidance_ver: 2
  guidance_at: 1
  guidance_stop: 0
  n_backwards: 3
```

### Running Hybrid CFG/GG Generation

```bash
MolCraftDiff generate interference=gen_cfggg
```
