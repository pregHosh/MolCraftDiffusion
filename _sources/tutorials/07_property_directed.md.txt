# Tutorial 7: Property-Directed Generation

> **Prerequisites:** [Tutorial 5 — Generation Overview](05_generation_overview.md) (plus a guidance model from [Tutorial 2](02_training_regressor.md)/[Tutorial 3](03_training_guidance.md) for GG) · **You'll learn:** CFG, Gradient Guidance, and hybrid guidance · **Next:** [Tutorial 8 — Predict & Evaluate](08_eval_predict.md)

This tutorial covers advanced generation techniques that steer the process towards desired chemical properties.

:::{warning}
**Guidance models must match the base diffusion model.** For Gradient Guidance (GG) and hybrid CFG/GG, the regressor and guidance model must be trained with the **same `diffusion_steps` and noise schedule** as the base diffusion model. A mismatch means the guidance model receives noise levels it was never trained on, producing erratic or broken gradients. Check that `diffusion_steps` and `noise_schedule` in your guidance config match the base model's training config exactly.
:::

## Contents

1.  **Introduction**: The concept of directing generation with external models or guidance schemes.
2.  **Classifier-Free Guidance (CFG)**: How to use CFG to amplify the effect of training conditions.
3.  **Gradient Guidance (GG)**: How to use a trained regressor model (from Tutorial 2) to guide generation towards a specific property value.
4.  **Hybrid CFG/GG Guidance**: How to combine both CFG and GG for multi-objective guidance.

---

## 1. Introduction

Property-directed generation allows you to guide the diffusion model to generate molecules with specific desired properties. This is achieved by providing an additional signal to the model during the sampling process. This tutorial covers three main techniques for property-directed generation. You can create your experiment configuration files in any directory, as the base templates are bundled with the package.

## 2. Classifier-Free Guidance (CFG)

Classifier-Free Guidance is a technique that amplifies the learned conditional distribution of the diffusion model. It uses two forward passes of the model: one with the condition and one without. The difference between the two outputs is then used to guide the generation process.

### Configuration

The configuration for CFG typically inherits from the `interference: gen_cfg` template.

| Parameter | Description |
| :--- | :--- |
| `task_type` | Must be set to `conditional`. |
| `target_values` | A list of positive target values for the properties specified in `property_names`. |
| `negative_target_value` | (Optional) A list of values to use as a "negative prompt". The model is guided *away* from these property values. |
| `property_names`| A list of property names that the model was trained on. |
| `cfg_scale` | A scaling factor that controls the strength of the guidance. A higher value will result in a stronger push towards the target properties. |

### Example `my_cfg.yaml`

```yaml
defaults:
  - tasks: diffusion
  - interference: gen_cfg # Base template bundled with package
  - _self_

name: "akatsuki"
chkpt_directory: "models/edm_formed_s1t1/"
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 600
seed: 9

interference:
  num_generate: 100
  target_values: [3,1.5]
  property_names: ["S1_exc", "T1_exc"]
  output_path: generated_mol
  condition_configs:
    cfg_scale: 1
    negative_target_value: [1.0, 3.0] # Push away from S1=1.0, T1=3.0
```

### Running CFG Generation

```bash
MolCraftDiff generate my_cfg
```

## 3. Gradient Guidance (GG)

Gradient Guidance uses a separate, pre-trained regressor or guidance model (like the one from Tutorial 2 or 3) to estimate the gradient of a desired property with respect to the molecule's latent representation. This gradient is then used to guide the diffusion process towards molecules with the desired property value.

### Configuration

The configuration for GG typically inherits from the `interference: gen_gg` template.

| Parameter | Description |
| :--- | :--- |
| `task_type` | Must be set to `gradient_guidance`. |
| `target_function` | Specifies the guidance model to use. This is configured using Hydra's instantiation syntax. `_target_` points to a callable class that will be instantiated to guide the generation. |
| `gg_scale` | A scaling factor for the gradient. |
| `max_norm` | The maximum norm of the gradient to prevent exploding gradients. |
| `scheduler` | A learning rate scheduler for the guidance. |
| `guidance_at` | The timestep at which to start applying the guidance. |
| `guidance_stop`| The timestep at which to stop applying the guidance. |
| `n_backwards` | The number of backward steps to take for the guidance. |

### Example `my_gg.yaml`

```yaml
defaults:
  - tasks: diffusion
  - interference: gen_gg # Base template bundled with package
  - _self_

name: "akatsuki"
chkpt_directory: "models/edm_formed_s1t1/"
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 600
seed: 9

interference:
  num_generate: 100
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
MolCraftDiff generate my_gg
```

## 4. Hybrid CFG/GG Guidance

It is also possible to combine CFG and GG to guide the generation with both the internal conditional model and an external guidance model.

### Configuration

The configuration for hybrid CFG/GG typically inherits from the `interference: gen_cfggg` template. It combines the parameters from both CFG and GG.

### Example `my_cfggg.yaml`

```yaml
defaults:
  - tasks: diffusion
  - interference: gen_cfggg # Base template bundled with package
  - _self_

name: "akatsuki"
chkpt_directory:  "models/edm_formed_s1t1/"
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 600
seed: 9

interference:
  num_generate: 100
  target_values: [3,1.5]
  property_names: ["S1_exc", "T1_exc"]
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
MolCraftDiff generate my_cfggg
```