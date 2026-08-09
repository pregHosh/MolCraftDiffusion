# Tutorial 4: Fine-Tuning a Diffusion Model

> **Prerequisites:** [Tutorial 1 — Training a Diffusion Model](01_training_diffusion.md) · **You'll learn:** adapting a pretrained model to new data, adding conditions, or specialising for outpainting · **Next:** [Tutorial 5 — Generation Overview](05_generation_overview.md)

## At a Glance

| | |
| :--- | :--- |
| **Objective** | Adapt a compatible pretrained checkpoint to new data or conditioning. |
| **You need** | A base checkpoint, its original architecture settings, and a target dataset. |
| **Main command** | `MolCraftDiff train finetune_add_condition.yaml` |
| **Success looks like** | The checkpoint loads without architecture mismatches and a new versioned run is created. |

Fine-tuning adapts a pre-trained model to a specific chemical space or teaches it new capabilities, like conditional generation.

**This tutorial assumes you are familiar with the override-only configuration workflow from [Tutorial 1](01_training_diffusion.md).**

:::{warning}
**Architecture must match exactly.** The model architecture in your config (`hidden_size`, `num_layers`, attention heads, etc.) must be identical to the checkpoint you are loading. A mismatch causes a hard failure at checkpoint load time — not during training. Always start from the `diffusion_pretrained` task template when fine-tuning one of the provided pretrained models, and do not change model-size parameters between runs.
:::

## Core Fine-Tuning Principles

1.  **Matching Architecture is Critical**: Your config file's model architecture (`hidden_size`, `num_layers`, etc.) **must exactly match** the pre-trained checkpoint, or loading will fail. Use the `diffusion_pretrained` task template when adapting our provided models.
2.  **Use a Low Learning Rate**: The model is already trained; you only need small adjustments. Start with a learning rate around `1e-5` to `2e-5`.
3.  **Unique Dataset Names**: Always use a unique `data.dataset_name` so you don't corrupt the cached `.pt` files from your broader pre-training runs.

Fine-tuning is activated by a single parameter: `tasks.chkpt_path`. 

Below are three common scenarios.

---

### Scenario 1: Continue Training on a New Dataset

**Goal:** To adapt a general, pre-trained model to a new, more specific dataset.

**Configuration:** This is the simplest case. You load the pre-trained model and point the trainer to your new data.

**Example `finetune_new_data.yaml`:**

```yaml
defaults:
  - data: my_new_dataset # Use your new dataset configuration
  - tasks: diffusion_pretrained
  - logger: wandb
  - trainer: default
  - _self_

name: "finetune_on_new_data"
seed: 42

trainer:
  output_path: "training_outputs/finetuned_new_data"
  num_epochs: 50 # Fine-tuning often requires fewer epochs
  lr: 1e-5       # Use a very small learning rate

tasks:
  # CRITICAL: Path to the pre-trained model to start from
  chkpt_path: "path/to/downloaded_model.ckpt"
```

---

### Scenario 2: Fine-Tune to Add a Condition

**Goal:** To teach a pre-trained unconditional model to generate molecules based on specific properties (e.g., for conditional generation or CFG).

**Configuration:** You load the unconditional model but provide it with conditional data and settings during fine-tuning.

**Key Parameters for Adding a Condition:**

| Parameter | Example | Description |
| :--- | :--- | :--- |
| `tasks.condition_names`| `["S1_exc", "T1_exc"]` | A list of property names from your dataset that the model should learn to associate with the molecules. |
| `tasks.context_mask_rate`| `0.1` | The probability of hiding the condition during training. A value greater than 0 is required to enable Classifier-Free Guidance (CFG) during generation. A common value is 0.1 (10% of the time). |
| `tasks.mask_value`| `[0, 0]` | The value to use when a condition is masked. This should be a list with the same length as `condition_names`. Typically, this is `0` or the mean value of the property in the dataset. |
| `tasks.normalize_condition` | `"maxmin"` | The method used to normalise conditional properties. Options are: `"maxmin"` (scales to [-1, 1]), `"mad"` (mean absolute deviation), `"value_N"` (divides by a specific value N, e.g. `value_10`), or `null` for no normalisation. Default: `value_10`. |

**Concatenation vs. Adapter Conditioning:**

By default, the model injects conditional properties by naively **concatenating** them to the input node features. 
Alternatively, you can use **Adapters** (MLP networks) to project the conditions into the model's hidden dimension before adding them to the node features. This can be more expressive.
To use an adapter for a specific condition, list its name in `tasks.adapter_conditions`.

| Parameter | Example | Description |
| :--- | :--- | :--- |
| `tasks.adapter_conditions`| `["S1_exc"]` | A list of condition names (must be a subset of `condition_names`) that should be routed through Adapter MLPs instead of being concatenated. |

**Example `finetune_add_condition.yaml`:**

```yaml
defaults:
  - data: my_conditional_dataset # A dataset with property labels
  - tasks: diffusion_pretrained
  - logger: wandb
  - trainer: default
  - _self_

name: "finetune_for_cfg"
seed: 42

trainer:
  output_path: "training_outputs/finetuned_cfg_model"
  num_epochs: 50
  lr: 1e-5

tasks:
  # CRITICAL: Path to the pre-trained unconditional model
  chkpt_path: "path/to/downloaded_unconditional_model.ckpt"

  # KEY CHANGE: Add the conditions to learn
  condition_names: ["S1_exc", "T1_exc"]
  adapter_conditions: ["S1_exc"] # Process S1_exc through an adapter, T1_exc is concatenated
  context_mask_rate: 0.1 # Make it CFG-ready
  normalize_condition: value_10
```

---

### Scenario 3: Fine-Tune for Outpainting

**Goal:** To specialise a model to become an expert at "growing" new functional groups from a common scaffold.

**Configuration:** You load a pre-trained model and fine-tune it on a dataset of molecules, telling it which atoms belong to the core scaffold.

**Key Parameter for Outpainting:**

| Parameter | Example | Description |
| :--- | :--- | :--- |
| `tasks.reference_indices` | `[0, 1, 2, 3, 4, 5]` | A list of 0-indexed atom indices that define the common scaffold (the "core") of the molecules in your dataset. These atoms will be treated as the fixed part of the molecule during training. |

:::{important}
**Data preprocessing.** For this fine-tuning scenario to work correctly, you **must** preprocess your dataset to ensure that the core atom indices are consistent across all molecules. For example, if your scaffold is a benzene ring, the atoms of the ring should have the same indices (e.g., 0 through 5) in every molecule's coordinate file in your training set.
:::

**Example `finetune_outpainting.yaml`:**

```yaml
defaults:
  - data: my_scaffold_dataset # A dataset of molecules with the same core
  - tasks: diffusion_pretrained
  - logger: wandb
  - trainer: default
  - _self_

name: "finetune_for_outpainting"
seed: 42

trainer:
  output_path: "training_outputs/finetuned_outpainting_model"
  num_epochs: 25 # This can be a very short fine-tuning task
  lr: 2e-5

tasks:
  # CRITICAL: Path to a pre-trained model
  chkpt_path: "path/to/downloaded_model.ckpt"

  # KEY CHANGE: Define the scaffold atoms
  reference_indices: [0, 1, 2, 3, 4, 5] # The indices of the core atoms
```

### Run Your Fine-Tuning Job

For any of these scenarios, you launch the training with the same `MolCraftDiff train` command, pointing to your experiment config file:

```bash
# Example for Scenario 2
MolCraftDiff train finetune_add_condition.yaml
```

## Verify the Result

The run should report that pretrained weights were loaded before optimisation begins. Inspect any missing or newly initialised parameter warnings: newly added conditioning channels may be expected, but unrelated missing backbone weights are not.

## Troubleshooting

- A tensor-size mismatch means the configured architecture differs from the checkpoint architecture.
- If old data appears after changing datasets, use a unique `data.dataset_name` to avoid reusing the previous processed cache.
- If fine-tuning quickly destroys sample quality, reduce the learning rate and validate more frequently.
