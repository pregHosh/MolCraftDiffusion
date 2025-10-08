# Tutorial 4: Fine-Tuning a Diffusion Model

Fine-tuning is a powerful technique where you take a pre-trained model and continue training it. This is useful for adapting a general model to a specific chemical space, teaching it new tricks, or simply improving its performance without starting from zero.

## The Core Concepts of Fine-Tuning

Fine-tuning is activated by one key parameter: `tasks.chkpt_path`. By providing a path to a pre-trained model checkpoint here, you tell the trainer to load those weights instead of starting from scratch.

Another universal best practice for fine-tuning is to use a **very low learning rate**. Since the model is already trained, you only need to make small adjustments. A learning rate of `1e-5` or `1e-6` is a good starting point.

> **Note: Pre-trained Models Available!**
>
> You don't have to train a model from scratch to get started. We provide a collection of pre-trained models on our Hugging Face Hub. You can download them from:
> **https://huggingface.co/pregH/MolecularDiffusion**
>
> You can use the path to one of these downloaded models in your `tasks.chkpt_path` to start fine-tuning immediately.

This tutorial covers three common fine-tuning scenarios.

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
  context_mask_rate: 0.1 # Make it CFG-ready
```

--- 

### Scenario 3: Fine-Tune for Outpainting

**Goal:** To specialize a model to become an expert at "growing" new functional groups from a common scaffold.

**Configuration:** You load a pre-trained model and fine-tune it on a dataset of molecules, telling it which atoms belong to the core scaffold.

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

For any of these scenarios, you launch the training with the same `MolCraftDiff train` command:

```bash
# Example for Scenario 2
MolCraftDiff train finetune_add_condition
```