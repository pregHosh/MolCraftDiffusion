# Tutorial 4: Fine-Tuning a Diffusion Model

Fine-tuning is a powerful technique where you take a pre-trained model and continue training it. This is useful for adapting a general model to a specific chemical space, teaching it new tricks, or simply improving its performance without starting from zero.

## The Core Concepts of Fine-Tuning

**Important Note:** The configuration files for this tutorial must be placed in the `configs/` directory at the root of the project for the scripts to read the settings.

Fine-tuning is activated by one key parameter: `tasks.chkpt_path`. By providing a path to a pre-trained model checkpoint here, you tell the trainer to load those weights instead of starting from scratch.

Another universal best practice for fine-tuning is to use a **very low learning rate**. Since the model is already trained, you only need to make small adjustments. A learning rate of `1e-5` or `1e-6` is a good starting point.

> **Note: Pre-trained Models Available!**
> 
> You don't have to train a model from scratch to get started. We provide a collection of pre-trained models on our Hugging Face Hub. You can download them from:
> **https://huggingface.co/pregH/MolecularDiffusion**
> 
> You can use the path to one of these downloaded models in your `tasks.chkpt_path` to start fine-tuning immediately.

This tutorial covers three common fine-tuning scenarios.

### Data Configuration for Fine-Tuning

The first time you use a dataset for a training job, the script processes the raw data (e.g., `.xyz` files) into an optimized format for faster loading. This processed data is saved as a file named `processed_data_{dataset_name}.pt` inside the directory specified by `data.root`. On subsequent runs, if this file exists, it will be loaded directly, saving significant time.

| Parameter | Example Override | Description |
| :--- | :--- | :--- |
| `data.root` | `data: {root: "data/processed"}` | The directory where processed dataset files are stored. |
| `data.dataset_name` | `data: {dataset_name: "my_molecule_set"}` | A unique name for your processed dataset. This becomes part of the saved filename (`processed_data_my_molecule_set.pt`). This is crucial for preventing conflicts when you work with multiple datasets. |
| `data.max_atom` | `data: {max_atom: 50}` | Sets the maximum molecular size (number of atoms). Larger molecules will be discarded. If not specified, the maximum size is determined automatically by scanning the dataset, which can be slow. |

**Best Practice:**
- Always assign a unique and descriptive `dataset_name` for each distinct dataset you use in your experiments.
- This practice ensures that you can easily manage and reuse your processed data without accidentally overwriting or loading the wrong file, which is especially important when switching between different fine-tuning tasks.

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

**Key Parameter for Outpainting:**

| Parameter | Example | Description |
| :--- | :--- | :--- |
| `tasks.reference_indices` | `[0, 1, 2, 3, 4, 5]` | A list of 0-indexed atom indices that define the common scaffold (the "core") of the molecules in your dataset. These atoms will be treated as the fixed part of the molecule during training. |

> **Important Data Preprocessing Note:**
> For this fine-tuning scenario to work correctly, you **must** preprocess your dataset to ensure that the core atom indices are consistent across all molecules. For example, if your scaffold is a benzene ring, the atoms of the ring should have the same indices (e.g., 0 through 5) in every molecule's coordinate file in your training set.

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