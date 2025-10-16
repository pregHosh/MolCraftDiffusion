# Tutorial 1: Training a Diffusion Model

This tutorial explains how to configure and run a training job for a diffusion model from scratch. We will focus on using a single configuration file for your experiment to override the project's default settings.

## The "Override-Only" Workflow

This project uses a powerful configuration framework called [Hydra](https://hydra.cc/). The easiest and cleanest way to use it is to have **one single YAML file for your experiment** where you define all your custom settings.

**Important Note:** The configuration files for this tutorial must be placed in the `configs/` directory at the root of the project for the scripts to read the settings.

### Step 1: Create Your Experiment File

Your experiment file is your personal workspace. Start by copying the example template:

```bash
cp configs/example_diffusion_config.yaml configs/my_first_run.yaml
```

Now, open `configs/my_first_run.yaml`. This is the only file you'll need to edit.

### Step 2: Understand the `defaults` List

The `defaults` list at the top of the file loads a set of pre-defined "templates" for each part of your experiment (data, model, trainer, etc.).

```yaml
defaults:
  - data: mol_dataset
  - tasks: diffusion
  - logger: default
  - trainer: default
  - _self_
```

**Think of these default files as a reference manual.** You can look at them (e.g., in `configs/data/`, `configs/tasks/`) to see what parameters are available, but you should not edit them directly. All changes are made in `my_first_run.yaml`.

### Step 3: Set Your Key Parameters

This is the most important step. You will override the default parameters to configure your specific experiment. Below are the most common parameters you will want to set.

#### **Essential Paths**

| Parameter | Example Override in `my_first_run.yaml` | Description |
| :--- | :--- | :--- |
| `trainer.output_path` | `trainer: {output_path: "results/my_run"}` | **CRITICAL:** Where all logs and checkpoints are saved. |
| `data.filename` | `data: {filename: "molecules.csv"}` | The CSV file with molecule information. |
| `data.xyz_dir` | `data: {xyz_dir: "xyz_files/"}` | The directory containing `.xyz` geometry files. |

#### Data Processing and Caching

The first time you run a training job, the script processes your raw dataset (`.xyz` files, etc.) into a format suitable for training. This processed data is saved as a file named `processed_data_{dataset_name}.pt` inside the directory specified by `data.root`. On subsequent runs, if this file exists, it will be loaded directly to save time.

| Parameter | Example Override | Description |
| :--- | :--- | :--- |
| `data.root` | `data: {root: "data/processed"}` | The directory where processed dataset files are stored. |
| `data.dataset_name` | `data: {dataset_name: "my_molecule_set"}` | A unique name for your processed dataset. This becomes part of the saved filename (`processed_data_my_molecule_set.pt`). This is crucial for preventing conflicts when you work with multiple datasets. |
| `data.max_atom` | `data: {max_atom: 50}` | Sets the maximum molecular size (number of atoms). Larger molecules will be discarded. If not specified, the maximum size is determined automatically by scanning the dataset, which can be slow. |

**Best Practice:**
- Set a descriptive `dataset_name` for each new dataset you work with.
- This ensures that you can easily manage and reuse your processed data without accidentally overwriting or loading the wrong file.

#### **Core Training Hyperparameters**

| Parameter | Example Override | Description |
| :--- | :--- | :--- |
| `trainer.num_epochs` | `trainer: {num_epochs: 200}` | How long to train. |
| `trainer.lr` | `trainer: {lr: 0.0001}` | The learning rate. |
| `data.batch_size` | `data: {batch_size: 64}` | Number of molecules per batch. |
| `seed` | `seed: 42` | Top-level parameter for reproducibility. |

#### **Model & Task Hyperparameters**

This section defines the model architecture and the specifics of the diffusion task. You can configure several training modes:

*   **Unconditional Mode:** The model learns the general distribution of molecules. To use this mode, ensure `tasks.condition_names` is an empty list `[]`.

*   **Conditional Mode:** The model learns to generate molecules given certain properties (e.g., energy, size). To use this mode, you must provide a list of property names in `tasks.condition_names` and ensure your dataset contains columns with these exact names.

*   **Classifier-Free Guidance (CFG) Training:** A subset of conditional training. Set `tasks.context_mask_rate > 0` (e.g., `0.1`). This randomly hides the condition during training, enabling CFG during generation.

*   **Self-Pace Learning (Advanced):** A strategy where the model learns from "easier" examples first. Enable it with `tasks.sp_regularizer_deploy: True`.

**Key Model & Task Parameters to Override:**

| Parameter | Example Override | Description |
| :--- | :--- | :--- |
| `tasks.condition_names`| `tasks: {condition_names: [prop1, prop2]}` | List of property names for conditional training. Leave empty `[]` for unconditional. |
| `tasks.context_mask_rate`| `tasks: {context_mask_rate: 0.1}` | **(Conditional Only)** Probability of masking the condition. `0` for standard conditional training, `> 0` for CFG training. |
| `tasks.hidden_size` | `tasks: {hidden_size: 256}` | The main dimension/width of the model. |
| `tasks.num_layers` | `tasks: {num_layers: 9}` | The number of layers (depth) in the model. |
| `tasks.diffusion_steps`| `tasks: {diffusion_steps: 500}` | Number of steps in the diffusion process. |
| `tasks.sp_regularizer_deploy` | `tasks: {sp_regularizer_deploy: True}` | Set to `True` to enable Self-Pace Learning. |
| `tasks.sp_regularizer_regularizer` | `tasks: {sp_regularizer_regularizer: 'logaritmic'}` | The pacing function. Options: `hard` (default), `linear`, `logaritmic`, `logistic`. |
| `tasks.sp_regularizer_lambda_` | `tasks: {sp_regularizer_lambda_: 1}` | A key parameter that controls the learning pace. |

#### **Experiment Logging**

You can control how results are logged by overriding parameters under the `logger:` key. The most important choice is whether to log to local files or to [Weights & Biases](https://wandb.ai/) (`wandb`).

**To switch between loggers, modify the `defaults` list in `my_first_run.yaml`:**
*   For simple local file logging: `defaults: [..., - logger: default, ...]`
*   For Weights & Biases: `defaults: [..., - logger: wandb, ...]`

**Key Logging Parameters to Override:**

| Parameter | Example Override | Description |
| :--- | :--- | :--- |
| `logger.log_interval` | `logger: {log_interval: 10}` | How often (in training steps) to log metrics like loss. |
| `logger.project_wandb` | `logger: {project_wandb: "My_Project"}` | (W&B only) The name of the project on your W&B dashboard. |
| `name` | `name: "complex_mols_run_1"` | The top-level `name` parameter is used as the **run name** for both local logs and W&B. |

### Step 4: Putting It All Together

Here is what a complete `my_first_run.yaml` for a **CFG-ready conditional model** might look like:

```yaml
# Inherit from the default templates
defaults:
  - data: mol_dataset
  - tasks: diffusion
  - logger: wandb
  - trainer: default
  - _self_

# Set top-level experiment parameters
name: "my_cfg_model_training"
seed: 42

# Override Essential Paths and Hyperparameters
trainer:
  output_path: "training_outputs/my_cfg_model"
  num_epochs: 200
  lr: 0.0002

logger:
  project_wandb: "My_Diffusion_Project"

data:
  batch_size: 64

tasks:
  condition_names: ["S1_exc", "T1_exc"] # Specify properties from our dataset
  context_mask_rate: 0.1 # Enable CFG training
  hidden_size: 256
```

### Step 5: Run Your Training

Launch the training using the `MolCraftDiff` command-line tool. Provide the `train` command followed by the name of your configuration file.

**Command:**
```bash
MolCraftDiff train my_first_run
```

The tool will automatically find `my_first_run.yaml` in your `configs` directory, build the full configuration from your defaults and overrides, and start the training. All results will be saved in the `trainer.output_path` you specified.
