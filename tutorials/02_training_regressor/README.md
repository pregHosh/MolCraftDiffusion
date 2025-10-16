# Tutorial 2: Training a Regressor Model

This tutorial explains how to train a model to predict specific molecular properties (e.g., energy, solubility). This regressor model can be used as a standalone predictor or, more powerfully, as a guidance model to steer molecule generation towards desired property values (as we will see in Tutorial 07).

## The "Override-Only" Workflow

We will follow the same workflow as before: all configuration will be done in a single experiment file that overrides a set of default templates.

**Important Note:** The configuration files for this tutorial must be placed in the `configs/` directory at the root of the project for the scripts to read the settings.

### Step 1: Create Your Experiment File

Let's create a new configuration file for our regression experiment.

```bash
# You can start with a copy of a general training config
cp configs/train.yaml configs/my_regressor_run.yaml
```

Now, open `configs/my_regressor_run.yaml` to begin editing.

### Step 2: Set the `defaults` for Regression

This is the most important change. We need to tell Hydra to use the `regression` task and its corresponding trainer settings, which are optimized for this task.

```yaml
# In my_regressor_run.yaml
defaults:
  - data: mol_dataset
  - tasks: regression      # Use the regression task configuration
  - logger: wandb
  - trainer: regression  # Use the regression-specific trainer settings
  - _self_
```

### Step 3: Set Your Key Parameters

Now, we override the defaults to configure our specific experiment. Below are the key parameters and recommended settings for training a regressor.

#### **Essential Paths**

| Parameter | Example Override | Description |
| :--- | :--- | :--- |
| `trainer.output_path` | `trainer: {output_path: "results/my_regressor"}` | **CRITICAL:** Where your trained regressor model is saved. |
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

#### **Data Settings**
| Parameter | Example Override | Notes / Recommendations |
| :--- | :--- | :--- |
| `data.batch_size` | `data: {batch_size: 128}` | A larger batch size can often be used for this task. |
| `data.data_type` | `data: {data_type: "pyg"}` | **CRITICAL:** For regression and guidance tasks, the data type must be set to `pyg`. |

#### **Regression Task Hyperparameters**

| Parameter | Example Override | Notes / Recommendations |
| :--- | :--- | :--- |
| `tasks.task_learn` | `tasks: {task_learn: ["S1_exc"]}` | **CRITICAL:** Tell the model which property from your dataset to predict. |
| `tasks.hidden_size` | `tasks: {hidden_size: 512}` | Regressors often benefit from being wider than diffusion models. `512` is a good starting point. |
| `tasks.act_fn` | `act_fn: {_target_: torch.nn.ReLU}` | `ReLU` is a common and effective activation function for regression tasks. |
| `tasks.num_layers` | `tasks: {num_layers: 1}` | For property prediction, it is preferred to have just one block of EGCL. |
| `tasks.num_sublayers`| `tasks: {num_sublayers: 4}` | Inside the single EGCL block, use multiple sublayers for a deeper model. |

#### **Trainer Settings for Regression**

| Parameter | Example Override | Notes / Recommendations |
| :--- | :--- | :--- |
| `trainer.optimizer_choice`| `trainer: {optimizer_choice: "adam"}` | `adam` is a solid default optimizer for regression. |
| `trainer.lr` | `trainer: {lr: 0.0005}` | Regression can often be trained with a slightly higher learning rate than diffusion models. |
| `trainer.scheduler` | `trainer: {scheduler: "reducelronplateau"}` | `reducelronplateau` is highly recommended. It automatically lowers the learning rate when validation loss stops improving. |
| `trainer.ema_decay` | `trainer: {ema_decay: 0.0}` | **Important:** Exponential Moving Average (EMA) is typically disabled for regressor training by setting the decay to `0.0`. |

#### **Experiment Logging**

| Parameter | Example Override | Description |
| :--- | :--- | :--- |
| `logger.project_wandb` | `logger: {project_wandb: "My_Regressor_Project"}` | (W&B only) The name of the project on your W&B dashboard. |
| `name` | `name: "s1_t1_regressor"` | The top-level `name` is used as the **run name** for logs. |

### Step 4: Putting It All Together

Here is what a complete `my_regressor_run.yaml` might look like:

```yaml
# Inherit from the regression-specific default templates
defaults:
  - data: mol_dataset
  - tasks: regression
  - logger: wandb
  - trainer: regression
  - _self_

# Set top-level experiment parameters
name: "my_s1_t1_regressor"
seed: 42

# Override Paths and Hyperparameters
trainer:
  output_path: "training_outputs/my_s1_t1_regressor"
  num_epochs: 100

logger:
  project_wandb: "My_Regressor_Project"

data:
  # Assuming default data paths are correct for this example
  data_type: "pyg" # Set data type for regression
  batch_size: 128

tasks:
  # Tell the model to learn the S1 and T1 excitation energies
  task_learn: ["S1_exc", "T1_exc"]
  hidden_size: 512
  num_layers: 1
  num_sublayers: 4
```

### Step 5: Run Your Training

Launch the training with the `MolCraftDiff train` command, pointing to your new config file:

```bash
MolCraftDiff train my_regressor_run
```

The trained model checkpoint will be saved in `training_outputs/my_s1_t1_regressor`.