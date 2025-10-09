# Tutorial 3: Training a Time-Aware Guidance Model

Training a guidance model is very similar to training a standard regressor (Tutorial 02), with one key difference: **the model is trained on noisy data.**

This process makes the model "time-aware," meaning it learns to predict properties of a molecule at various stages of the diffusion denoising process (from noisy to clean). This is the crucial feature that allows it to effectively guide a generative model.

## The "Override-Only" Workflow

We will follow the same workflow as before, using a single configuration file to set up the experiment.

### Step 1: Create Your Experiment File

```bash
cp configs/train.yaml configs/my_guidance_run.yaml
```

Open `configs/my_guidance_run.yaml` to begin editing.

### Step 2: Set the `defaults` for Guidance Training

We need to tell Hydra to use the `guidance` task. The trainer settings are often similar to regression, so we can start with the `regression` trainer.

```yaml
# In my_guidance_run.yaml
defaults:
  - data: mol_dataset
  - tasks: guidance      # Use the guidance task configuration
  - logger: wandb
  - trainer: regression  # The regression trainer is a good starting point
  - _self_
```

### Step 3: Set Your Key Parameters

Most settings are identical to the regressor setup. The main difference is the addition of "Noise Injection" parameters.

#### **Essential Paths**

| Parameter | Example Override | Description |
| :--- | :--- | :--- |
| `trainer.output_path` | `trainer: {output_path: "results/my_guidance"}` | **CRITICAL:** Where your trained guidance model is saved. |
| `data.root` | `data: {root: "/path/to/my/dataset"}` | The root directory of your dataset. |
| `data.filename` | `data: {filename: "molecules.csv"}` | The CSV file with molecule information. |
| `data.xyz_dir` | `data: {xyz_dir: "xyz_files/"}` | The directory containing `.xyz` geometry files. |

#### **Data Settings**
| Parameter | Example Override | Notes / Recommendations |
| :--- | :--- | :--- |
| `data.batch_size` | `data: {batch_size: 128}` | A larger batch size can often be used for this task. |
| `data.data_type` | `data: {data_type: "pyg"}` | **CRITICAL:** For regression and guidance tasks, the data type must be set to `pyg`. |

#### **Guidance Task Hyperparameters**

| Parameter | Example Override | Notes / Recommendations |
| :--- | :--- | :--- |
| `tasks.task_learn` | `tasks: {task_learn: ["S1_exc"]}` | **CRITICAL:** The property from your dataset the model should learn to predict on noisy data. |
| `tasks.hidden_size` | `tasks: {hidden_size: 512}` | A wider model (`512`) is often a good choice. |
| `tasks.num_layers` | `tasks: {num_layers: 1}` | For property prediction, it is preferred to have just one block of EGCL. |
| `tasks.num_sublayers`| `tasks: {num_sublayers: 4}` | Inside the single EGCL block, use multiple sublayers for a deeper model. |

#### **Noise Injection Settings (The Key Difference)**
These parameters control how noise is added to the molecules during training.

| Parameter | Example Override | Notes / Recommendations |
| :--- | :--- | :--- |
| `tasks.diffusion_steps` | `tasks: {diffusion_steps: 900}` | The number of steps in the diffusion process. **Should match your generative diffusion model.** |
| `tasks.diffusion_noise_schedule` | `tasks: {diffusion_noise_schedule: 'polynomial_2'}` | The noise schedule. **Should match your generative diffusion model.** |
| `tasks.t_max` | `tasks: {t_max: 0.7}` | The maximum noise timestep (from 0.0 to 1.0) the model will be trained on. |

### Step 4: Putting It All Together

Here is what a complete `my_guidance_run.yaml` might look like:

```yaml
# Inherit from the default templates
defaults:
  - data: mol_dataset
  - tasks: guidance
  - logger: wandb
  - trainer: regression
  - _self_

# Set top-level experiment parameters
name: "my_time_aware_guidance_model"
seed: 42

# Override Paths and Hyperparameters
trainer:
  output_path: "training_outputs/my_guidance_model"
  num_epochs: 100

logger:
  project_wandb: "My_Guidance_Models"

data:
  data_type: "pyg" # Set data type for guidance
  batch_size: 128

tasks:
  task_learn: ["S1_exc", "T1_exc"]
  hidden_size: 512
  num_layers: 1
  num_sublayers: 4
  # Noise injection settings
  diffusion_steps: 900
  t_max: 0.8
```

### Step 5: Run Your Training

Launch the training with the `MolCraftDiff train` command:

```bash
MolCraftDiff train my_guidance_run
```

The trained model checkpoint will be saved in `training_outputs/my_guidance_model`.