# Tutorial 2: Training a Regressor Model

This tutorial explains how to train a model to predict specific molecular properties (e.g., energy, solubility). This regressor model can be used as a standalone predictor or, more powerfully, as a guidance model to steer molecule generation towards desired property values (as we will see in Tutorial 07).

## Configuration

We use the exact same override-only configuration workflow introduced in **[Tutorial 1: Training a Diffusion Model](01_training_diffusion.md)**, but we load the `regression` templates via the `defaults` list.

```yaml
# Inside my_regressor_run.yaml
defaults:
  - data: mol_dataset
  - tasks: regression      # Use the regression task configuration
  - logger: wandb
  - trainer: regression    # Use the regression-specific trainer settings
  - _self_
```

## Key Parameters for Regression

Below are the key parameters and recommended settings to override when training a regression model.

| Parameter | Example Override | Description |
| :--- | :--- | :--- |
| `trainer.output_path` | `trainer: {output_path: "results/my_regressor"}` | **CRITICAL:** Where your trained regressor model is saved. |
| `data.ase_db_path` | `data: {ase_db_path: "data/my_dataset.db"}` | Path to your compiled ASE database containing molecular properties. |

*(Note: Ensure you have prepared your database as described in [Tutorial 0: Data Preparation & Management](00_data_preparation.md).)*

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

## Putting It All Together

Here is a complete `my_regressor_run.yaml` example:

```yaml
defaults:
  - data: mol_dataset
  - tasks: regression
  - logger: wandb
  - trainer: regression
  - _self_

name: "my_s1_t1_regressor"
seed: 42

trainer:
  output_path: "training_outputs/my_s1_t1_regressor"
  num_epochs: 100

logger:
  project_wandb: "My_Regressor_Project"

data:
  data_type: "pyg"
  batch_size: 128

tasks:
  task_learn: ["S1_exc", "T1_exc"]
  hidden_size: 512
  num_layers: 1
  num_sublayers: 4
```

Launch the training as usual:

```bash
MolCraftDiff train my_regressor_run
```

---

## Next Steps: Property Prediction and Guidance

Once your regression model is trained, you can use it in two main ways:

1.  **As a Standalone Predictor**: Use the `predict` module to predict properties for batches of existing 3D molecules.
2.  **As a Guidance Model**: Use the predictions to steer the creation of new molecules. 

Learn how to use your regressor to guide diffusion generation in **[Tutorial 7: Property-Directed Generation (CFG/GG)](07_property_directed.md)**.