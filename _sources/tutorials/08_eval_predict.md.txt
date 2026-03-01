# Tutorial 8: Prediction and Evaluation Modes

This tutorial explains how to use the inference capabilities of the MolecularDiffusion package. There are two main modes for inference:

1.  **Prediction (`predict`)**: For generating predictions on a set of new molecules (XYZ files) without ground truth labels.
2.  **Evaluation (`eval_predict`)**: For benchmarking a model against a labeled dataset to calculate error metrics.

---

## Part 1: Prediction Mode (`predict`)

Use this mode when you have a folder of geometry files (e.g., `.xyz`) and want to predict their properties using a trained model.

### 1. Configuration

Create a configuration file (e.g., `my_prediction.yaml`) to specify your input files and model checkpoint. You can create this file in any directory.

```yaml
# @package _global_

defaults:
  - tasks: guidance      # Base template bundled with package
  - interference: prediction # Base template bundled with package
  - _self_

# 1. Run Name (used for logging)
name: "screening_run"

# 2. Model Checkpoint
# Path to the .pkl file of your trained model.
# Note: Even though the parameter is named 'directory', it expects a file path.
chkpt_directory: "trained_models/guidance-epoch=195-metric=0.1975.pkl"

# 3. Input Data
# Directory containing your .xyz files.
xyz_directory: "test_xyz_pred/"

# 4. Output Location
output_directory: "test_output/my_predictions"

# 5. Model Specifics
# These must match the configuration used during training.
# If unknown, check the 'atom_vocab' and 'node_feature' in your training config.
atom_vocab: [H, B, C, N, O, F, Al, Si, P, S, Cl, As, Se, Br, I, Hg, Bi]
node_feature: null  # e.g., null, "atom_geom", "atom_topological"

# 6. Constraints
# Skip molecules larger than this size to avoid memory issues
max_atoms: 100
```

### 2. Running the Command

Execute the prediction using the `MolCraftDiff` CLI, pointing to your config file:

```bash
MolCraftDiff predict my_prediction
```

*(Note: Do not include the `.yaml` extension in the command)*

### 3. Output

The script will process each XYZ file in the directory and output the results to your specified `output_directory`:

*   **`predictions.csv`**: A table containing the filename of each molecule and its predicted properties.
*   **`*_hist.png`**: Histogram of the predicted values.
*   **`*_kde.png`**: Kernel Density Estimation plot of the predicted distribution.
*   **`kde_all.png`**: A combined plot if multiple properties were predicted.

---

## Part 2: Evaluation Mode (`eval_predict`)

Use this mode when you have a labeled dataset (ground truth) and want to quantify how well your model performs (e.g., calculating Mean Absolute Error, plotting Correlation).

### 1. Configuration

Create a configuration file (e.g., `my_evaluation.yaml`). This looks more like a training config because it needs to load a full dataset object.

```yaml
# @package _global_

defaults:
  - data: mol_dataset    # Base template bundled with package
  - tasks: guidance      # Base template bundled with package
  - trainer: default
  - hydra: default
  - _self_

name: "benchmark_run"
output_directory: "output_pred/benchmark"

# 1. Data Configuration
data:
  # Path where processed data (.pt files) are stored/cached
  root: "data/processed/"
  
  # Path to the CSV file containing ground truth labels
  filename: "data/test_set.csv"
  
  # Unique name for this dataset (creates 'processed_data_test_set.pt')
  dataset_name: "test_set_benchmark"
  
  # Directory containing corresponding .xyz files
  xyz_dir: "data/test_xyz/"
  
  max_atom: 100
  data_type: pyg # Keep as 'pyg' for regression/guidance tasks
  
  # CRITICAL: Set train_ratio to 0.0 to treat the whole file as a test set
  train_ratio: 0.0 
  batch_size: 1

# 2. Task & Model Configuration
tasks:
  # Path to the trained model checkpoint
  chkpt_path: "trained_models/guidance-epoch=195-metric=0.1975.pkl"
  
  # List of tasks/columns to evaluate against
  task_learn: ["gap", "homo", "lumo"] 

# 3. Reproducibility
seed: 9
```

### 2. Running the Command

Execute the evaluation using the `MolCraftDiff` CLI, pointing to your config file:

```bash
MolCraftDiff eval_predict my_evaluation
```

### 3. Output

The script calculates predictions and matches them with the ground truth from your CSV. Results are saved to `output_directory`:

*   **`predictions.csv`**: Contains `filename`, `y_true` (ground truth), and `y_pred` (prediction) for every molecule.
*   **`*_correlation.png`**: Scatter plot comparing True vs. Predicted values.
*   **`*_kde.png` / `*_hist.png`**: Distribution plots.
*   **Console Output**: Summary statistics (Mean, Std, Min, Max) for the predictions.

> **⚠️ Important Notice on Data Formats:**
> 
> The `eval_predict` command currently **only supports `.csv` files** for ground truth labels. It does **not** natively support extracting ground truth data directly from ASE `.db` files. Ensure you have a generated CSV containing the `"filename"` and property columns.

---

## Summary of Differences

| Feature | Prediction Mode (`predict`) | Evaluation Mode (`eval_predict`) |
| :--- | :--- | :--- |
| **Input** | Folder of XYZ files | CSV file + Folder of XYZ files |
| **Ground Truth** | Not required | Required (in CSV) |
| **Output** | Predictions only | Predictions vs. Ground Truth |
| **Use Case** | Screening new molecules | Benchmarking model accuracy |
| **Config Key** | `chkpt_directory` | `tasks.chkpt_path` |
