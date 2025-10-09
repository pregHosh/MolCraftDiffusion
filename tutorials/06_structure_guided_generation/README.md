# Tutorial 6: Structure-Guided Generation

This tutorial explains how to guide molecule generation using structural constraints, such as filling in a missing piece (inpainting) or growing a molecule from a fragment (outpainting).

## Contents

1.  **Introduction**: The concept of guiding generation with a structural template.
2.  **Building Your Own Configuration**: How to create a custom configuration file.
3.  **3D Geometric Constraints**: How to tune the geometric constraints.
4.  **Inpainting**: How to configure and run generation to fill in a missing portion of a molecule.
    *   Key configuration file: `configs/interference/gen_inpaint.yaml`
    *   Explanation of the input template file (see `assets/BINOLCpHHH.xyz`).
5.  **Outpainting**: How to grow a molecule from a given substructure.
    *   Key configuration file: `configs/interference/gen_outpaint.yaml`
    *   Explanation of the input fragment file (see `assets/BINOLCp.xyz`).

## 1. Introduction

Structure-guided generation allows you to influence the output of the diffusion model by providing a starting molecular structure. This is useful for tasks like:

*   **Inpainting**: Completing a molecule where a part is missing.
*   **Outpainting**: Extending a molecule from a given fragment.

The process involves providing a reference structure in an XYZ file and specifying which parts of the structure to modify or keep fixed. **Note that all atom indices are 0-indexed.**

## 2. Building Your Own Configuration

While this tutorial uses pre-made configurations, you can easily create your own for custom inpainting or outpainting tasks. The recommended workflow is to copy an existing configuration and modify it.

### Step 1: Create Your Experiment File

For inpainting, copy the `gen_inpaint.yaml` file:

```bash
cp configs/interference/gen_inpaint.yaml configs/interference/my_inpaint.yaml
```

For outpainting, copy the `gen_outpaint.yaml` file:

```bash
cp configs/interference/gen_outpaint.yaml configs/interference/my_outpaint.yaml
```

Open your new file (e.g., `configs/interference/my_inpaint.yaml`) to begin editing.

### Step 2: Set Key Parameters

The most important parameters to change are in the `condition_configs` section.

#### For Inpainting (`my_inpaint.yaml`):

| Parameter | Description |
| :--- | :--- |
| `reference_structure_path` | **CRITICAL:** Path to your own XYZ file containing the molecule you want to inpaint. |
| `mask_node_index` | **CRITICAL:** A list of **0-indexed** atom indices from your XYZ file that you want to remove and have the model regenerate. |
| `denoising_strength` | Controls how much noise is added to the masked region before generation. Higher values give the model more creative freedom. |
| `mol_size` | The expected size of the final molecule. **This should be larger than or equal to the number of atoms in the reference structure.** |

**Example `my_inpaint.yaml`:**

```yaml
_target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
task_type: inpaint
sampling_mode: "ddpm"
num_generate: 50
mol_size: [50, 60] # Target size of the generated molecule
output_path: "results/my_inpainting_run"
condition_configs:
  reference_structure_path: "assets/BINOLCpHHH.xyz"
  # To vary the BINOL part of the molecule, we mask the following 0-indexed atoms:
  mask_node_index: [5, 30, 31, 6, 7, 45, 8, 32, 9, 10, 33, 11, 34, 12, 35, 13, 36, 14, 15, 16, 17, 18, 37, 19, 38, 20, 39, 21, 40, 22, 23, 41, 24, 44, 25, 26, 43, 42]
  denoising_strength: 0.8
  # ... (other parameters can often be left as default)
```

#### For Outpainting (`my_outpaint.yaml`):

| Parameter | Description |
| :--- | :--- |
| `reference_structure_path` | **CRITICAL:** Path to your own XYZ file containing the fragment you want to grow from. |
| `connector_dicts` | **CRITICAL:** A dictionary where keys are the **0-indexed** indices of atoms in your fragment, and values are the number of new connections to grow from that atom. |
| `mol_size` | The expected size of the final molecule (fragment + generated part). |

**Example `my_outpaint.yaml`:**

```yaml
_target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
task_type: outpaint
sampling_mode: "ddpm"
num_generate: 50
mol_size: [30, 40] # Target size of the generated molecule
output_path: "results/my_outpainting_run"
condition_configs:
  reference_structure_path: "assets/BINOLCp.xyz"
  # To decorate BINOL-Cp with substituents at 0-indexed atoms 1, 2, and 3, each with 3 bonds:
  connector_dicts:
    1: [3]
    2: [3]
    3: [3]
  # ... (other parameters can often be left as default)
```

### Step 3: Run Your Custom Generation

You can run the generation using either the `python` script or the `MolCraftDiff` CLI.

Using `python`:
```bash
# For inpainting:
python scripts/generate.py --config-name=interference/my_inpaint

# For outpainting:
python scripts/generate.py --config-name=interference/my_outpaint
```

Using `MolCraftDiff` CLI:
```bash
# For inpainting:
MolCraftDiff generate interference/my_inpaint

# For outpainting:
MolCraftDiff generate interference/my_outpaint
```

Your generated molecules will be saved in the `output_path` you specified.

## 3. 3D Geometric Constraints

The generation process for both inpainting and outpainting is guided by a set of geometric constraints that can be tuned in the `condition_configs` section of your configuration file. These parameters control how the model handles collisions, connectivity, and the overall shape of the generated molecule.

### Inpainting Constraints

For inpainting, the main goal is to fill a missing region while respecting the geometry of the fixed part of the molecule.

| Parameter | Description |
| :--- | :--- |
| `d_threshold_f` | The minimum allowed distance (in Angstroms) between a generated atom and the fixed (frozen) atoms. If a generated atom is closer than this threshold, it will be pushed away. |
| `w_b` | A weight that controls a push/pull force on the generated atoms. A higher value results in a stronger push away from the fixed atoms. |
| `all_frozen` | If `True`, all atoms in the reference structure are considered frozen and cannot be moved. |
| `use_covalent_radii`| If `True`, the collision avoidance logic will use the covalent radii of the atoms to determine the minimum allowed distance, instead of the fixed `d_threshold_f`. |
| `scale_factor` | A scaling factor for the covalent radii when `use_covalent_radii` is `True`. A value greater than 1.0 increases the effective size of the atoms, creating more space between them. |
| `t_critical_1`, `t_critical_2` | These parameters control the timesteps during the diffusion process at which the geometric constraints are most strongly applied. |

### Outpainting Constraints

Outpainting uses a similar set of constraints, with a few additions to handle the process of growing a molecule from a fragment.

| Parameter | Description |
| :--- | :--- |
| `d_threshold_f` | The minimum allowed distance between a generated atom and the atoms of the initial fragment. |
| `d_threshold_c` | The minimum allowed distance between a generated atom and the connector atom it is attached to. |
| `w_b` | A weight that controls a push/pull force on the generated atoms, influencing their position relative to the fragment and its centroid. |
| `all_frozen` | If `False`, the atoms in the initial fragment can be slightly adjusted during the generation process. |
| `use_covalent_radii`| Same as in inpainting, but applied to the fragment and the generated atoms. |
| `scale_factor` | Same as in inpainting. |
| `t_critical_1`, `t_critical_2` | These parameters control the timesteps during the diffusion process at which the geometric constraints are most strongly applied. |

By tuning these parameters, you can gain fine-grained control over the geometry of the generated molecules.

## 4. Inpainting

Inpainting is the process of filling in a missing part of a molecule. You provide a template molecule and specify which atoms to "mask". The diffusion model will then generate the missing atoms and connect them to the rest of the molecule.

### Configuration

The configuration for inpainting is in `configs/interference/gen_inpaint.yaml`. Here are the key parameters:

*   `reference_structure_path`: Path to the XYZ file of the template molecule. In this tutorial, we use `assets/BINOLCpHHH.xyz`.
*   `mask_node_index`: A list of atom indices to be masked and regenerated. The atoms that are *not* in this list will be kept fixed.
*   `denoising_strength`: Controls how much the original structure is changed. A value of 1.0 means the masked part is completely replaced.

### The Inpainting Template (`assets/BINOLCpHHH.xyz`)

The `assets/BINOLCpHHH.xyz` file contains the complete molecule that will be used as a template for inpainting. The atoms specified in `mask_node_index` will be removed and regenerated by the model.


### Running Inpainting

To run inpainting, use the `scripts/generate.py` script with the `gen_inpaint.yaml` config:

```bash
python scripts/generate.py --config-name=interference/gen_inpaint
```

The generated molecules will be saved in the `generated_mol` directory.

## 5. Outpainting

Outpainting is the process of growing a molecule from a given fragment. You provide a starting fragment, and the model will add new atoms to it.

### Configuration

The configuration for outpainting is in `configs/interference/gen_outpaint.yaml`. Here are the key parameters:

*   `reference_structure_path`: Path to the XYZ file of the fragment. In this tutorial, we use `assets/BINOLCp.xyz`.
*   `connector_dicts`: A dictionary specifying which atoms of the fragment should be connected to the newly generated part. The keys are the indices of the atoms in the fragment, and the values are the number of new bonds to form.

### The Outpainting Template (`assets/BINOLCp.xyz`)

The `assets/BINOLCp.xyz` file contains the molecular fragment from which the new molecule will be grown.


### Running Outpainting

To run outpainting, use the `scripts/generate.py` script with the `gen_outpaint.yaml` config:

```bash
python scripts/generate.py --config-name=interference/gen_outpaint
```

The generated molecules will be saved in the `generated_mol` directory.
