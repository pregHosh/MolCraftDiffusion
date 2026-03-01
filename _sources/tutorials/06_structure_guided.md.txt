# Tutorial 6: Structure-Guided Generation

This tutorial explains how to guide molecule generation using structural constraints, such as filling in a missing piece (inpainting) or growing a molecule from a fragment (outpainting).

## Contents

1.  **Introduction**: The concept of guiding generation with a structural template.
2.  **Inpainting**: How to configure and run generation to fill in a missing portion of a molecule.
3.  **Outpainting**: How to grow a molecule from a given substructure.
4.  **3D Geometric Constraints**: How to tune the geometric constraints.

## 1. Introduction

Structure-guided generation allows you to influence the output of the diffusion model by providing a starting molecular structure. This is useful for tasks like:

*   **Inpainting**: Varying initial structures (either the whole molecule or replacing a specific part of it).
*   **Outpainting**: Extending a molecule from a given fragment.

The process involves providing a reference structure in an XYZ file and specifying which parts of the structure to modify or keep fixed. **Note that all atom indices are 0-indexed.** You can create your experiment configuration files in any directory, as the base templates are bundled with the package.

## 2. Inpainting

Inpainting allows you to vary initial structures. You provide a template molecule and specify which atoms to "mask". The diffusion model will then generate new structures for the masked atoms and connect them to the rest of the molecule, allowing you to vary specific parts or the entire structure.

### Key Inpainting Parameters

The `condition_configs` section for inpainting uses a sub-dictionary called `inpaint_cfgs` to group all specific inpainting settings.

| Parameter | Location | Description |
| :--- | :--- | :--- |
| `mol_size` | `interference` (top-level) | The expected size of the final molecule. **This should be larger than or equal to the number of atoms in the reference structure.** |
| `reference_structure_path` | `condition_configs` | **CRITICAL:** Path to your own XYZ file containing the molecule you want to inpaint. |
| `condition_component` | `condition_configs` | Component to inpaint (available choice: x, h, xh). |
| `n_retrys` | `condition_configs` | Number of retry attempts in case of bad molecules. |
| `t_retry` | `condition_configs` | Timestep to start retrying. |
| `n_frames` | `condition_configs` | Number of frames to keep for trajectory visualization. |
| `inpaint_cfgs` | `condition_configs` | **CRITICAL:** A sub-dictionary containing all settings specific to the inpainting algorithm, including `mask_node_index` and `denoising_strength`. |
| `mask_node_index` | `inpaint_cfgs` | **CRITICAL:** A list of **0-indexed** atom indices from your XYZ file that you want to remove and have the model regenerate. |
| `denoising_strength` | `inpaint_cfgs` | Controls how much noise is added to the masked region before generation. Higher values give the model more creative freedom. |


### Configuration

Here is an example of a complete configuration file for inpainting, which you can save as `my_inpaint.yaml` in your working directory:

```yaml
# my_inpaint.yaml
defaults:
  - tasks: diffusion
  - interference: gen_inpaint # Base template bundled with package
  - _self_

name: "akatsuki"
chkpt_directory: "models/edm_pretrained/"
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 600
seed: 9

interference:
  num_generate: 50
  mol_size: [50, 60] # Target size of the generated molecule
  output_path: "results/my_inpainting_run"
  condition_configs:
    reference_structure_path: "assets/BINOLCpHHH.xyz"
    condition_component: xh
    inpaint_cfgs:
      # To vary the BINOL part of the molecule, we mask the following 0-indexed atoms:
      mask_node_index: [5, 30, 31, 6, 7, 45, 8, 32, 9, 10, 33, 11, 34, 12, 35, 13, 36, 14, 15, 16, 17, 18, 37, 19, 38, 20, 39, 21, 40, 22, 23, 41, 24, 44, 25, 26, 43, 42]
      denoising_strength: 0.8
```

### Running Inpainting

Use the `MolCraftDiff generate` command with your configuration file:

```bash
MolCraftDiff generate my_inpaint
```

## 3. Outpainting

Outpainting is the process of growing a molecule from a given fragment. You provide a starting fragment, and the model will add new atoms to it.

### Key Outpainting Parameters

The `condition_configs` section for outpainting uses a sub-dictionary called `outpaint_cfgs` to group all specific outpainting settings.

| Parameter | Location | Description |
| :--- | :--- | :--- |
| `mol_size` | `interference` (top-level) | The expected size of the final molecule (fragment + generated part). |
| `reference_structure_path` | `condition_configs` | **CRITICAL:** Path to your own XYZ file containing the fragment you want to grow from. |
| `condition_component` | `condition_configs` | Component to outpaint (available choice: x, h, xh). |
| `n_retrys` | `condition_configs` | Number of retry attempts in case of bad molecules. |
| `t_retry` | `condition_configs` | Timestep to start retrying. |
| `n_frames` | `condition_configs` | Number of frames to keep for trajectory visualization. |
| `outpaint_cfgs` | `condition_configs` | **CRITICAL:** A sub-dictionary containing all settings specific to the outpainting algorithm, including `connector_dicts` and new node initialization settings. |
| `connector_dicts` | `outpaint_cfgs` | **CRITICAL:** A dictionary where keys are the **0-indexed** indices of atoms in your fragment, and values are the number of new connections to grow from that atom. |
| `t_start` | `outpaint_cfgs` | Timestep to start the generation. |
| `seed_dist` | `outpaint_cfgs` | The distance (in Å) from the connector atom to initially place the new seed nodes. Default is 2.0. |
| `min_dist` | `outpaint_cfgs` | Minimum allowed initial distance between the new node and any existing atoms in the template to avoid clashes. Default is 1.0. |
| `spread` | `outpaint_cfgs` | The standard deviation for the normal distribution when sampling the positions of the new extra nodes around the seed location. Default is 1.0. |
| `n_bq_atom` | `outpaint_cfgs` | Number of "boundary/query" atoms at the end of the template strictly used for seeding positions. Default is 0. |

> **Tuning Outpainting Outcomes**
> 
> You can explicitly control how the new atoms are initially placed in 3D space. By tuning the geometric parameters (`seed_dist`, `min_dist`, `spread`), you dictate where the diffusion process *starts* adding the new fragment. 
> 
> For example, increasing the `seed_dist` forces the model to generate the new atoms further away from the connector atom, while decreasing the `spread` tightly clusters the new generated atoms together. Adjusting these initialization settings gives you fine-grained control over the final shape and direction of the outpainted molecule.
### Configuration

Here is an example of a complete configuration file for outpainting, which you can save as `my_outpaint.yaml` in your working directory:

```yaml
# my_outpaint.yaml
defaults:
  - tasks: diffusion
  - interference: gen_outpaint # Base template bundled with package
  - _self_

name: "akatsuki"
chkpt_directory: "models/edm_pretrained/"
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 600
seed: 9

interference:
  num_generate: 50
  mol_size: [30, 40] # Target size of the generated molecule
  output_path: "results/my_outpainting_run"
  condition_configs:
    reference_structure_path: "assets/BINOLCp.xyz"
    condition_component: xh
    outpaint_cfgs:
      # To decorate BINOL-Cp with substituents at 0-indexed atoms 1, 2, and 3, each with 3 bonds:
      connector_dicts:
        1: [3]
        2: [3]
        3: [3]
      t_start: 0.8
```

### Running Outpainting

Use the `MolCraftDiff generate` command with your configuration file:

```bash
MolCraftDiff generate my_outpaint
```