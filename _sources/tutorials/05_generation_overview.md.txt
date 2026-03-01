# Tutorial 5: Molecule Generation Overview

This tutorial provides an overview of the different ways you can generate molecules using a trained model. The generation process is controlled via the `MolCraftDiff generate` command, which relies on a configuration file to specify the desired behavior.

There are three primary modes for generating molecules:

1.  **Unconditional Generation**: Generating novel molecules without any specific constraints or guidance. This is the simplest form of generation and is the focus of this tutorial.
2.  **Structure-Guided Generation**: Generating molecules by building upon a predefined chemical scaffold (a process known as outpainting). For a detailed guide on this, please see **[Tutorial 6: Structure-Guided Generation](06_structure_guided.md)**.
3.  **Property-Directed Generation**: Guiding the generation process to produce molecules that are optimised for specific chemical or physical properties (e.g., high solubility, specific energy levels). For a detailed guide on this, please see **[Tutorial 7: Property-Directed Generation](07_property_directed.md)**.

---

## Unconditional Generation

Unconditional generation is the most straightforward way to sample molecules from your trained diffusion model. It explores the chemical space the model has learned without steering it in any particular direction.

### How It Works

You use the `MolCraftDiff generate` command, providing it with a configuration file that specifies the model to use and the number of molecules to generate. You can create your configuration file in any directory.

-   **Entry Point**: `MolCraftDiff generate`
-   **Key Configuration Components**: The generation process uses base templates like `tasks: diffusion` and `interference: gen_unconditional` which are bundled with the package.

### Example Configuration

A typical configuration for unconditional generation looks like this. You can save this as `my_gen.yaml` in your working directory. Note that more advanced options can be specified to control the generation process.

```yaml
# my_gen.yaml
defaults:
  - tasks: diffusion
  - interference: gen_unconditional
  - _self_

name: "akatsuki"
chkpt_directory: "models/edm_pretrained/"
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 600
seed: 9

interference:
  num_generate: 100
  mol_size:  [16]
  output_path: generated_mol
```

### Key Generation Parameters

While the example above is minimal, you can control the generation process with several important parameters (overriding the defaults in `interference: gen_unconditional`):

*   `diffusion_steps`: (Integer) The number of steps to run the reverse diffusion process. A higher number can lead to better quality molecules but increases generation time. This usually defaults to the value the model was trained with.
*   `interference.sampling_mode`: (String) The sampling algorithm to use. Common choices are `"ddpm"` (Denoising Diffusion Probabilistic Models) and `"ddim"` (Denoising Diffusion Implicit Models). ddim is generally faster as it can skip steps.
*   `interference.mol_size`: (Integer) Specifies the maximum number of atoms for the molecules you want to generate. This should typically not exceed the maximum number of atoms the model was trained on.
*   `interference.num_generate`: (Integer) The total number of molecules you wish to generate in one run.
*   `chkpt_directory`: (String) Path to the directory containing the trained model checkpoint.
*   `interference.output_path`: (String) Where to save the output file.

### Running Unconditional Generation

Use the `MolCraftDiff generate` command with your config file:

```bash
MolCraftDiff generate my_gen
```