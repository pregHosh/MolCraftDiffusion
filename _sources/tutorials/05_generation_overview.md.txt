# Tutorial 5: Molecule Generation Overview

> **Prerequisites:** [Tutorial 1 — Training a Diffusion Model](01_training_diffusion.md) · **You'll learn:** unconditional sampling and the three generation modes · **Next:** [Tutorial 6 — Structure-Guided Generation](06_structure_guided.md)

This tutorial provides an overview of the different ways you can generate molecules using a trained model. The generation process is controlled via the `MolCraftDiff generate` command, which relies on a configuration file to specify the desired behavior.

There are three primary modes for generating molecules:

1.  **Unconditional Generation**: Generating novel molecules without any specific constraints or guidance. This is the simplest form of generation and is the focus of this tutorial.
2.  **Structure-Guided Generation**: Generating molecules from a reference structure — filling in a masked region (inpainting), building outward from a scaffold (outpainting), or softly steering a whole molecule towards a reference without freezing any atoms (SILVR). For a detailed guide on this, please see **[Tutorial 6: Structure-Guided Generation](06_structure_guided.md)**.
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

While the example above is minimal, you can control the generation process with several important parameters (overriding the defaults in `interference: gen_unconditional`).

**Top-level keys** (siblings of `defaults`):
*   `chkpt_directory`: path to the directory containing the trained model checkpoint.
*   `diffusion_steps`: number of reverse-diffusion steps. Higher can improve quality but is slower; usually left at the value the model was trained with.

**Under `interference:`**
*   `num_generate`: how many molecules to sample in one run.
*   `mol_size`: a **list of ints** controlling atom count — either a single fixed size (`[16]`) or a `[min, max]` range (`[16, 40]`) sampled per molecule. Each end is clamped to the model's `max_atom`.
*   `sampling_mode`: `"ddpm"` (default) or `"ddim"` — ddim is faster because it can skip steps.
*   `output_path`: directory to save the generated molecules.

### Running Unconditional Generation

Use the `MolCraftDiff generate` command with your config file:

```bash
MolCraftDiff generate my_gen
```