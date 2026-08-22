# Quickstart: Generate Your First Molecules

> **Prerequisites:** [Installation](../installation.md) · **You'll learn:** how to fetch a pretrained model and generate molecules without training anything · **Next:** [Tutorial 5 — Generation Overview](05_generation_overview.md) for more generation modes, or [Tutorial 0 — Data Preparation](00_data_preparation.md) to train your own

## At a Glance

| | |
| :--- | :--- |
| **Objective** | Generate 3D molecules from a pretrained model, in three commands. |
| **You need** | A working installation. No data, no training, no GPU-hours. |
| **Main command** | `MolCraftDiff zoo fetch --model kgdiff` then `MolCraftDiff generate` |
| **Success looks like** | A directory of `.xyz` files with chemically sensible geometry. |

Training a 3D molecular diffusion model takes a prepared dataset and hours of
GPU time. Before you commit to that, you can run an already-trained one.

The **model zoo** ships pretrained weights, the datasets they need, and a
ready-to-run configuration for every model in MolCraftDiffusion. You fetch a
model by name and generate — nothing to download by hand, no paths to edit.

---

## Three commands

```bash
MolCraftDiff zoo list                        # see what is available
MolCraftDiff zoo fetch --model kgdiff        # ~12 MB
MolCraftDiff generate examples/kgdiff_generate.yaml
```

That writes `.xyz` files into `generated_kgdiff/` in your current directory.

:::{note}
The zoo repositories are private during development. Set a HuggingFace token
before fetching:

```bash
export HF_TOKEN=hf_xxxxxxxx     # or: hf auth login
```
:::

You can run this from **any directory** — the example configs are installed
with the package, so `examples/kgdiff_generate.yaml` is not a file in your
folder.

---

## Step 1 · Find a model

```bash
MolCraftDiff zoo list
```

```
  MODEL                 SIZE  FAMILY / TAGS
* diffdec             7.0 MB  scaffold-decoration
                              scaffold-decoration, pocket-conditioned, r-group
  kgdiff             13.0 MB  pocket-conditioned-diffusion
                              pocket-conditioned, property-guided
  midi              101.8 MB  bond-generating-diffusion
                              unconditional, bond-generating, qm9
  ...
```

A `*` marks models you have already fetched. Narrow the list by capability:

```bash
MolCraftDiff zoo list --tag pocket-conditioned
MolCraftDiff zoo list --tag unconditional
```

Then read what a model actually does before spending time on it:

```bash
MolCraftDiff zoo info midi
```

```
midi  (bond-generating-diffusion)

  MiDi is the first model in this platform that generates the molecular graph
  itself — bond orders and formal charges are diffused jointly with the 3D
  coordinates, so a sample arrives with an explicit bond table instead of
  needing post-hoc perception.

  tags     : unconditional, bond-generating, qm9
  task_type: diffusion_midi

  variant: default
    checkpoint       midi/pretrained            92.1 MB  not fetched  MIT
    data             midi/data                   5.2 MB  not fetched  CC0-1.0
```

**Which model should you pick?** It depends on what you want to make:

| You want | Try |
| :--- | :--- |
| Novel drug-like molecules from nothing | `midi`, `flowmol_graph3d`, `gcdm` |
| Molecules that fit a protein pocket | `kgdiff`, `diffsbdd`, `ipdiff` |
| 3D conformers of a molecule you already have | `ditmc`, `loqi` |
| To grow a scaffold or link fragments | `diffdec`, `difflinker` |
| Metal-complex ligands | `ligandiff` |

---

## Step 2 · Fetch it

Fetch a whole model, or just one piece:

```bash
MolCraftDiff zoo fetch --model kgdiff        # weights + data
MolCraftDiff zoo fetch kgdiff/pretrained     # weights only
```

Check the cost before committing to a large one:

```bash
MolCraftDiff zoo fetch --model nextmol --dry-run
```

```
  fetch          nextmol/dmt                   213 MB   MIT
  BUILD LOCALLY  nextmol/mollama              1.9 GB   none declared
  total: 2.1 GB across 2 assets
```

Files land in `~/.cache/molcraft/zoo/`. Put them somewhere else if your home
directory is small:

```bash
export MOLCRAFT_ASSETS=/data/molcraft-zoo
```

Every file is checked against a checksum as it downloads, and fetching again
skips anything already correct.

:::{note}
Some models show **BUILD LOCALLY**. Their upstream projects do not grant
permission to redistribute the weights, so the zoo ships the recipe instead of
the file. Run `MolCraftDiff zoo recipe <asset>` and it prints the download URL,
the conversion command and the expected checksum.
:::

---

## Step 3 · Generate

Every model ships a runnable config:

```bash
MolCraftDiff generate examples/kgdiff_generate.yaml
```

Change any setting on the command line — no need to copy or edit the file:

```bash
MolCraftDiff generate examples/kgdiff_generate.yaml \
    interference.num_generate=100 \
    interference.output_path=my_run
```

If you forget to fetch something first, you get the command to fix it:

```
Asset 'kgdiff/pretrained' not found at
  /home/you/.cache/molcraft/zoo/kgdiff/pretrained
Fetch it with:
  MolCraftDiff zoo fetch kgdiff/pretrained   (11.0 MB, MIT)
```

To make permanent changes, copy the config into your own directory and edit it
there — a local file of the same name takes precedence over the bundled one:

```bash
MolCraftDiff zoo config kgdiff_generate.yaml .   # copy it out
MolCraftDiff generate kgdiff_generate.yaml       # your copy now wins
```

---

## Working from a config

If a colleague hands you a config and you do not know what it needs, ask the
zoo to work it out:

```bash
MolCraftDiff zoo fetch --config their_run.yaml
```

It reads the asset references out of the file and fetches exactly those — no
more, and nothing you already have.

---

## How assets are named

Three kinds of thing, three prefixes — so you can tell what something is from
its name alone:

| Prefix | What it is | Example |
| :--- | :--- | :--- |
| `data/` | a **corpus** — never named after a model | `data/qm9/graph3d` |
| `inputs/` | a **run input** — a reference structure, a single protein target | `inputs/templates` |
| *(model name)* | **weights** and their sidecars | `kgdiff/pretrained` |

Corpora are named `data/<corpus>/<variant>`: the corpus first, then how it was
processed. So the CrossDocked pockets that KGDiff, IPDiff and DiffSBDD all read
are `data/crossdocked/pockets10a` — not named after whichever model happened to
use them first. Different processings of one corpus sit together:

```
data/qm9/ase           data/qm9/graph3d          data/geom/ase
data/qm9/ase-4k        data/qm9/graph3d-4k       data/geom/graph3d-val
```

Two practical consequences:

- **Fetch a second model that shares a corpus and the download is already
  done.** `zoo fetch --model ipdiff` reuses `data/crossdocked/pockets10a` if
  kgdiff put it there.
- **A model's own weights stay under its own name**, because those genuinely
  belong to it. Only data moved.


## Command summary

| Command | What it does |
| :--- | :--- |
| `zoo list [--tag T] [--fetched]` | List models, optionally filtered |
| `zoo info <model>` | Description, variants, sizes, licences |
| `zoo fetch --model <m>` | Download everything that model needs |
| `zoo fetch <asset> [--dry-run]` | Download one asset; preview the size first |
| `zoo fetch --config <f.yaml>` | Download exactly what a config references |
| `zoo verify` | Re-check cached files against their checksums |
| `zoo path <asset>` | Print a local path, for use in shell scripts |
| `zoo recipe <asset>` | How to build an asset that cannot be redistributed |

---

## Where to go next

- **More generation modes** — [Tutorial 5: Generation Overview](05_generation_overview.md)
  covers unconditional, structure-guided and property-directed sampling.
- **Judge what you made** — [Tutorial 9: Analysis](09_analyze.md) scores
  validity, uniqueness and geometry.
- **Train your own** — start at
  [Tutorial 0: Data Preparation](00_data_preparation.md). A zoo checkpoint is
  also a good starting point for fine-tuning; see
  [Tutorial 4: Fine-tuning](04_finetuning.md).
