# Using a Pretrained Model

> **Prerequisites:** [Installation](../installation.md) · **You'll learn:** how to find, fetch and run a pretrained model without training anything · **Next:** [Using zoo datasets](data.md), or [Tutorial 5 — Generation Overview](../tutorials/05_generation_overview.md)

Training a 3D molecular diffusion model takes a prepared dataset and hours of
GPU time. Before you commit to that, run one that is already trained.

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
  itself -- bond orders and formal charges are diffused jointly with the 3D
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

[Model Architectures](../architectures.md) has the full comparison.

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

:::{note}
Some assets show **BUILD LOCALLY**. Their upstream projects do not grant
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

That writes `.xyz` files into `generated_kgdiff/` in your current directory.
You can run it from **any directory** — the example configs are installed with
the package, so `examples/kgdiff_generate.yaml` is not a file in your folder.

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
MolCraftDiff zoo config                          # list what is available
MolCraftDiff zoo config kgdiff_generate.yaml .   # copy it out
MolCraftDiff generate kgdiff_generate.yaml       # your copy now wins
```

Each example config is commented with what the model conditions on, what its
knobs do, and any known limitation of the bundled weights — worth reading once
before you start changing values.

---

## Bring your own input

Structure-guided models need something to work from, and the zoo ships an
example of each so you can see the expected format. Keep the pretrained
weights and point the input key at your own file:

| Model kind | Key to override | Example input shipped |
| :--- | :--- | :--- |
| Pocket-conditioned | `interference.pocket_db` | `data/crossdocked/pockets10a` |
| Fragment linking | `interference.sample_input` | `inputs/difflinker/fragments` |
| Inpaint / outpaint / SILVR | `condition_configs.reference_structure_path` | `inputs/templates`, `inputs/silvr/reference` |
| Conformer generation | `interference.sample_input` | `inputs/loqi/stereo-ref` |

```bash
MolCraftDiff zoo fetch inputs/templates
MolCraftDiff zoo path inputs/templates          # look at what is in there

MolCraftDiff generate examples/silvr_generate.yaml \
    condition_configs.reference_structure_path=my_fragment.xyz
```

Pocket-conditioned models read a prepared ASE database rather than a raw PDB;
[Tutorial 0 — Data Preparation](../tutorials/00_data_preparation.md) covers
building one.

---

## Working from someone else's config

If a colleague hands you a config and you do not know what it needs, ask the
zoo to work it out:

```bash
MolCraftDiff zoo fetch --config their_run.yaml
```

It reads the asset references out of the file and fetches exactly those — no
more, and nothing you already have.

---

## Offline and shared machines

On a cluster node with no internet, package the cache on a machine that has
one and copy it over:

```bash
MolCraftDiff zoo bundle --out zoo_bundle       # on the connected machine
# copy zoo_bundle.tar.gz across, unpack it, then:
export MOLCRAFT_ASSETS=/path/to/unpacked
MolCraftDiff zoo verify --all
```

Pointing `MOLCRAFT_ASSETS` at a shared directory also lets a whole group share
one copy of the large corpora.

---

## Where to go next

- **Train on the same data** — [Using zoo datasets](data.md)
- **More generation modes** — [Tutorial 5: Generation Overview](../tutorials/05_generation_overview.md)
- **Judge what you made** — [Tutorial 9: Analysis](../tutorials/09_analyze.md)
- **Fine-tune from a zoo checkpoint** — [Tutorial 4: Fine-tuning](../tutorials/04_finetuning.md)
