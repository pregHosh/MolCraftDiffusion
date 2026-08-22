# Model Zoo

Pretrained weights, the datasets they were trained on, and a ready-to-run
configuration for every model in MolCraftDiffusion — fetched by name, with
nothing to download by hand and no paths to edit.

```bash
MolCraftDiff zoo list                        # what models are there
MolCraftDiff zoo fetch --model kgdiff        # get one
MolCraftDiff generate examples/kgdiff_generate.yaml
```

:::{note}
The zoo repositories are private during development. Set a HuggingFace token
before fetching:

```bash
export HF_TOKEN=hf_xxxxxxxx     # or: hf auth login
```
:::

---

## What you can do with it

Four workflows, roughly in order of how much work they ask of you. Most people
start at the top and only move down when they need to.

### 1 · Run a pretrained model as-is

No data, no training. Fetch a model and generate.

```bash
MolCraftDiff zoo fetch --model midi
MolCraftDiff generate examples/midi_generate.yaml
```

→ [Using a pretrained model](models.md)

### 2 · Run a pretrained model on *your* input

Structure-guided models need something to work from — a protein pocket, a
scaffold, a reference geometry. Keep the zoo's weights, swap the input for
yours:

```bash
MolCraftDiff generate examples/kgdiff_generate.yaml \
    interference.pocket_db=my_pockets.db \
    interference.output_path=my_run
```

The zoo ships example inputs under `inputs/` so you can see the expected format
before preparing your own.

→ [Using a pretrained model § Bring your own input](models.md#bring-your-own-input)

### 3 · Fine-tune a zoo checkpoint on your chemistry

A pretrained checkpoint is a much better starting point than random weights.
Fetch the weights, point a training config at your own dataset, and resume:

```bash
MolCraftDiff zoo fetch midi/pretrained
MolCraftDiff train my_finetune          # resume_ckpt: ${asset:midi/pretrained/...}
```

→ [Tutorial 4: Fine-tuning](../tutorials/04_finetuning.md)

### 4 · Train from scratch on a zoo corpus

The datasets are fetchable on their own — useful for reproducing a published
result, or for training a new architecture on an established benchmark without
re-running anyone's preprocessing.

```bash
MolCraftDiff zoo fetch data/qm9/graph3d-4k
MolCraftDiff train my_train             # ase_db_path: ${asset:data/qm9/...}
```

→ [Using zoo datasets](data.md)

:::{tip}
These compose. A common path is **fetch a corpus → adapt an example train
config → train → adapt the matching example generation config to point at your
new checkpoint.** Every example config is a working starting point for exactly
that, and `MolCraftDiff zoo config <name> .` copies one out to edit.
:::

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
  belong to it. Only data is shared.

---

## Where things land

Files are cached in `~/.cache/molcraft/zoo/`. Put them somewhere else if your
home directory is small:

```bash
export MOLCRAFT_ASSETS=/data/molcraft-zoo
```

Every file is checked against a checksum as it downloads, and fetching again
skips anything already correct. The cache is shared across projects — fetch a
corpus once and every config on the machine that references it is satisfied.

---

## Command summary

| Command | What it does |
| :--- | :--- |
| `zoo list [--tag T] [--family F] [--fetched]` | List models, optionally filtered |
| `zoo list --data` | List data corpora and run inputs, and who reads them |
| `zoo info <model\|asset>` | Description, variants, sizes, licences |
| `zoo fetch --model <m>` | Download everything that model needs |
| `zoo fetch <asset> [--dry-run]` | Download one asset; preview the size first |
| `zoo fetch --config <f.yaml>` | Download exactly what a config references |
| `zoo path <asset>` | Print a local path, for use in shell scripts |
| `zoo config [<name>] [<dest>]` | List bundled example configs, or copy one out |
| `zoo verify [--all]` | Re-check cached files against their checksums |
| `zoo recipe <asset>` | How to build an asset that cannot be redistributed |
| `zoo bundle` | Package the cache for an offline machine |
| `zoo add <name> ...` | Register your own model — see [Registering](registering.md) |

```{toctree}
:hidden:
:maxdepth: 1

models
data
registering
```
