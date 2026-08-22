# Using Zoo Datasets

> **Prerequisites:** [Using a pretrained model](models.md) · **You'll learn:** how to fetch a corpus by name and train on it · **Next:** [Tutorial 1 — Training a Diffusion Model](../tutorials/01_training_diffusion.md)

The zoo's datasets are fetchable on their own, not just as a side effect of
fetching a model. That is useful when you want to reproduce a published result,
train a new architecture on an established benchmark, or simply get a small
subset to develop against without preprocessing anything yourself.

---

## Step 1 · Find a corpus

```bash
MolCraftDiff zoo list --data
```

```
  CORPUS                                  SIZE  USED BY
  data/chembl3d/stereo-200            868.0 KB  loqi, loqi_flow
* data/crossdocked/pockets10a           1.3 MB  diffsbdd, ipdiff, kgdiff
  data/geom/ase                       618.8 MB  -
  data/qm9/ase                        253.0 MB  -
  data/qm9/ase-4k                       7.4 MB  -
  data/qm9/graph3d                    191.7 MB  jodo
  data/qm9/graph3d-4k                   5.2 MB  midi
  data/zinc/difflinker                666.1 MB  difflinker
  inputs/templates                      4.7 KB  -
  ...
```

**USED BY** lists the models that read it — which is the fastest way to find a
corpus in the right format for the architecture you have in mind. A `-` means
no bundled example config points at it; the corpus is still perfectly usable.

Read the terms and the file list before downloading a large one:

```bash
MolCraftDiff zoo info data/qm9/graph3d
```

### Choosing a variant

Corpora are named `data/<corpus>/<variant>`, and the variant tells you how it
was processed. The three things it encodes:

| In the name | Meaning |
| :--- | :--- |
| `ase` | an ASE database — coordinates and atom types only |
| `graph3d` | explicit bond orders and formal charges as well |
| a number (`-4k`, `-300`, `-400`) | a small subset, for smoke tests and development |

So `data/qm9/ase` and `data/qm9/graph3d` are the same molecules prepared for
different model families, and `data/qm9/graph3d-4k` is a 4,000-molecule slice
of the second. **Match the variant to your model's `data_type`** — a
bond-generating model cannot train on an `ase` variant, because the bonds are
not there. [Model Architectures](../architectures.md) lists what each model
needs.

---

## Step 2 · Fetch it

```bash
MolCraftDiff zoo fetch data/qm9/graph3d-4k --dry-run    # check the size first
MolCraftDiff zoo fetch data/qm9/graph3d-4k
```

Ask where it landed:

```bash
MolCraftDiff zoo path data/qm9/graph3d-4k
```

```
/home/you/.cache/molcraft/zoo/data/qm9/graph3d-4k
```

`zoo path` prints nothing but the path, so it composes in a shell:

```bash
ls "$(MolCraftDiff zoo path data/qm9/graph3d-4k)"
```

---

## Step 3 · Point a training config at it

Use an `${asset:...}` reference in place of the path. It resolves to the cache
location at run time, so the config works unchanged on any machine that can
fetch the corpus.

```yaml
defaults:
  - data: midi_qm9_dataset
  - tasks: diffusion_midi
  - trainer: default
  - logger: default
  - _self_

data:
  ase_db_path: ${asset:data/qm9/graph3d-4k/midi_smoke.db}
  root: ./work            # your directory -- see the note below
  batch_size: 4
```

```bash
MolCraftDiff train my_train
```

:::{important}
**`root:` is not the dataset location — it is a working directory.** The
dataset is read from `ase_db_path`; `root` is where the platform *writes* the
processed cache (`processed_data_<tag>.pt`, and `chunks_<tag>/` for chunked
datasets). Point it at a directory of your own, never at the asset cache: the
cache is checksum-verified, and mixing derived files into it makes
`MolCraftDiff zoo verify` report the corpus as modified.
:::

The fastest way to a working config is to start from one that already runs:

```bash
MolCraftDiff zoo config                              # list the examples
MolCraftDiff zoo config midi_generate.yaml .         # copy one out
```

The bundled data groups under `src/MolecularDiffusion/configs/data/` are the
other half of that — `defaults: - data: midi_qm9_dataset` pulls in the atom
vocabulary, feature choices and collation that corpus expects, and you override
only `ase_db_path`, `root` and `batch_size` on top.

---

## Step 4 · Generate from what you trained

Training writes a checkpoint under your run directory. To sample from it, take
the model's example generation config and point it at your checkpoint instead
of the zoo's:

```bash
MolCraftDiff zoo config midi_generate.yaml .
# edit midi_generate.yaml -> chkpt_directory: logs/my_train/checkpoints/last.ckpt
MolCraftDiff generate midi_generate
```

or without editing anything:

```bash
MolCraftDiff generate examples/midi_generate.yaml \
    chkpt_directory=logs/my_train/checkpoints/last.ckpt \
    interference.output_path=my_samples
```

Everything else in the example config — the atom vocabulary, the sampling
schedule, the output handling — stays valid, because it describes the
architecture rather than the particular weights. That is the whole loop:
**fetch a corpus → adapt an example train config → train → adapt the matching
example generation config.**

---

## Run inputs

Assets under `inputs/` are not training corpora — they are single files a
generation run consumes: a reference geometry, one protein target, a set of
fragments. They are shipped so you can see the expected format before preparing
your own:

```bash
MolCraftDiff zoo fetch inputs/templates
ls "$(MolCraftDiff zoo path inputs/templates)"
```

See [Bring your own input](models.md#bring-your-own-input) for which key each
model reads them through.

---

## Notes on licensing

Datasets carry the terms of the corpus they came from, which are not the same
as the terms of the model that trains on them. `zoo list --data` marks anything
that cannot be redistributed as **(build locally)**, and `zoo recipe <asset>`
prints how to obtain it from the original source. `zoo info` shows the licence
for any single asset.

---

## Where to go next

- **Train on it properly** — [Tutorial 1: Training a Diffusion Model](../tutorials/01_training_diffusion.md)
- **Prepare a corpus of your own** — [Tutorial 0: Data Preparation](../tutorials/00_data_preparation.md)
- **Start from pretrained weights instead** — [Tutorial 4: Fine-tuning](../tutorials/04_finetuning.md)
- **Publish your own model and data** — [Registering your own model](registering.md)
