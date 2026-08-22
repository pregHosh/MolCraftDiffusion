# Registering Your Own Model

> **Prerequisites:** a working integration — see [Adding New Models](../adding_new_models.md) · **You'll learn:** how to register weights, data and an example config so someone else can fetch them

An integration is not finished when generation works — it is finished when
someone else can *get* your weights. Registering makes a model appear in
`zoo list`, gives its assets fetchable names, and turns your config into one
that runs on any machine.

This page is for contributors. If you only want to *use* models, start at
[Using a pretrained model](models.md).

---

## Step 1 · Register the assets

```bash
MolCraftDiff zoo add <name> \
    --ckpt path/to/checkpoints/converted/ \
    --data path/to/data/ \
    --config configs/<name>_train.yaml \
    --config configs/<name>_generate.yaml \
    --task-type <task_type> \
    --family <family> --tag <tag> --tag <tag>
```

This hashes every file, records its size, and writes the entry into
`src/MolecularDiffusion/zoo.yaml`. Always use the command — hand-editing the
manifest is how checksums drift out of sync with the files.

---

## Step 2 · Classify what you registered

Three rules, and getting them wrong is the most common mistake:

**An asset is a directory.** Every file in one entry is fetched into the same
directory. That is what keeps a checkpoint's sidecars (`edm_stat.pkl`,
`edm_chem.pkl`, a `config.yaml`) beside it, which `cli/generate.py` relies on
when `chkpt_directory` names a directory. Do not split a checkpoint from its
sidecars across two assets.

**A sidecar reached by a different config key is its own asset.** IPDiff's
interaction prior is read via `tasks.net_cond_ckpt`, not via
`chkpt_directory`, so it is `ipdiff/ipnet` rather than a file inside
`ipdiff/pretrained`.

**Datasets are named by corpus, not by model.** A corpus belongs to nobody, so
it is keyed `data/<corpus>/<variant>` and every model that reads it references
the same key:

```yaml
# in your model's variant
data: data/qm9/graph3d          # not mymodel/data
```

Before registering any dataset, check whether the corpus is already there:

```bash
MolCraftDiff zoo list --data
```

If your model trains on a corpus another model already registered, reference
that key rather than uploading a second copy — that duplication is exactly what
corpus-first naming exists to prevent. Only register a new `data/...` entry if
the corpus or its processing is genuinely new. Run inputs — a reference
structure, one protein target, a precomputed cache keyed to your vocabulary —
go under `inputs/<model>/...` instead, because those really are yours.

---

## Step 3 · Settle the licence

`zoo add` deliberately leaves two fields as `TODO`, because they are human
judgement and must not be guessed:

- **`license`** — the upstream project's licence. Check the repo's `LICENSE`
  file *and* the GitHub Licensing API (`https://api.github.com/repos/<owner>/<repo>`
  → `.license.spdx_id`); they sometimes disagree.
- **`redistribute`** — whether we may host the converted weights. Set `false`
  and add a `reason:` when we may not; the entry then ships the upstream URL,
  the conversion script and the sha256 instead of the bytes.

How to read what you find:

| Finding | Verdict |
| :--- | :--- |
| MIT, BSD, Apache | `redistribute: true`, record the SPDX id |
| No licence at all | **all rights reserved** — `redistribute: false`, not "probably fine" |
| Non-commercial (e.g. PolyForm) | `redistribute: false`, name the licence |
| Gated dataset (HTTP 401) | `redistribute: false` |
| `NOASSERTION` from the API | the API could not classify it — read the file yourself |

Two traps, both of which have happened here:

- **Code licence ≠ data licence.** A repo can be MIT while asserting different
  terms for the corpus it trains on. Audit them separately.
- **A model's weights are not the same asset as an upstream corpus it bundles.**
  Attach each verdict to the asset it actually describes.

`MolCraftDiff zoo verify --all` fails while any `TODO` remains, so an unaudited
model cannot quietly reach a release.

---

## Step 4 · Write the example config

Copy your working `configs/<name>_generate.yaml` into
`src/MolecularDiffusion/configs/examples/` and rewrite every path-valued key to
an `${asset:...}` reference:

```yaml
chkpt_directory: ${asset:<name>/pretrained}
ase_db_path: ${asset:data/<corpus>/<variant>/<file>.db}
```

Both forms resolve, so use whichever the key needs — `${asset:m/pretrained}`
for a directory, `${asset:m/pretrained/file.ckpt}` for one file. Rewrite
*values only*, never text inside comments. Leave `output_path` a plain relative
directory: it is an output, and must never point into the asset cache.

Literal paths keep working everywhere, so this file is purely additive — your
original config stays exactly as it was.

---

## Step 5 · Prove it runs

Registration that has not been exercised is a claim, not a fact. From a
directory **outside the repository**, with a scratch cache:

```bash
export MOLCRAFT_ASSETS=/tmp/zoo-check
MolCraftDiff zoo fetch --config src/MolecularDiffusion/configs/examples/<name>_generate.yaml
MolCraftDiff generate examples/<name>_generate.yaml interference.num_generate=4
MolCraftDiff zoo verify --all
```

Ask the config's `interference:` group which keys it declares before reducing
anything — some generation modes deliberately do not accept `num_steps` or
`num_generate`, and overriding a key a group does not have produces a Hydra
error that looks like a broken config but is not.

---

## Publishing

Uploading is a separate, deliberate step; registering does not push anything.
Once the licence is settled and the example config runs, stage and upload:

```bash
python scripts/build_zoo_manifest.py stage \
    --manifest src/MolecularDiffusion/zoo.yaml --out .zoo-staging
python scripts/build_zoo_manifest.py upload \
    --manifest src/MolecularDiffusion/zoo.yaml --staging .zoo-staging
```

Staging copies only the assets marked `redistribute: true`, so an asset you
marked withheld cannot be uploaded by accident.

---

## Where to go next

- **The full integration guide** — [Adding New Models](../adding_new_models.md)
- **What users will see** — [Using a pretrained model](models.md)
