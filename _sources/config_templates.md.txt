# Configuration Templates

There are two kinds of ready-made config, and which one you want depends on
whether you are bringing your own model or running one of ours.

| | **Model examples** | **Blank templates** |
| :--- | :--- | :--- |
| Run against | a pretrained model from the zoo | your own checkpoint and data |
| Paths | already resolved — nothing to edit | you fill them in |
| Where | installed with the package | `docs/cfg_examples/` in the repository |
| Start with | `MolCraftDiff generate examples/<name>.yaml` | copy, edit, run |

New here? Use a **model example** — it runs as-is. Reach for a **template**
once you have trained something of your own.

---

## Model examples (ready to run)

Every model in the zoo ships a config that already points at its weights and
data, so it runs unedited on any machine:

```bash
MolCraftDiff zoo fetch --model kgdiff
MolCraftDiff generate examples/kgdiff_generate.yaml
```

`examples/` is inside the installed package, not a folder in your working
directory — so this works from anywhere. List what is there, and copy one out
if you want to keep changes:

```bash
MolCraftDiff zoo config                              # list them
MolCraftDiff zoo config kgdiff_generate.yaml .       # copy into the current dir
```

A local file of the same name takes precedence over the bundled one, so an
edited copy is picked up automatically.

For one-off changes, skip copying and override on the command line:

```bash
MolCraftDiff generate examples/kgdiff_generate.yaml \
    interference.num_generate=100 interference.output_path=my_run
```

See the [model zoo quickstart](tutorials/quickstart_model_zoo.md) for the full
workflow, and [Model Architectures](architectures.md) for choosing a model.

---

## Blank templates (bring your own model)

These live in the repository under `docs/cfg_examples/`. Copy the relevant
file, fill in your checkpoint and data paths, and run it. The packaged Hydra
defaults they build on are under `src/MolecularDiffusion/configs/`.

### Training

| Template | Purpose |
| :--- | :--- |
| `docs/cfg_examples/train_diffusion.yaml` | Train an EDM diffusion model from scratch. |
| `docs/cfg_examples/finetune_diffusion.yaml` | Fine-tune from a pretrained diffusion checkpoint. |
| `docs/cfg_examples/train_regressor.yaml` | Train a property regressor. |
| `docs/cfg_examples/train_guidance.yaml` | Train a guidance (property) model. |

```bash
MolCraftDiff train docs/cfg_examples/train_diffusion
```

### Generation

| Template | Mode (`interference.task_type`) |
| :--- | :--- |
| `docs/cfg_examples/gen_unconditional.yaml` | `unconditional` |
| `docs/cfg_examples/gen_cfg.yaml` | `cfg` — classifier-free guidance |
| `docs/cfg_examples/gen_gradient_guidance.yaml` | `gg` — gradient guidance |
| `docs/cfg_examples/gen_hybrid_cfg_gg.yaml` | `cfggg` — hybrid CFG + GG |
| `docs/cfg_examples/gen_inpaint.yaml` | `inpaint` — structure inpainting |
| `docs/cfg_examples/gen_outpaint.yaml` | `outpaint` — fragment extension |

```bash
MolCraftDiff generate docs/cfg_examples/gen_cfg
```

:::{note}
The structure-guided templates (`gen_inpaint`, `gen_outpaint`) need a
reference `.xyz`. Three are shipped with the zoo — fetch them with
`MolCraftDiff zoo fetch inputs/templates` — or point
`condition_configs.reference_structure_path` at your own.
:::

---

Config-driven commands accept Hydra-style dotted overrides, e.g.
`MolCraftDiff generate docs/cfg_examples/gen_cfg interference.num_generate=200`.
See [Tutorials](tutorials/index.md) for end-to-end walkthroughs.
