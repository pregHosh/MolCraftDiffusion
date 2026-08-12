# Configuration Templates

Ready-to-use template configs live in the repository under `docs/cfg_examples/`.
Copy the relevant file, fill in your checkpoint and data paths, and run it. The
full packaged Hydra defaults these templates build on are under
`src/MolecularDiffusion/configs/`.

## Training

| Template | Purpose |
| :--- | :--- |
| `docs/cfg_examples/train_diffusion.yaml` | Train an EDM diffusion model from scratch. |
| `docs/cfg_examples/finetune_diffusion.yaml` | Fine-tune from a pretrained diffusion checkpoint. |
| `docs/cfg_examples/train_regressor.yaml` | Train a property regressor. |
| `docs/cfg_examples/train_guidance.yaml` | Train a guidance (property) model. |

```bash
MolCraftDiff train docs/cfg_examples/train_diffusion
```

## Generation

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

Config-driven commands accept Hydra-style dotted overrides, e.g.
`MolCraftDiff generate docs/cfg_examples/gen_cfg interference.num_generate=200`.
See [Model Architectures](architectures.md) for choosing a training backbone and
[Tutorials](tutorials/index.md) for end-to-end walkthroughs.
