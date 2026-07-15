# Model Architectures

MolCraftDiffusion ships several backbone families as first-class options. You
select one by choosing a **task config** (`MolCraftDiff train <config>`); the
config's `_target_` factory builds the matching model, so adding or swapping an
architecture never touches the core engine.

## Available architectures

| Task config (`configs/tasks/…`) | `task_type` | Model | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion.yaml` | `diffusion` | EDM (E(n)-equivariant diffusion, EGCL backbone) | **Default.** Cartesian-space DDPM; the checkpoints on Hugging Face use this. |
| `diffusion_extraf.yaml` | `diffusion` | EDM + extra atom features | Same backbone, feature-aware conditioning. |
| `diffusion_pretrained.yaml` | `diffusion` | EDM | Fine-tuning from a pretrained EDM checkpoint. |
| `diffusion_egt.yaml` | `diffusion` | EGT (equivariant graph transformer) | Transformer backbone via `tasks_egt`. |
| `diffusion_gfmdiff.yaml` | `diffusion` | GFMDiff | Geometric full-molecule diffusion (`tasks_gfmdiff`). |
| `diffusion_tabasco.yaml` | `diffusion_tabasco` | TABASCO | Flow-matching architecture. |
| `diffusion_adit.yaml` | `diffusion_adit` | ADiT / DiT-based LDM | Latent diffusion with a DiT denoiser. |
| `diffusion_difflinker.yaml` | `diffusion_difflinker` | DiffLinker | Fragment linking / linker design. |
| `vae_transformer.yaml` | `vae_transformer` | GeoLDM VAE (transformer enc/dec) | Trains the autoencoder for latent diffusion. |
| `vae_equiformer.yaml` | `vae_equiformer` | GeoLDM VAE (Equiformer enc/dec) | Equivariant autoencoder variant. |
| `pharmacophore.yaml` | `pharmacophore` | Pharmacophore-conditioned dynamics | Requires `open3d`. |
| `ssl3d_egcl.yaml` / `ssl3d_egt.yaml` / `ssl3d_equiformer.yaml` / `ssl3d_esen.yaml` | `ssl3d_*` | Self-supervised 3D pretraining | Prefix-matched to the `ssl3d` task family. |
| `regression.yaml` / `guidance.yaml` | `regression` / `guidance` | Property predictor / guidance head (EGCL backbone) | **Default.** Used for property prediction and gradient guidance. |
| `regression_esen.yaml` / `guidance_esen.yaml` | `regression` / `guidance` | Property predictor / guidance head (eSEN backbone) | Same task classes as above, eSEN backbone. |
| `regression_equiformer.yaml` | `regression` | Property predictor (EquiformerV2 backbone) | No `guidance`/diffusion config exists yet for this backbone — only `regression` and `ssl3d_equiformer`. |

ShEPhERD is integrated as a scoring/architecture module (`modules/models/shepherd_arch/`,
`utils/shepherd_score/`) used by the guidance and `analyze metrics --metrics shepherd`
paths rather than as a standalone training config.

## How the wiring works

Each task config declares a `_target_` factory and a `task_type`. Hydra
instantiates the factory, which builds the model plus its Lightning task; the
engine trains it. `task_type` is dispatched (exact match, then prefix) by
`runmodes/train/eval.py`'s `TASK_REGISTRY` to pick the right validation metric
and whether generative sampling runs during evaluation.

## Adding a new architecture

1. Drop a sub-package under `src/MolecularDiffusion/modules/models/<arch>/`.
2. Add a task module under `modules/tasks/` if it needs a new training objective.
3. Add a `configs/tasks/<arch>.yaml` pointing `_target_` at your factory.
4. Register a `task_type` entry in `runmodes/train/eval.py` (or reuse a prefix
   family like `diffusion`, `vae`, `ssl3d`).
5. Add a `tests/models/test_<arch>_compat.py` compatibility test.

No change to `core/` is required — that separation is the point of the layered
design.
