# Model Architectures

MolCraftDiffusion ships several backbone families as first-class options. You
select one by choosing a **task config** (`MolCraftDiff train <config>`); the
config's `_target_` factory builds the matching model, so adding or swapping an
architecture never touches the core engine.

Packaged task configs live in `src/MolecularDiffusion/configs/tasks/`. The tables below list one row per **distinct
model**; configs that differ only in hyperparameters or a starting checkpoint
are not listed separately.

## 1. De novo 3D generation

Whole-molecule generators trained on a plain 3D molecule dataset. No
conditioning input beyond the atom count.

| Task config | `task_type` | Model | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion.yaml` | `diffusion` | EDM (E(n)-equivariant diffusion, EGCL backbone) | **Default.** Cartesian-space DDPM; the checkpoints on Hugging Face use this. |
| `diffusion_egt.yaml` | `diffusion` | EGT (equivariant graph transformer) | Transformer backbone via `tasks_egt`. |
| `diffusion_gfmdiff.yaml` | `diffusion` | GFMDiff | Geometric full-molecule diffusion (`tasks_gfmdiff`). |
| `diffusion_painn.yaml` | `diffusion` | PaiNN (scalar+vector message passing) | OM-Diff's `EquivNet` backbone under the default EDM objective (`tasks_painn`). |
| `diffusion_tabasco.yaml` | `diffusion_tabasco` | TABASCO | Flow matching; simplified, fast, tuned for physical quality. |
| `diffusion_equifm.yaml` | `diffusion_equifm` | EquiFM (flow matching, EGNN backbone) | Hybrid probability transport: an OT path for coordinates regularised by an **equivariant OT coupling** (Kabsch rotation + `scipy` linear-sum assignment between prior and data), and a VP path for atom types and charges. Reuses the GeoLDM EGNN; fixed-step RK4 sampler, so `num_steps` is honoured. Upstream released **sampling code only**, so the training objective is reconstructed from the paper's Alg. 1/3 and is *not* the authors' released objective. The converted QM9 checkpoint must use `discrete_path: HB_path`; use `OT_path` for locally trained models. Bond-free and unconditional. |
| `diffusion_flowmol.yaml` | `diffusion_flowmol` | FlowMol (SE(3)-equivariant GVP) | Flow matching for coordinates, atom types, and formal charges. **Bond-free variant only:** bonds are not modelled or generated; graph edges serve geometric message passing only. Needs the `[flowmol]` extra (DGL). |

## 2. Latent-space diffusion (two-stage)

Train an autoencoder first, then diffuse in its latent space. Each family needs
**both** of its configs, VAE first.

| Family | Stage 1 — VAE | Stage 2 — diffusion | Notes |
| :--- | :--- | :--- | :--- |
| GeoLDM | `vae_geoldm.yaml` (`vae_geoldm`) | `diffusion_geoldm.yaml` (`diffusion_geoldm`) | Equivariant point-cloud autoencoder + latent DDPM. |
| ADiT | `vae_transformer.yaml` (`vae_transformer`) or `vae_equiformer.yaml` (`vae_equiformer`) | `diffusion_adit.yaml` (`diffusion_adit`) | DiT denoiser over the latent. Two encoder choices: plain transformer or Equiformer. |

## 3. Conditional and structure-aware generation

Generation steered by an external input — a shape, a pocket, a set of fragments,
a pharmacophore. These need **paired** data (the condition alongside the
molecule); a plain molecule dataset is not enough.

| Task config | `task_type` | Conditioned on | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion_diffsmol.yaml` | `diffusion_diffsmol` | Molecular **shape** | DiffSMol (UniTransformerO2 GVP). Continuous DDPM on coordinates + D3PM categorical diffusion on atom types, conditioned on an equivariant `(128,3)` surface-shape latent with classifier-free guidance. Bond-free; heavy atoms only. Needs an offline shape cache (`[shape]` extra). |
| `diffusion_diffsbdd.yaml` | `diffusion_diffsbdd` | **Protein pocket** | DiffSBDD (flat-scatter EGNN). The reference **structure-based (SBDD)** backbone: E(3)-equivariant DDPM on coordinates + 10-class element one-hot, heavy atoms only, bond-free. Two modes — `pocket_conditioning` (pocket as clean context, **recommended**) and `joint` (ligand+pocket diffused together; use `diffusion_diffsbdd_joint_moad.yaml`, the CrossDocked joint weights are defective upstream) — plus RePaint `inpaint()` in both for scaffold hopping (`src/MolecularDiffusion/configs/interference/gen_diffsbdd_inpaint.yaml`). Ligand size comes from a pocket-conditioned 2D histogram. Generation goes through `DiffSBDDPocketGenerator` (`gen_diffsbdd_pocket.yaml`), not `GenerativeFactory`. The two modes share identical state-dict keys and shapes, so a `mode_id` buffer raises on a mismatched checkpoint. |
| `diffusion_diffpharma.yaml` | `diffusion_diffpharma` | **Protein pocket** + pharmacophore particles | DiffPharma (EGNN over 3 parallel interaction graphs). A **structure-based (SBDD)** backbone: generates a ligand *inside a given pocket*. Four node sets — ligand, full-atom pocket, `interh`, `interhp` — and **only the ligand is noised**, so it cannot train on a ligand-only dataset. Ligand size comes from a 2D histogram conditioned on pocket size. Bond-free. Novel pockets from raw PDB+SDF need the `[bio]` extra. |
| `diffusion_pmdm.yaml` | `diffusion_pmdm` | **Protein pocket** | PMDM (dual EGNN + SchNet encoders, ligand↔pocket cross-attention). A **structure-based (SBDD)** backbone, like DiffPharma but pocket-only. Two node sets — ligand and full-atom pocket — and **only the ligand is noised**; the pocket is fixed context, so a ligand-only dataset cannot train it. Continuous DDPM (sigmoid schedule, T=1000) on coordinates + atom types, with a global (6 Å) and a local (3 Å) branch over the joined cloud. Bond-free: the real bonds are discarded before the model sees them. Generation goes through `PMDMPocketGenerator` (`gen_pmdm_pocket.yaml`), not `GenerativeFactory`, since `sample()` has no channel for "which pocket". Inpainting/linker sampling is not ported. |
| `diffusion_kgdiff.yaml` | `diffusion_kgdiff` | **Protein pocket** | KGDiff (SE(3)-equivariant attention transformer, kNN-32 graph). **Structure-based (SBDD)**, ligand + full-atom 10 Å pocket, only the ligand noised. Continuous DDPM on coordinates + D3PM categorical diffusion on a 13-class `(element, is_aromatic)` vocabulary; centred on the **pocket centroid, not zero-CoM**. Bond-free. Its distinguishing feature: a per-atom affinity head trained alongside the denoiser and used as **its own classifier guide** at sampling time (`guide_mode: joint`), so one checkpoint both denoises and steers towards higher predicted affinity (`wo` is the unguided ablation). Ligand size comes from a static pocket-extent table, so the prior needs no training data. **Also runs [TargetDiff](https://arxiv.org/abs/2303.03543)** — identical backbone and defaults: set `use_classifier_guide=false`, sample with `guide_mode=wo`. Generation goes through `KGDiffPocketGenerator` (`gen_kgdiff_pocket.yaml`), not `GenerativeFactory`. |
| `diffusion_ipdiff.yaml` | `diffusion_ipdiff` | **Protein pocket** | IPDiff. Same TargetDiff lineage as KGDiff — identical backbone, data pipeline, 13-class vocabulary and D3PM/DDPM objective — but it conditions on a **frozen pretrained interaction prior (IPNet)** instead of gradient guidance. IPNet's 128-d per-atom features are concatenated into the token embeddings (*prior conditioning*) and drive a learned shift `k_t·f(h_bap, t)` added to **both the forward noising process and the reverse posterior** (*prior shifting*) — so the conditioning is baked into training, not applied at sampling time. `h_bap` is recomputed from the current predicted x0 at every reverse step. No classifier, no CFG, no property head. The IPNet weights ship with the port and are **mandatory** (`net_cond_ckpt`). Bond-free. Generation goes through `IPDiffPocketGenerator` (`gen_ipdiff_pocket.yaml`). The released checkpoint is carbon-saturated (93% C versus 64% in reference ligands). |
| `diffusion_difflinker.yaml` | `diffusion_difflinker` | **Fragments** to join | DiffLinker. Linker design: generates the connecting atoms between held-fixed fragments. |
| `pharmacophore.yaml` | `diffusion_pharmacophore` | **Pharmacophore** points | Ligand-derived pharmacophore conditioning (no protein). Requires `open3d`. |

## 4. Property prediction and guidance

Not generators. `regression` predicts a property; `guidance` exposes the same
head as a gradient signal to steer a diffusion sampler.

| Task config | `task_type` | Backbone |
| :--- | :--- | :--- |
| `regression.yaml` / `guidance.yaml` | `regression` / `guidance` | EGCL. **Default.** |
| `regression_esen.yaml` / `guidance_esen.yaml` | `regression` / `guidance` | eSEN. |
| `regression_equiformer.yaml` | `regression` | EquiformerV2. No `guidance` config for this backbone yet. |

## 5. Self-supervised pretraining

Pretrain a backbone on unlabeled 3D structures, then fine-tune for regression or
guidance.

| Task config | `task_type` | Backbone |
| :--- | :--- | :--- |
| `ssl3d_egcl.yaml` | `ssl3d` | EGCL |
| `ssl3d_egt.yaml` | `ssl3d` | EGT |
| `ssl3d_esen.yaml` | `ssl3d` | eSEN |
| `ssl3d_equiformer.yaml` | `ssl3d_equiformer` | EquiformerV2 |

## References

Backbones and objectives integrated here are based on the following work.

- **EDM** — Hoogeboom, Satorras, Vignac & Welling. *Equivariant Diffusion for
  Molecule Generation in 3D.* ICML 2022. [arXiv:2203.17003](https://arxiv.org/abs/2203.17003)
- **EGT** — Vignac et al. *MiDi: Mixed Graph and 3D Denoising Diffusion for
  Molecule Generation.* ECML PKDD 2023. [arXiv:2302.09048](https://arxiv.org/abs/2302.09048)
- **GFMDiff** — Xu et al. *Geometric-Facilitated Denoising Diffusion Model for
  3D Molecule Generation.* AAAI 2024. [arXiv:2401.02683](https://arxiv.org/abs/2401.02683)
- **PaiNN / OM-Diff** — the backbone is the `EquivNet` of *OM-Diff: Inverse-design
  of organometallic catalysts with guided equivariant denoising diffusion*, 2024.
  [doi:10.26434/chemrxiv-2024-882hh](https://doi.org/10.26434/chemrxiv-2024-882hh) — itself a variant
  of **PaiNN**: Schütt, Unke & Gastegger, *Equivariant message passing for the
  prediction of tensorial properties and molecular spectra*, ICML 2021
  ([arXiv:2102.03150](https://arxiv.org/abs/2102.03150)).
- **TABASCO** — Vonessen, Harris, Cretu & Liò. *TABASCO: A Fast, Simplified
  Model for Molecular Generation with Improved Physical Quality.* 2025.
  [arXiv:2507.00899](https://arxiv.org/abs/2507.00899)
- **FlowMol** — Dunn & Koes. *Mixed Continuous and Categorical Flow Matching
  for 3D De Novo Molecule Generation.* 2024.
  [arXiv:2404.19739](https://arxiv.org/abs/2404.19739)
- **EquiFM** — Song, Gong, Xu, Cao, Lan, Ermon, Zhou & Ma. *Equivariant Flow
  Matching with Hybrid Probability Transport for 3D Molecule Generation.*
  NeurIPS 2023. [arXiv:2312.07168](https://arxiv.org/abs/2312.07168)
- **GeoLDM** — Xu, Powers, Dror, Ermon & Leskovec. *Geometric Latent
  Diffusion Models for 3D Molecule Generation.* ICML 2023. [arXiv:2305.01140](https://arxiv.org/abs/2305.01140)
- **ADiT** — Joshi et al. *All-atom Diffusion Transformers: Unified generative
  modelling of molecules and materials.* 2025. [arXiv:2503.03965](https://arxiv.org/abs/2503.03965) —
  built on the **DiT** backbone of Peebles & Xie, *Scalable Diffusion Models with
  Transformers*, ICCV 2023 ([arXiv:2212.09748](https://arxiv.org/abs/2212.09748)).
- **DiffSMol** — Chen, Peng, Zhai, Adu-Ampratwum & Ning. *Generating 3D Binding
  Molecules Using Shape-Conditioned Diffusion Models with Guidance.* Nature
  Machine Intelligence 2025. [arXiv:2502.06027](https://arxiv.org/abs/2502.06027)
- **DiffSBDD** — Schneuing, Harris, Du, Didi, Jamasb, Igashov, Du, Gomes,
  Blundell, Liò, Welling, Bronstein & Correia. *Structure-based drug design with
  equivariant diffusion models.* Nature Computational Science 4(12), 899–909,
  2024. [doi:10.1038/s43588-024-00737-x](https://doi.org/10.1038/s43588-024-00737-x)
  ([arXiv:2210.13695](https://arxiv.org/abs/2210.13695))
- **DiffPharma** — Sekijima Lab (Institute of Science Tokyo). ChemRxiv preprint,
  2025. [chemrxiv.org/…/684c1f943ba0887c3310534d](https://chemrxiv.org/engage/chemrxiv/article-details/684c1f943ba0887c3310534d) —
  a pharmacophore-conditioned extension of **DiffSBDD** (above).
- **PMDM** — Huang, Yang, Zhou, Zhang, Chen, Zhang, Wang & Tang. *A dual
  diffusion model enables 3D molecule generation and lead optimization based on
  target pocket.* Nature Communications 2024.
  [doi:10.1038/s41467-024-46569-1](https://doi.org/10.1038/s41467-024-46569-1)
- **KGDiff** — Qian, Huang, Tu & Xu. *KGDiff: towards explainable target-aware
  molecule generation with knowledge guidance.* Briefings in Bioinformatics
  25(1), 2024. [doi:10.1093/bib/bbad435](https://doi.org/10.1093/bib/bbad435)
- **TargetDiff** — Guan, Qian, Peng, Su, Peng & Ma. *3D Equivariant Diffusion
  for Target-Aware Molecule Generation and Affinity Prediction.* ICLR 2023.
  [arXiv:2303.03543](https://arxiv.org/abs/2303.03543) — KGDiff is built on it,
  so `diffusion_kgdiff.yaml` runs both. IPDiff is also built on it.
- **IPDiff** — Huang, Yang, Zhou, Zhang, Zhang, Zheng, Chen, Wang, Cui & Yang.
  *Protein-Ligand Interaction Prior for Binding-aware 3D Molecule Diffusion
  Models.* ICLR 2024.
  [openreview:qH9nrMNTIW](https://openreview.net/forum?id=qH9nrMNTIW)
- **DiffLinker** — Igashov et al. *Equivariant 3D-Conditional Diffusion Model
  for Molecular Linker Design.* Nature Machine Intelligence 2024. [arXiv:2210.05274](https://arxiv.org/abs/2210.05274)
- **eSEN** — Fu et al. *Learning Smooth and Expressive Interatomic Potentials
  for Physical Property Prediction.* 2025. [arXiv:2502.12147](https://arxiv.org/abs/2502.12147)
- **EquiformerV2** — Liao, Wood, Das & Smidt. *EquiformerV2: Improved Equivariant
  Transformer for Scaling to Higher-Degree Representations.* ICLR 2024. [arXiv:2306.12059](https://arxiv.org/abs/2306.12059)
- **ShEPhERD** — Adams, Abeywardane, Fromer & Coley. *ShEPhERD: Diffusing shape,
  electrostatics, and pharmacophores for bioisosteric drug design.* ICLR 2025.
  [arXiv:2411.04130](https://arxiv.org/abs/2411.04130)
