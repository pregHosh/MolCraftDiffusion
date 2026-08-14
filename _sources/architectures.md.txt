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
| `diffusion.yaml` | `diffusion` | EDM (E(n)-equivariant diffusion, EGCL backbone) | **Start here.** The default and best-tested path; the Hugging Face checkpoints for this repo use it. |
| `diffusion_egt.yaml` | `diffusion` | EGT (equivariant graph transformer) | Same objective as the default, with a transformer backbone instead of message passing. |
| `diffusion_gfmdiff.yaml` | `diffusion` | GFMDiff | Alternative de novo backbone; drop-in swap for the default. |
| `diffusion_painn.yaml` | `diffusion` | PaiNN (scalar+vector message passing) | Same objective again, with the backbone from OM-Diff's organometallic work. |
| `diffusion_tabasco.yaml` | `diffusion_tabasco` | TABASCO | Pick this when you care about **speed and clean geometry** — it is the simplified, physics-tuned option. |
| `diffusion_equifm.yaml` | `diffusion_equifm` | EquiFM (flow matching, EGNN backbone) | Flow matching, so sampling honours `num_steps` and can go short. Caveat: upstream released sampling code only, so **training here follows the paper, not an official implementation** — treat locally trained results as unverified. The converted QM9 checkpoint needs `discrete_path: HB_path`. |
| `diffusion_flowmol.yaml` | `diffusion_flowmol` | FlowMol (SE(3)-equivariant GVP) | Flow matching that also generates formal charges. Needs the `[flowmol]` extra (DGL). Bond-free variant only. |

## 2. Latent-space diffusion (two-stage)

Train an autoencoder first, then diffuse in its latent space. Each family needs
**both** of its configs, VAE first.

| Family | Stage 1 — VAE | Stage 2 — diffusion | Notes |
| :--- | :--- | :--- | :--- |
| GeoLDM | `vae_geoldm.yaml` (`vae_geoldm`) | `diffusion_geoldm.yaml` (`diffusion_geoldm`) | The established latent option. Diffusing in latent space is cheaper per step than Cartesian EDM, at the cost of training two models. |
| ADiT | `vae_transformer.yaml` (`vae_transformer`) or `vae_equiformer.yaml` (`vae_equiformer`) | `diffusion_adit.yaml` (`diffusion_adit`) | Transformer-scale latent diffusion; the encoder is your choice (plain transformer is cheaper, Equiformer is equivariant). Aimed at scaling up rather than small datasets. |

## 3. Conditional and structure-aware generation

Generation steered by an external input — a shape, a pocket, a set of fragments,
a pharmacophore. These need **paired** data (the condition alongside the
molecule); a plain molecule dataset is not enough. The shape- and
pocket-conditioned models all generate heavy atoms only, with no bonds —
bonds are perceived afterwards.

| Task config | `task_type` | Conditioned on | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion_diffsmol.yaml` | `diffusion_diffsmol` | Molecular **shape** | For **shape-matching / bioisostere** work: give it a reference molecule's surface, get different chemistry with the same shape. No protein needed. Requires an offline shape cache (`[shape]` extra). |
| `diffusion_diffsbdd.yaml` | `diffusion_diffsbdd` | **Protein pocket** | The **reference SBDD choice** — best documented, most flexible. Use the recommended `pocket_conditioning` mode; the `joint` mode needs `diffusion_diffsbdd_joint_moad.yaml`, since the CrossDocked joint weights are defective upstream. Uniquely here, it also does **scaffold hopping** by inpainting part of a known ligand (`gen_diffsbdd_inpaint.yaml`). |
| `diffusion_diffpharma.yaml` | `diffusion_diffpharma` | **Protein pocket** + pharmacophore particles | SBDD for when you know the **interaction pattern** you want, not just the pocket. Needs pocket-paired training data; a ligand-only dataset cannot train it. Novel pockets from raw PDB+SDF need the `[bio]` extra. |
| `diffusion_diffint.yaml` | `diffusion_diffint` | **Protein pocket** + hydrogen-bond interaction particles | SBDD steered by an **explicit H-bond pattern**: two pseudo-atoms per detected donor–acceptor pair are added to the pocket, so generation is biased towards reproducing that interaction geometry. Unlike every other pocket model here, sampling **requires a reference ligand pose** — the H-bonds are protein↔ligand, so a bare pocket is not enough. The pocket is CA-only (one node per residue). Needs pocket-paired training data and the `[bio]` extra. |
| `diffusion_pmdm.yaml` | `diffusion_pmdm` | **Protein pocket** | Plain pocket-conditioned SBDD with a dual short/long-range design. Also targets **lead optimisation**. Needs pocket-paired data. No scaffold hopping or linker sampling. |
| `diffusion_kgdiff.yaml` | `diffusion_kgdiff` | **Protein pocket** | Choose this when you want samples **steered towards predicted binding affinity** — its affinity head guides its own sampling, no second model to train. The same config **also runs [TargetDiff](https://arxiv.org/abs/2303.03543)** (`use_classifier_guide=false`, `guide_mode=wo`), so it doubles as the unguided baseline. |
| `diffusion_ipdiff.yaml` | `diffusion_ipdiff` | **Protein pocket** | Like KGDiff's lineage, but binding awareness is **baked into training** via a frozen interaction prior rather than applied at sampling. Prior weights ship with it and are mandatory. Caveat before you trust it: the released checkpoint is carbon-saturated (93% C against 64% in real ligands). |
| `diffusion_apo2mol.yaml` | `diffusion_apo2mol` | **Apo protein pocket** | The one for **flexible receptors**: condition on an **apo** (ligand-free) structure — what you actually have when there is no known binder — and it generates the ligand *and* the pocket's induced-fit conformation together. All the others assume a fixed, ligand-shaped pocket. Needs apo/holo **paired** training data. Generated pockets are written as `.pdb` sidecars next to the ligands. **Sample with the full schedule** (leave `num_steps` unset): unlike the flow-matching models, truncating it degrades the chemistry badly — 205 of 1000 steps returns near-random elements. |
| `diffusion_difflinker.yaml` | `diffusion_difflinker` | **Fragments** to join | **Linker design / fragment growing**: hold fragments fixed, generate the atoms connecting them. |
| `pharmacophore.yaml` | `diffusion_pharmacophore` | **Pharmacophore** points | ShEPhERD. Ligand-based design when you have a pharmacophore hypothesis but **no protein structure**. Requires `open3d`. |

## 4. Transition-metal complex generation

Ligand design *around a metal centre*: freeze the metal and the retained
ligands, re-diffuse the rest. Both are bond-free, conditional-only (every
sample needs an input complex), generate via the bundled `outpaint` mode, and
need a dataset built with `data.use_row_data_features: true` for their
per-atom conditioning columns.

| Task config | `task_type` | Regenerates | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion_ligandiff.yaml` | `diffusion_ligandiff` | Exactly **one** ligand per run | Swap a single ligand for new chemistry while the rest of the complex stays put — the coordination geometry is preserved, since the slot to fill is given. Generating from the released weights needs `diffusion_noise_schedule: learned`. |
| `diffusion_ligandiff_multi.yaml` | `diffusion_ligandiff_multi` | **Any subset**, one ligand up to the whole coordination sphere | The broader option: retain *k* ligands and regenerate the rest, or retain none and build a complex from a bare metal. It also decides the **number of new ligands and their denticities**, so use it when the coordination sphere itself is up for redesign. Released weights cover Cr–Zn only. |

## 5. Property prediction and guidance

Not generators. `regression` predicts a property; `guidance` exposes the same
head as a gradient signal to steer a diffusion sampler.

| Task config | `task_type` | Backbone |
| :--- | :--- | :--- |
| `regression.yaml` / `guidance.yaml` | `regression` / `guidance` | EGCL. **Default.** |
| `regression_esen.yaml` / `guidance_esen.yaml` | `regression` / `guidance` | eSEN. |
| `regression_equiformer.yaml` | `regression` | EquiformerV2. No `guidance` config for this backbone yet. |

## 6. Self-supervised pretraining

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
- **DiffInt** — Sako, Yasuo & Sekijima. *DiffInt: A Diffusion Model for
  Structure-Based Drug Design with Explicit Hydrogen Bond Interaction Guidance.*
  Journal of Chemical Information and Modeling 65(1), 71–82, 2025.
  [doi:10.1021/acs.jcim.4c01385](https://doi.org/10.1021/acs.jcim.4c01385) —
  like **DiffPharma** (above), from the Sekijima Lab and built on **DiffSBDD**;
  the network is DiffSBDD's unchanged, the contribution is the added
  interaction particles.
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
- **Apo2Mol** — Zheng, Jiang, Seabra, Li & Li. *Apo2Mol: 3D molecule generation
  via dynamic pocket-aware diffusion models.* AAAI 2026, 40(2), 1614–1622.
  [arXiv:2511.14559](https://arxiv.org/abs/2511.14559) — built on the
  **TargetDiff** backbone (above); its PMINet prior follows **IPDiff**'s
  prior-conditioning idea with a different network.
- **LigandDiff** — Jin & Merz. *LigandDiff: de Novo Ligand Design for 3D
  Transition Metal Complexes with Diffusion Models.* Journal of Chemical Theory
  and Computation 20(10), 4377–4384, 2024.
  [doi:10.1021/acs.jctc.4c00232](https://doi.org/10.1021/acs.jctc.4c00232)
- **multi-LigandDiff** — Jin & Merz. *Partial to Total Generation of 3D
  Transition-Metal Complexes.* Journal of Chemical Theory and Computation, 2024.
  [doi:10.1021/acs.jctc.4c00775](https://doi.org/10.1021/acs.jctc.4c00775) —
  an extension of **LigandDiff** (above), itself built on **DiffLinker**.
- **DiffLinker** — Igashov et al. *Equivariant 3D-Conditional Diffusion Model
  for Molecular Linker Design.* Nature Machine Intelligence 2024. [arXiv:2210.05274](https://arxiv.org/abs/2210.05274)
- **eSEN** — Fu et al. *Learning Smooth and Expressive Interatomic Potentials
  for Physical Property Prediction.* 2025. [arXiv:2502.12147](https://arxiv.org/abs/2502.12147)
- **EquiformerV2** — Liao, Wood, Das & Smidt. *EquiformerV2: Improved Equivariant
  Transformer for Scaling to Higher-Degree Representations.* ICLR 2024. [arXiv:2306.12059](https://arxiv.org/abs/2306.12059)
- **ShEPhERD** — Adams, Abeywardane, Fromer & Coley. *ShEPhERD: Diffusing shape,
  electrostatics, and pharmacophores for bioisosteric drug design.* ICLR 2025.
  [arXiv:2411.04130](https://arxiv.org/abs/2411.04130)
