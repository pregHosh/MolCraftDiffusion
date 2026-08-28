# Model Architectures

MolCraftDiffusion ships several backbone families as first-class options. You
select one by choosing a **task config** (`MolCraftDiff train <config>`); the
config's `_target_` factory builds the matching model, so adding or swapping an
architecture never touches the core engine.

Packaged task configs live in `src/MolecularDiffusion/configs/tasks/`. The tables below list one row per **distinct
model**; configs that differ only in hyperparameters or a starting checkpoint
are not listed separately.

:::{tip}
**You do not have to train these yourself to try them.** Most models below ship
pretrained weights, the data they need, and a runnable config through the model
zoo — fetch one by name and generate:

```bash
MolCraftDiff zoo list                        # what is available, with sizes
MolCraftDiff zoo fetch --model kgdiff
MolCraftDiff generate examples/kgdiff_generate.yaml
```

Where a row below says the shipped checkpoint behaves a certain way, that is
the checkpoint you get. Three cases:

- **Fetch and go** — apo2mol, diffdec, diffint, diffpharma, diffsbdd,
  flowmol_graph3d, gcdm, ipdiff, kgdiff, ligandiff, ligandiff_multi, loqi, midi.
- **Build locally** — diffsmol, ditmc, equifm, nextmol, pmdm. Weights exist,
  but their upstream projects do not permit redistribution, so the zoo ships
  the recipe instead: `MolCraftDiff zoo recipe <asset>` prints the download,
  the conversion command and the expected checksum.
- **Train it yourself** — difflinker and silvr have no pretrained weights at
  all. The zoo still carries their datasets and example configs.

See the [Model Zoo](model_zoo/index.md) to get
started.
:::

## 1. De novo 3D generation

Whole-molecule generators trained on a plain 3D molecule dataset. The first
five can also be steered towards a target property value; the rest generate
freely.

| Task config | `task_type` | Model | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion.yaml` | `diffusion` | EDM (E(n)-equivariant diffusion, EGCL backbone) | **Start here.** The default and best-tested path; the Hugging Face checkpoints for this repo use it. Also the slowest to sample. |
| `diffusion_egt.yaml` | `diffusion` | EGT (equivariant graph transformer) | Behaves like the default — same conditioning, roughly twice as fast to sample. Worth a try if the default underfits your data. |
| `diffusion_gfmdiff.yaml` | `diffusion` | GFMDiff | Another same-behaviour alternative to the default, also about twice as fast to sample. |
| `diffusion_painn.yaml` | `diffusion` | PaiNN (scalar+vector message passing) | Same again, using the backbone from OM-Diff's organometallic work — the one to try on metal-containing systems. |
| `diffusion_gcdm.yaml` | `diffusion` | GCDM (GCPNet backbone) | **The one that edits existing molecules** — refines structures towards a property target instead of generating from scratch. Trade-off: slowest to sample of these backbones. |
| `diffusion_tabasco.yaml` | `diffusion_tabasco` | TABASCO | **The fast one** — around nine times fewer steps than the default, with cleaner geometry. Trade-off: no property targeting, so it only generates freely. |
| `diffusion_equifm.yaml` | `diffusion_equifm` | EquiFM (flow matching, EGNN backbone) | Flow matching rather than diffusion. Note it is **not faster** than the default out of the box — pick TABASCO if speed is what you want. Stick to the shipped checkpoint; results from training it yourself are unverified. |
| `diffusion_flowmol.yaml` | `diffusion_flowmol` | FlowMol (SE(3)-equivariant GVP) | The earlier, smaller FlowMol: no bonds, and trained for neutral molecules. Use FlowMol3 below unless you have a reason not to. |
| `diffusion_midi.yaml` | `diffusion_midi` | MiDi (relational graph transformer with equivariant coordinate updates) | The one that **draws the bonds for you**. Everything above gives you atoms in space and leaves you to guess the chemistry afterwards; this one hands you a finished molecule. Best on small, QM9-like molecules. Two things to know: it needs a bond-aware dataset built in advance, and you must ask for SDF output or the bonds are thrown away. |
| `diffusion_flowmol_graph3d.yaml` | `diffusion_flowmol_graph3d` | FlowMol3 (SE(3)-equivariant GVP, CTMC discrete flow matching) | Same idea as MiDi, but for **drug-sized molecules** — pick MiDi instead when your molecules are small. It trims the atom count you ask for but never exceeds it. Most setup work of anything here: you build the drug-scale dataset yourself, and you must ask for SDF output or the bonds are thrown away. |
| `diffusion_nextmol.yaml` | `diffusion_nextmol` | NExT-Mol (MoLlama language model + DMT diffusion transformer) | The one that **writes the molecule down before building it in 3D** — a language model proposes the molecule, then diffusion places it in space, so you can read and filter the molecule list before any 3D work starts. Two things to know: the molecules follow general drug-like chemistry rather than your dataset, and anything containing S, Cl or Br is dropped and reported. |
| `diffusion_jodo.yaml` | `diffusion_jodo` | JODO (diffusion graph transformer, joint 2D+3D) | Draws the bonds for you, like MiDi, and can aim at a target property (gap, dipole, polarizability, HOMO, LUMO, heat capacity). Small or drug-sized molecules. Ask for SDF output or the bonds are dropped. |

Two families take the same de novo idea but split it into two stages: train an
autoencoder first, then diffuse in its latent space instead of in coordinate
space directly. Each needs **both** of its configs, VAE first.

| Family | Stage 1 — VAE | Stage 2 — diffusion | Notes |
| :--- | :--- | :--- | :--- |
| GeoLDM | `vae_geoldm.yaml` (`vae_geoldm`) | `diffusion_geoldm.yaml` (`diffusion_geoldm`) | The established latent option, but you train two models and the autoencoder here uses the setup the original authors themselves reported as unstable — expect worse than the published numbers. The default EDM is the safer choice. |
| ADiT | `vae_transformer.yaml` (`vae_transformer`) or `vae_equiformer.yaml` (`vae_equiformer`) | `diffusion_adit.yaml` (`diffusion_adit`) | For **large datasets and long training runs**, not for a few thousand molecules. Two encoder choices: the plain transformer is cheaper, Equiformer is more robust to how the molecule is oriented. Nothing here is pre-trained — you train both stages yourself. |

## 2. Synthesizable / retroanalysis-constrained generation

Molecules are **assembled**, not drawn atom by atom: the model picks catalogue
building blocks and the reactions that join them, and generates the coordinates
at the same time. So every result arrives with a synthesis route and its bonds
already drawn. The trade-off is the catalogue — these models can only ever make
what their building blocks and reactions allow, and they need their own
reaction-graph dataset rather than a plain molecule set. Ask for SDF output or
the bonds are thrown away.

| Task config | `task_type` | Conditioned on | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion_syncogen.yaml` | `diffusion_syncogen` | nothing — generates freely | **The one whose molecules come with a synthesis route** — you get something you can order and make. The catalogue is the constraint: 93 building blocks, 19 reactions, at most five blocks per molecule. |
| `diffusion_syncogen_pharm.yaml` | `diffusion_syncogen_pharm` | **Pharmacophore** points from a reference ligand | The same generator steered to match a reference molecule's pharmacophore — pick it over ShEPhERD in section 3 when the result has to be makeable, and ShEPhERD when you also need shape and electrostatics. |

## 3. Conditional and structure-aware generation

Generation steered by an external input — a shape, a pocket, a set of fragments,
a pharmacophore. These need **paired** data (the condition alongside the
molecule); a plain molecule dataset is not enough. Except for ShEPhERD, none
of them generate bonds — you get atoms in space and the chemistry is perceived
afterwards. Most are heavy-atom only; KGDiff, PMDM and Apo2Mol also place
hydrogens.

Synthesizable generation can also be steered by a pharmacophore — see section 2.

| Task config | `task_type` | Conditioned on | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion_diffsmol.yaml` | `diffusion_diffsmol` | Molecular **shape** | Intended for **shape-matching / bioisostere** work, but as shipped the pretrained model **ignores the shape you give it** — the original authors released an incomplete shape encoder. Treat it as a plain generator, or a starting point for your own training. For working shape-based design use ShEPhERD below. |
| `diffusion_diffsbdd.yaml` | `diffusion_diffsbdd` | **Protein pocket** | **Start here for pocket-based design** — the best-tested and most flexible of the SBDD options. It is also the only one that does **scaffold hopping**: keep part of a known ligand and regenerate the rest — though the part you keep has to be chosen from a ligand already in your converted dataset. |
| `diffusion_diffpharma.yaml` | `diffusion_diffpharma` | **Protein pocket** + pharmacophore particles | Give it a pocket **plus a known binder's pose**; it reads that binder's contacts and designs new molecules that reproduce them. You cannot hand-author the interaction pattern yourself. |
| `diffusion_diffint.yaml` | `diffusion_diffint` | **Protein pocket** + hydrogen-bond interaction particles | Narrower than DiffPharma: it keeps only the **hydrogen bonds** a known binder makes, rather than its full contact pattern. Like DiffPharma it needs that binder's pose, not just a pocket. If no hydrogen bonds are found it quietly degrades to plain DiffSBDD. |
| `diffusion_pmdm.yaml` | `diffusion_pmdm` | **Protein pocket** | Straightforward pocket-conditioned design, and a reasonable second opinion alongside DiffSBDD. De novo generation only here — the paper's lead-optimisation and linker modes were not ported. |
| `diffusion_kgdiff.yaml` | `diffusion_kgdiff` | **Protein pocket** | Choose this when you want samples **pushed towards better predicted binding affinity** — it scores and steers itself, with no second model to train. Turning that steering off gives you plain [TargetDiff](https://arxiv.org/abs/2303.03543), so this config doubles as the unguided baseline to compare against. The steering only means anything if your training set carries **real measured affinities**. |
| `diffusion_ipdiff.yaml` | `diffusion_ipdiff` | **Protein pocket** | Binding awareness is learned during training rather than steered at sampling, so there is no knob to turn. The heaviest sampler here, and the shipped checkpoint tends to produce carbon-heavy, chemically dull molecules — check your output before trusting it. |
| `diffusion_apo2mol.yaml` | `diffusion_apo2mol` | **Apo protein pocket** | The one for **targets with no known binder**: it takes a ligand-free structure and reshapes the pocket as it designs, instead of assuming the pocket is already the right shape. Run the full sampling schedule: shorten it and the pocket never moves, which defeats the point. On the one complex tested, the shipped weights moved the pocket *away* from the true bound shape — validate before relying on it. |
| `diffusion_difflinker.yaml` | `diffusion_difflinker` | **Fragments** to join | **Linker design**: hold fragments fixed, generate the atoms joining them. You choose the linker length — it does not pick one for you. No pocket and no pretrained weights, so you train it yourself; use DiffDec instead if you need either. |
| `diffusion_diffdec.yaml` | `diffusion_diffdec` | **Scaffold** + anchor atom + **protein pocket** | **R-group decoration**: keep a scaffold fixed, pick one attachment point, and grow a substituent there inside the pocket. Choose it over DiffLinker when you are growing off a scaffold rather than bridging two fragments. One R-group per run, and the model picks its size for you, up to about 10 heavy atoms. |
| `pharmacophore.yaml` | `diffusion_pharmacophore` | **Pharmacophore** points, electrostatics, shape | ShEPhERD — ligand-based design when you have a reference molecule but **no protein structure**, and the one option here whose conditioning actually works end to end. Shape matching needs to be trained in; the shipped setup covers pharmacophores and electrostatics. It is also the only model in this table that generates bonds. |

## 4. Conformer generation

These do not design molecules. You already know *what* the molecule is — you
want to know what **shape** it takes. The generators on this page invent new
chemistry; this section keeps yours exactly as drawn and only works out the
geometry. That also means an ordinary molecule dataset is enough
to train on, with no paired conditions to assemble.

Because the molecule is yours, you ask for a number of shapes **per molecule**
rather than a total, and there is no molecule size to set — your structure
already fixes that. Results come back in one folder per input molecule, next to
a table listing every shape produced, which molecule it belongs to, and how far
it moved from the structure you supplied.

| Task config | `task_type` | Notes |
| :--- | :--- | :--- |
| `diffusion_loqi_flow.yaml` | `diffusion_loqi_flow` | **Start here.** Hand it a structure and it gives back realistic, low-energy 3D shapes of that same molecule, keeping the left/right-handedness and double-bond geometry you drew. Reach for it when the quick built-in conformer tools are not good enough — flexible molecules, large rings, and anything you are about to dock or minimise. You can dial sampling up for quality or down for speed. |
| `diffusion_loqi.yaml` | `diffusion_loqi` | The same model trained a different way. Slightly rougher structures than the above and fixed to one sampling setting, so prefer the flow version unless you specifically want this checkpoint. |
| `diffusion_ditmc.yaml` | `diffusion_ditmc` | A transformer alternative to the two above, worth a second opinion on floppy molecules. Give it a molecule and it returns 3D poses of exactly that molecule. Nothing ready-made comes with it, so you must train it first — on a multi-conformer set — and start with LoQi if you cannot. |
| `diffusion_etflow.yaml` | `diffusion_etflow` | Another second opinion alongside DiTMC, but this one **comes with ready-made weights** — one set for small molecules, one for drug-sized ones — so there is nothing to train first. It needs a **single connected molecule**: given a salt or anything in two pieces it returns a hugely exploded structure instead of stopping with an error, so split those off first. |

## 5. Structure elucidation

You have the compound already — it is the sample in front of you — and you
want to know what it is. Give these a measured spectrum and they propose the
structure. Any measurement that pins a structure down fits here; NMR is the
first one shipped.

Most of these also need the **molecular formula** as an input, not something
the model works out for you — in practice it comes from high-resolution mass
spec, so run that first — and every candidate then has exactly the atoms you
declared. DiffSpectra is the exception: it generates the atom types and the
bonds itself, so it needs only the spectrum. What comes back is a ranked
shortlist per measurement, not a single answer. On carbon spectra alone the
shipped ChefNMR weights name roughly four compounds in ten outright; with
proton spectra as well, substantially better.

| Task config | `task_type` | Measurement | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion_chefnmr.yaml` | `diffusion_chefnmr` | **1D NMR** — 1H and 13C, or 13C alone | ChefNMR — names an unknown from its NMR. Your own unknown goes in as a short file listing the peaks and the formula; the prepared spectrum collections are only needed to reproduce published benchmark numbers. Structures come back with hydrogens placed but no bonds drawn — the chemistry is read off the geometry afterwards, and a candidate that cannot be read as a molecule is dropped before you see it. |
| `diffusion_diffspectra.yaml` | `diffusion_diffspectra` | **IR, Raman, or UV-Vis** — alone or combined | DiffSpectra — proposes a complete structure, bonds included, from the spectrum alone; no molecular formula needed, unlike ChefNMR above. QM9-scale only (≤9 heavy atoms), nothing drug-sized. |

## 6. Transition state generation

Give these a reactant and a product and they give you back the transition
state, in seconds rather than hours. Good enough to optimise from, or to
screen a lot of reactions cheaply — not a final answer.

These learn from **reactions**, so you need a reaction dataset, not a molecule
one. Your reactant and product also have to be numbered the same way, atom for
atom.

| Task config | `task_type` | Takes | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion_reactot.yaml` | `diffusion_reactot` | **Reactant + product** geometries | **Start here.** Roughly ten passes instead of a few hundred, so it is quick enough to run over a whole set of reactions. It also covers more ground than OA-ReactDiff: it was trained on reactions where two molecules come together or break apart, not just ones that stay in one piece. The catch is that it gives you the same structure every time you ask, so if an answer looks wrong, asking again will not help — use OA-ReactDiff instead. |
| `diffusion_oareactdiff.yaml` | `diffusion_oareactdiff` | **Reactant + product** geometries | **Reach for this when one attempt is not enough.** It gives a different structure each time, so you can ask for a batch and keep the best — the usual way to rescue a reaction the fast model gets wrong. Narrower scope: small organic reactions made of H, C, N, O, up to about 23 atoms, with each side a single connected molecule. Slower, by roughly the ratio above. |

Both usually land close enough to the true transition state to optimise from.
They have not been compared head to head on the same reactions here, so treat
the choice above as one of speed, coverage and whether you want more than one
attempt — not as a ranking on accuracy.

## 7. Transition-metal complex generation

Ligand design *around a metal centre*: freeze the metal and the retained
ligands, re-diffuse the rest. Neither generates bonds, every run starts from an
input complex, and both need complexes prepared with the bundled converter — a
plain coordinate file is not enough. The shipped weights for both only ever saw
the metals Cr through Zn.

| Task config | `task_type` | Regenerates | Notes |
| :--- | :--- | :--- | :--- |
| `diffusion_ligandiff.yaml` | `diffusion_ligandiff` | Exactly **one** ligand per run | Swap **one** ligand for new chemistry while the rest of the complex stays put; the coordination geometry is preserved. Ligand assignments ship for one published dataset — bringing your own complexes needs molSimplify. |
| `diffusion_ligandiff_multi.yaml` | `diffusion_ligandiff_multi` | **Any subset**, one ligand up to the whole coordination sphere | The broader option: keep any number of ligands and regenerate the rest, so use it when more of the coordination sphere is up for redesign. **Octahedral complexes only.** How the free sites are divided between new ligands is chosen at random unless you specify it — the model does not predict it. |

## 8. Property prediction and guidance

Not generators. `regression` predicts a property; `guidance` exposes the same
head as a gradient signal to steer a diffusion sampler. Only EGCL and eSEN can
be used for that steering.

| Task config | `task_type` | Backbone | Notes |
| :--- | :--- | :--- | :--- |
| `regression.yaml` / `guidance.yaml` | `regression` / `guidance` | EGCL | **Start here.** Fastest to train, lightest on memory, and the safest choice for steering generation. |
| `regression_esen.yaml` / `guidance_esen.yaml` | `regression` / `guidance` | eSEN | Slower and heavier; reach for it when EGCL's accuracy plateaus. |
| `regression_equiformer.yaml` | `regression` | EquiformerV2 | The most expensive option, for prediction runs where accuracy matters more than cost. It **cannot** steer generation. |

## 9. Self-supervised pretraining

Train a backbone on unlabeled 3D structures. The resulting checkpoint is usable
today as a **molecular featuriser** (`MolCraftDiff analyze featurize --backend
ssl3d`); fine-tuning one into a regression or guidance model is **not wired up
yet** and will be refused.

| Task config | `task_type` | Backbone | Notes |
| :--- | :--- | :--- | :--- |
| `ssl3d_egcl.yaml` | `ssl3d` | EGCL | **Start here** — cheap enough to sweep settings on. |
| `ssl3d_egt.yaml` | `ssl3d` | EGT | Looks at every atom pair, so memory grows with the square of the molecule size — keep molecules small. |
| `ssl3d_esen.yaml` | `ssl3d` | eSEN | Slower; pick it to match an eSEN model downstream. |
| `ssl3d_equiformer.yaml` | `ssl3d_equiformer` | EquiformerV2 | The slowest run here; only worth it to match an EquiformerV2 model downstream. |

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
- **GCDM** — Morehead & Cheng. *Geometry-Complete Diffusion for 3D Molecule
  Generation and Optimization.* Communications Chemistry 7, 150 (2024).
  [arXiv:2302.04313](https://arxiv.org/abs/2302.04313) — the same diffusion
  objective as **EDM** (above) with the EGNN backbone replaced by the
  geometry-complete GCPNet, plus the property-optimization mode.
- **TABASCO** — Vonessen, Harris, Cretu & Liò. *TABASCO: A Fast, Simplified
  Model for Molecular Generation with Improved Physical Quality.* 2025.
  [arXiv:2507.00899](https://arxiv.org/abs/2507.00899)
- **FlowMol** — Dunn & Koes. *Mixed Continuous and Categorical Flow Matching
  for 3D De Novo Molecule Generation.* 2024.
  [arXiv:2404.19739](https://arxiv.org/abs/2404.19739)
- **FlowMol3** — Dunn & Koes. *FlowMol3: Flow Matching for 3D De Novo
  Small-Molecule Generation.* 2025.
  [arXiv:2508.12629](https://arxiv.org/abs/2508.12629) — the bond-generating
  successor to **FlowMol** (above), adding self-conditioning, fake atoms and
  train-time geometry distortion. The discrete CTMC flow matching that carries
  its bond, atom-type and charge modalities comes from the intermediate
  *Exploring Discrete Flow Matching for 3D De Novo Molecule Generation*,
  MLSB @ NeurIPS 2024
  ([arXiv:2411.16644](https://arxiv.org/abs/2411.16644)).
- **MiDi** — Vignac, Osman, Toni & Frossard. *MiDi: Mixed Graph and 3D Denoising
  Diffusion for Molecule Generation.* ECML PKDD 2023.
  [arXiv:2302.09048](https://arxiv.org/abs/2302.09048) — the same paper the
  **EGT** backbone (above) is taken from; `diffusion_midi.yaml` ports the full
  joint graph-and-coordinate diffusion objective rather than the backbone alone.
- **JODO** — Huang, Sun, Du & Lv. *Learning Joint 2D & 3D Diffusion Models
  for Complete Molecule Generation.* 2023.
  [arXiv:2305.12347](https://arxiv.org/abs/2305.12347) — diffuses atoms,
  coordinates, formal charges and the bond graph together, the same joint
  objective **MiDi** (above) takes, with a relational graph transformer and
  an optional property-conditional variant. MIT licensed.
- **EquiFM** — Song, Gong, Xu, Cao, Lan, Ermon, Zhou & Ma. *Equivariant Flow
  Matching with Hybrid Probability Transport for 3D Molecule Generation.*
  NeurIPS 2023. [arXiv:2312.07168](https://arxiv.org/abs/2312.07168)
- **NExT-Mol** — Liu, Luo, Huang, Zhang, Li, Fang, Shi, Wang, Kawaguchi &
  Chua. *NExT-Mol: 3D Diffusion Meets 1D Language Modeling for 3D Molecule
  Generation.* ICLR 2025. [arXiv:2502.12638](https://arxiv.org/abs/2502.12638)
  — pairs the paper's own DMT relational graph transformer, which diffuses
  coordinates onto a fixed 2D graph, with the released MoLlama SELFIES
  language model that writes that graph.
- **SynCoGen** — Rekesh, Cretu, Shevchuk, Somnath, Liò, Batey, Tyers,
  Koziarski & Liu. *SynCoGen: Synthesizable 3D Molecule Generation via Joint
  Reaction and Coordinate Modeling.* 2025.
  [arXiv:2507.11818](https://arxiv.org/abs/2507.11818) — generates a
  building-block and reaction graph jointly with the coordinates, so the
  bonds fall out of the assembled molecule instead of being diffused. The
  masked discrete diffusion over those two channels is **MDLM**: Sahoo et
  al., *Simple and Effective Masked Diffusion Language Models*, NeurIPS 2024
  ([arXiv:2406.07524](https://arxiv.org/abs/2406.07524)).
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
- **ChefNMR** — Xiong, Zhang, Alauddin, Cheng, An, Seyedsayamdost & Zhong.
  *Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR
  Spectra.* NeurIPS 2025. [arXiv:2512.03127](https://arxiv.org/abs/2512.03127)
- **DiffSpectra** — Wang, Rong, Xu, Zhong, Liu, Wang, Zhao, Liu, Wu, Wang &
  Zhang. *DiffSpectra: Molecular Structure Elucidation from Spectra using
  Diffusion Models.* 2025. [arXiv:2507.06853](https://arxiv.org/abs/2507.06853)
  — the DMT backbone is **JODO**'s (above) with a SpecFormer spectral
  encoder replacing JODO's property-conditioning branch.
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
- **DiffDec** — Xie, Chen, Lei & Yang. *DiffDec: Structure-Aware Scaffold
  Decoration with an End-to-End Diffusion Model.* Journal of Chemical
  Information and Modeling 64(7), 2554–2564, 2024.
  [doi:10.1021/acs.jcim.3c01466](https://doi.org/10.1021/acs.jcim.3c01466) — a
  fork of **DiffLinker** (above): the same EDM objective and EGNN backbone,
  with the pocket added and fragment-joining swapped for anchored R-group
  growth.
- **eSEN** — Fu et al. *Learning Smooth and Expressive Interatomic Potentials
  for Physical Property Prediction.* 2025. [arXiv:2502.12147](https://arxiv.org/abs/2502.12147)
- **EquiformerV2** — Liao, Wood, Das & Smidt. *EquiformerV2: Improved Equivariant
  Transformer for Scaling to Higher-Degree Representations.* ICLR 2024. [arXiv:2306.12059](https://arxiv.org/abs/2306.12059)
- **ShEPhERD** — Adams, Abeywardane, Fromer & Coley. *ShEPhERD: Diffusing shape,
  electrostatics, and pharmacophores for bioisosteric drug design.* ICLR 2025.
  [arXiv:2411.04130](https://arxiv.org/abs/2411.04130)
- **LoQI** — Nikitin, Anstine, Zubatyuk, Paliwal & Isayev. *Scalable
  Low-Energy Molecular Conformer Generation with Quantum Mechanical Accuracy.*
  ChemRxiv 2025.
  [doi:10.26434/chemrxiv-2025-k4h7v](https://doi.org/10.26434/chemrxiv-2025-k4h7v)
  — built on the **Megalodon** co-design architecture: Reidenbach, Nikitin,
  Isayev & Paliwal, *Applications of Modular Co-Design for De Novo 3D Molecule
  Generation*, 2025 ([arXiv:2505.18392](https://arxiv.org/abs/2505.18392)).
- **DiTMC** — Frank, Ripken, Lied, Müller, Unke & Chmiela. *Sampling 3D
  Molecular Conformers with Diffusion Transformers.* NeurIPS 2025.
  [arXiv:2506.15378](https://arxiv.org/abs/2506.15378) — like **ADiT** (above)
  it builds on the **DiT** backbone of Peebles & Xie, here conditioned on the
  input molecule's bond graph rather than generating composition.
- **ET-Flow** — Hassan, Shenoy, Lee, Stark, Thaler & Beaini. *ET-Flow:
  Equivariant Flow-Matching for Molecular Conformer Generation.* NeurIPS 2024.
  [arXiv:2410.22388](https://arxiv.org/abs/2410.22388) — like **DiTMC** (above)
  it places an existing bond graph in 3D, but starts sampling from a prior
  built out of that graph rather than from plain noise.
- **OA-ReactDiff** — Duan, Du, Jia & Kulik. *Accurate transition state
  generation with an object-aware equivariant elementary reaction diffusion
  model.* Nature Computational Science 3, 1045–1055 (2023).
  [arXiv:2304.06174](https://arxiv.org/abs/2304.06174) — the **EDM** objective
  (above) run over three objects at once, reactant and product held at their
  own centres of mass while the transition state is denoised between them.
  The backbone is **LEFTNet**: Du, Du, Wang, Feng, Wang, Ji, Gomes & Ma, *A
  new perspective on building efficient and expressive 3D equivariant graph
  neural networks*, NeurIPS 2023
  ([arXiv:2304.04757](https://arxiv.org/abs/2304.04757)).
- **React-OT** — Duan, Liu, Du, Chen, Zhao, Jia, Gomes, Theodorou & Kulik.
  *Optimal transport for generating transition states in chemical reactions.*
  Nature Machine Intelligence 7, 615–626 (2025).
  [doi:10.1038/s42256-025-01010-0](https://doi.org/10.1038/s42256-025-01010-0)
  — the successor to **OA-ReactDiff** (above), by the same authors and over the
  same **LEFTNet** backbone, replacing the denoising diffusion with an
  optimal-transport bridge that starts from the reactant/product midpoint
  rather than from noise. The preprint carries a different title: *React-OT:
  Optimal Transport for Generating Transition State in Chemical Reactions*
  ([arXiv:2404.13430](https://arxiv.org/abs/2404.13430)).
