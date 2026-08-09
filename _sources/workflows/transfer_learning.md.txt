# Workflow Example: Transfer Learning and Generate

This workflow adapts a **pre-trained** model to a more target-relevant dataset before generation. The main goal is to shift the model from broad chemical knowledge towards a narrower and more useful design space — **without paying the cost of training a diffusion model from scratch.**

## Prefer Fine-Tuning Over Training From Scratch

Training a 3D molecular diffusion model from scratch is expensive: it needs a large dataset, many GPU-hours, and careful tuning before it even produces valid geometries. In almost all cases you should **not** start there.

Instead, start from one of the pre-trained models we ship and fine-tune it on your target chemistry. Fine-tuning reuses the broad structural knowledge the base model already learned (bonding, valence, 3D geometry) and only nudges it towards your domain, so it converges in a fraction of the data and compute. This is exactly how the singlet-fission and IFLP models in the paper were built — they are the GEOM foundation model fine-tuned on task-specific data, not trained fresh.

> Worakul, Azzouzi, Wodrich, Corminboeuf, *MolCraftDiffusion: 3D Molecular Generation Framework for Data-driven Molecular Applications*, JACS 2026 ([10.1021/jacs.5c19960](https://pubs.acs.org/doi/10.1021/jacs.5c19960)).

**Pre-trained bases to fine-tune from:**

| Base model | Trained on | Good starting point for |
| :--- | :--- | :--- |
| `EDM_geom` (GEOM-drugs foundation) | Large, drug-like 3D chemical space | Most organic / main-group design problems |
| `EDM_qm9` | Small molecules (≤ 9 heavy atoms) | Small-molecule tasks, property conditioning |

**Where to get them:** download the pre-trained checkpoints from the Zenodo release — <https://zenodo.org/records/19511401>.

The default answer to "should I train or fine-tune?" is **fine-tune** — unless your chemistry falls outside the base model's atom-type coverage (see below).

## When Fine-Tuning Is *Not* Enough — Atom-Type Coverage

A diffusion model has a **fixed atom-type head**: one output channel per element in the vocabulary it was trained with. Fine-tuning reuses that head, so it can reweight and rearrange the elements the base already knows, but it **cannot invent a new element** — there is no channel to represent one.

The pre-trained models we ship cover exactly these **17 atom types**:

```text
H, B, C, N, O, F, Al, Si, P, S, Cl, As, Se, Br, I, Hg, Bi
```

(Notably absent: `Ge, Sn, Sb, Te`, most transition metals, and the lanthanides/actinides.)

So the decision is:

- **Your molecules use only these elements** → fine-tune. This is the feasible, recommended path.
- **Your molecules need an element outside this set** → fine-tuning cannot introduce it. You must extend the atom vocabulary and **train from scratch** (or, at minimum, re-initialise and retrain the atom-type embedding/head), which is the expensive path fine-tuning was meant to avoid.

**How to check before you commit compute:** list the distinct elements in your target dataset and compare them against the 17 above. If every element is covered, fine-tune; if not, you are in the from-scratch case.

## When to Use This

Use this workflow when:

- you have a set of known molecules related to your design problem,
- unconditional generation from the base model is too broad,
- you want new molecules that remain closer to a task-relevant chemical domain,
- and (the usual case) your target chemistry stays within the pre-trained atom-type set.

## Conceptual Flow

```text
[Pre-trained diffusion model]   <- GEOM foundation or QM9 (do NOT train from scratch)
            |
            v
[Atom-type check: are all target elements in the base vocab?]
            |  yes                         |  no
            v                              v
[Transfer learning on             [Extend vocab + train from
 target-relevant molecules]        scratch — the expensive path]
            |
            v
[Generate molecules in adapted chemical space]
            |
            v
        [Analyse and filter]
```

## What Changes After Transfer Learning

Transfer learning changes the part of chemical space the model tends to sample from. Instead of correcting the output after generation, you first adapt the model so that its default generations are already closer to your target chemistry.

## Typical Outcome

Compared with direct sampling from the base model, this workflow often gives:

- molecules that look more like the chemistry you care about,
- less wasted sampling in irrelevant regions of chemical space,
- a better starting point for later conditional generation,
- all at a small fraction of the cost of training from scratch.

## Where to Go Next

- [Tutorial 4: Fine-Tuning](../tutorials/04_finetuning.md)
- [Tutorial 5: Generation Overview](../tutorials/05_generation_overview.md)
- [Tutorial 9: Analysis](../tutorials/09_analyze.md)
