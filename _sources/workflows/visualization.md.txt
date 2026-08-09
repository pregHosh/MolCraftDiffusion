# Workflow: Visualise the Generated Chemical Space

Featurise your molecules into fixed-size vectors and project them to 2D, so you can inspect a generated set as a *distribution* rather than one sample at a time.

## Quickstart

```bash
# 1. Featurise both sets with identical settings
MolCraftDiff analyze featurize generated_molecules/ --backend soap --output features_generated
MolCraftDiff analyze featurize training_data/       --backend soap --output features_reference
```

Each run writes `<output>.npy` (the `(N, D)` feature matrix), `<output>.csv` (row → source file), and `<output>_meta.json`. Then project and plot with the [Step 2 snippet](#step-2-project-to-2d) → `chemical_space.png`.

:::{note}
**Why bother?** Scalar metrics can't tell you *how* a set is distributed. A 2D map can:
- **Concentrated vs. spread** — a tight blob suggests mode collapse; broad spread suggests diversity.
- **Distinct clusters** — often map to chemical families / privileged scaffolds.
- **Overlap with the reference** — overlap = reproduces known chemistry; extension beyond it = genuine exploration.
- **Effect of guidance** — plot unconditional vs. property-directed side-by-side to see the shift.
:::

```text
 [Generated 3D molecules]     [Reference / training set]
            |                             |
            +---------- Featurise --------+      (same backend & settings)
                          |
                  [Stack feature matrix]
                          |
                 [UMAP / t-SNE to 2D]
                          |
        [Scatter — colour by source, property, or cluster]
```

---

## Step 1: Featurise

**Goal: one fixed-size vector per molecule, using the same backend for both sets.**

`MolCraftDiff analyze featurize <dir> --output <stem>` converts a directory of XYZ files into a feature matrix. Pick a backend:

| Backend | Setup | Notes |
| :--- | :--- | :--- |
| `soap` (default) | `[data]` only, CPU | Fast; scales to large sets (>10 000). Quality depends on the `species` list — pass the **same** `--species` (or `--autodetect`) for both sets, covering every element in either. |
| `uma` | fairchem clone + checkpoint ([Installation](../installation.md)) | Pretrained UMA neural-network potential embeddings; GPU with `--device cuda`. |
| `ssl3d` | your own SSL3D checkpoint | Pass `--ssl3d-checkpoint <path>`. |

```bash
MolCraftDiff analyze featurize generated_molecules/ --backend soap --output features_generated
MolCraftDiff analyze featurize training_data/       --backend soap --output features_reference
```

Each command writes three files next to the stem:
- `<stem>.npy` — feature array of shape `(N, D)`, one row per molecule (load with `np.load`).
- `<stem>.csv` — `index, file, frame` for each row (use it to trace a point back to a structure, or to join your own per-molecule property values for colouring).
- `<stem>_meta.json` — backend, structure count, feature dim, and parameters.

---

## Step 2: Project to 2D

**Goal: fit one projection on both sets combined, so the two share coordinates.**

```python
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP  # pip install umap-learn

gen = np.load("features_generated.npy")   # (N_gen, D)
ref = np.load("features_reference.npy")   # (N_ref, D)

X = np.vstack([ref, gen])

# Fit on the combined set so both share the same projection
reducer = UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
embedding = reducer.fit_transform(X)
emb_ref, emb_gen = embedding[:len(ref)], embedding[len(ref):]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(*emb_ref.T, s=10, alpha=0.4, label="Reference", color="steelblue")
ax.scatter(*emb_gen.T, s=10, alpha=0.6, label="Generated", color="coral")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.legend()
plt.tight_layout()
plt.savefig("chemical_space.png", dpi=150)
```

To **colour by a property** instead of by source, read the property per row (e.g. join `features_generated.csv`'s `file` column with your own values) and pass it as `c=` to `scatter`.

Prefer t-SNE? Swap `UMAP` for `sklearn.manifold.TSNE`. It's slower and has no `transform()`, so fit it on the combined matrix exactly as above.

---

## Step 3: Interpret the Plot

| What you see | What it means | What to do |
| :--- | :--- | :--- |
| Generated points fully inside the reference cloud | Reproduces training data; no novel exploration | Lower temperature, generate unconditionally from scratch, or use outpainting |
| Generated points extending beyond the reference cloud | Extrapolating — potentially novel chemistry | Run `analyze metrics` to check validity; filter before downstream use |
| Tight generated cluster, reference scattered | Mode collapse or over-conditioning | Reduce `cfg_scale`, raise `diffusion_steps`, check `denoising_strength` |
| Two or more distinct generated clusters | Structural families in the output | Inspect representative members of each cluster |

---

## See Also

- [Tutorial 9: Analysis](../tutorials/09_analyze.md) — `featurize` subcommand reference
- [Workflow: End-to-End](end_to_end.md) — where visualisation fits in the full pipeline
- [Applications](../applications/index.md) — research-level use cases
