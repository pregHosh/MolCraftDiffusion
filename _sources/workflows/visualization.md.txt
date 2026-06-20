# Workflow: Visualize the Generated Chemical Space

Understanding a set of generated molecules as a *distribution* — not just as individual samples — is often the most important quality-control step before downstream use. This workflow featurizes your molecules into fixed-size vectors and projects them into a 2D space for visual inspection.

## Conceptual Flow

```text
 [Generated 3D molecules]          [Reference / training set]
            |                                  |
            v                                  v
        [Featurize]  ←─────────── same featurizer ────────────┘
            |
            v
  [Stack feature matrix]
            |
            v
  [UMAP / t-SNE projection]
            |
            v
  [Scatter plot — color by source, property, or cluster]
```

## What Visualization Tells You

Visualization answers questions that scalar metrics cannot:

- **Is the generated set concentrated or broadly distributed?** A tight cluster suggests the model collapsed to a small region of chemical space; broad spread indicates diversity.
- **Are there distinct structural clusters?** Clusters often correspond to recognizable chemical families — useful for identifying privileged scaffolds in the output.
- **Do generated molecules overlap with or extend beyond the reference set?** Overlap = the model reproduces known chemistry; extension = the model is genuinely exploring.
- **How does guided generation shift the distribution?** Plot unconditional vs. property-directed outputs side-by-side to measure the shift.

---

## Step 1: Featurize

Use `MolCraftDiff analyze featurize` to convert XYZ files into a numpy matrix of fixed-size vectors. The default `--backend soap` requires no additional setup beyond `[data]`; the `--backend uma` backend produces richer embeddings but requires the fairchem clone (see [Installation](../installation.md)).

```bash
# SOAP features (default) — one vector per molecule, fast, no GPU needed
MolCraftDiff analyze featurize generated_molecules/ \
    --backend soap \
    --output features_generated.npz

# Featurize the reference/training set with the same settings
MolCraftDiff analyze featurize training_data/ \
    --backend soap \
    --output features_reference.npz
```

Each `.npz` file contains:
- `features`: array of shape `(N, D)` — one row per molecule
- `labels` (if `--labels` was passed): array of shape `(N,)` with property values for coloring

---

## Step 2: Project to 2D

```python
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP  # pip install umap-learn

# Load features
gen = np.load("features_generated.npz")
ref = np.load("features_reference.npz")

X = np.vstack([ref["features"], gen["features"]])
labels = np.array(["reference"] * len(ref["features"]) +
                  ["generated"] * len(gen["features"]))

# Fit UMAP on the combined set so both share the same projection
reducer = UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
embedding = reducer.fit_transform(X)

# Split back
n_ref = len(ref["features"])
emb_ref, emb_gen = embedding[:n_ref], embedding[n_ref:]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(*emb_ref.T, s=10, alpha=0.4, label="Reference", color="steelblue")
ax.scatter(*emb_gen.T, s=10, alpha=0.6, label="Generated", color="coral")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend()
plt.tight_layout()
plt.savefig("chemical_space.png", dpi=150)
```

Replace `UMAP` with `sklearn.manifold.TSNE` for t-SNE if preferred. t-SNE is slower and does not support `transform()`, so fit on the combined matrix as shown above.

---

## Step 3: Interpret the Plot

| What you see | What it means | What to do |
| :--- | :--- | :--- |
| Generated points fully inside reference cloud | Model reproduces training data; no novel exploration | Lower temperature, try unconditional generation from scratch, or use outpainting |
| Generated points extending beyond reference cloud | Model is extrapolating — potentially novel chemistry | Run `analyze metrics` to check validity; filter before downstream use |
| Tight cluster of generated points, reference scattered | Mode collapse or over-conditioning | Reduce `cfg_scale`, increase `diffusion_steps`, check `denoising_strength` |
| Two or more distinct clusters in generated set | Structural families in output | Inspect representative members from each cluster manually |

---

## SOAP vs UMA Features

| Feature type | Strengths | When to prefer |
| :--- | :--- | :--- |
| SOAP | Fast (CPU), no pretrained model needed, well-understood | Quick QC, large sets (>10 000 molecules), when UMA is unavailable |
| UMA | Richer semantic encoding, better at distinguishing subtly different scaffolds | Final analysis, small–medium sets, when fairchem clone is available |

For SOAP, the projection quality depends on the `species` list passed to the featurizer. Use the same species list for both generated and reference sets, and include all atomic species present in either.

---

## See Also

- [Tutorial 9: Analysis](../tutorials/09_analyze.md) — `featurize` subcommand reference
- [Workflow: End-to-End](end_to_end.md) — where visualization fits in the full pipeline
- [Applications](../applications/index.md) — research-level use cases
