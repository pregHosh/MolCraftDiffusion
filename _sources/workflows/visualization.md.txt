# Workflow Example: Visualize the Generated Space

This workflow focuses on understanding the generated molecule set as a distribution rather than only as individual samples. After generation, molecules are featurized and projected into a lower-dimensional representation for interpretation.

## Conceptual Flow

```text
 [Generated 3D molecules]
            |
            v
        [Featurize]
            |
            v
[Project to low-dimensional space]
            |
            v
[Visualize clusters, spread, and outliers]
```

## What Visualization Helps You See

Visualization helps answer questions that summary metrics cannot show clearly:

- Is the generated set concentrated or broadly distributed?
- Are there distinct structural clusters?
- Do generated molecules overlap with or extend beyond the reference set?

This is especially useful when comparing:

- a base model versus a transfer-learned model,
- unconditional versus guided generation,
- different filtering or ranking strategies.

## Common Projection Ideas

The draft mentions feature representations such as SOAP or learned embeddings, followed by low-dimensional projections such as UMAP or t-SNE. The exact representation is less important than the goal: building an interpretable picture of the generated chemical space.

## Where to Go Next

- [Tutorial 9: Analysis](../tutorials/09_analyze.md)
- [Applications](../applications/index.md)
