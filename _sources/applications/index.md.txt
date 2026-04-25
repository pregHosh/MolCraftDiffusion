# Applications

This section connects the core tutorials to concrete research workflows. Instead of introducing new configuration concepts, it shows how to **combine the existing training, generation, and analysis modules** for common molecular design objectives discussed in the MolCraftDiffusion manuscript.

## Application Map

The current examples fall into three broad categories:

| # | Application | Core mechanism | Entry point |
| :--: | :--- | :--- | :--- |
| 1 | **Local chemical space exploration** | Inpainting — vary a region while fixing the scaffold | [Local Chemical Space](local_chemical_space.md) |
| 2 | **Library design** | Outpainting — expand a scaffold into a diverse candidate set | [Library Design](library_design.md) |
| 3 | **Inverse design** | Property- or geometry-directed generation toward a design objective | [Inverse Design](inverse_design.md) |

## Before You Start

:::{important}
These application notes assume you are already familiar with the following tutorials:

- [Tutorial 6: Structure-Guided Generation](../tutorials/06_structure_guided.md)
- [Tutorial 7: Property-Directed Generation](../tutorials/07_property_directed.md)
- [Tutorial 8: Property Prediction and Evaluation](../tutorials/08_eval_predict.md)
- [Tutorial 9: Analysis](../tutorials/09_analyze.md)
:::

You will also need:

- a **trained or fine-tuned checkpoint** in `chkpt_directory`,
- **reference XYZ structures** for structure-guided workflows,
- any required **regressor or guidance checkpoints** for property-directed workflows.

## Pages

```{toctree}
:maxdepth: 1

local_chemical_space
library_design
inverse_design
```
