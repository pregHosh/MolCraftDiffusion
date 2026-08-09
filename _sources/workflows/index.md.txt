# Workflows

This section gives example end-to-end workflows built from the MolCraftDiffusion modules. Unlike the tutorials, which explain individual capabilities one by one, these pages show how those capabilities are combined in practice.

## Workflow Families

The examples in this section follow four common patterns:

1. **Generate and filter** from a pre-trained diffusion model.
2. **Transfer learning and generate** for a more target-relevant chemical space.
3. **Conditioned generation** with property or structure guidance.
4. **End-to-end design loop** from generation to ranking and validation.

## How to Use This Section

Use these pages when you already understand the individual modules, but want a clearer picture of which sequence of steps fits your design goal.

```{note}
**Adapt a pre-trained model; don't train from scratch.** Training a 3D diffusion model from
scratch is expensive and rarely necessary. Start from one of the shipped pre-trained bases
(GEOM-drugs foundation, or QM9) and **fine-tune** it towards your target chemistry — this is the
default recommended path. The only case that forces training from scratch is chemistry that uses
atom types outside the pre-trained vocabulary (`H, B, C, N, O, F, Al, Si, P, S, Cl, As, Se, Br, I,
Hg, Bi`). Pre-trained checkpoints: <https://zenodo.org/records/19511401>. See
[Transfer Learning and Generate](transfer_learning.md).
```

## Workflow Comparison

| Workflow | Best for | Main idea | Typical next tutorial |
| :--- | :--- | :--- | :--- |
| Generate and filter | Broad exploration | Sample first, narrow later | [Tutorial 5](../tutorials/05_generation_overview.md), [Tutorial 9](../tutorials/09_analyze.md) |
| Transfer learning and generate | Target-focused exploration | Adapt the model to a more relevant dataset before sampling | [Tutorial 4](../tutorials/04_finetuning.md), [Tutorial 5](../tutorials/05_generation_overview.md) |
| Conditioned generation | Directed search | Bias generation towards a structure or property target | [Tutorial 6](../tutorials/06_structure_guided.md), [Tutorial 7](../tutorials/07_property_directed.md) |
| End-to-end design loop | Candidate prioritisation | Combine generation, filtering, optimisation, prediction, and ranking | [Tutorial 8](../tutorials/08_eval_predict.md), [Tutorial 9](../tutorials/09_analyze.md) |

## Pages

```{toctree}
:maxdepth: 1

generate_and_filter
transfer_learning
conditioned_generation
end_to_end
visualization
```
