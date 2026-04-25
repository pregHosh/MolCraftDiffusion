# Workflow Example: Transfer Learning and Generate

This workflow adapts a pre-trained model to a more target-relevant dataset before generation. The main goal is to shift the model from broad chemical knowledge toward a narrower and more useful design space.

## When to Use This

Use this workflow when:

- you have a set of known molecules related to your design problem,
- unconditional generation from the base model is too broad,
- you want new molecules that remain closer to a task-relevant chemical domain.

## Conceptual Flow

```text
[Pre-trained diffusion model]
            |
            v
[Transfer learning on target-relevant molecules]
            |
            v
[Generate molecules in adapted chemical space]
            |
            v
        [Analyze and filter]
```

## What Changes After Transfer Learning

Transfer learning changes the part of chemical space the model tends to sample from. Instead of correcting the output after generation, you first adapt the model so that its default generations are already closer to your target chemistry.

## Typical Outcome

Compared with direct sampling from the base model, this workflow often gives:

- molecules that look more like the chemistry you care about,
- less wasted sampling in irrelevant regions of chemical space,
- a better starting point for later conditional generation.

## Where to Go Next

- [Tutorial 4: Fine-Tuning](../tutorials/04_finetuning.md)
- [Tutorial 5: Generation Overview](../tutorials/05_generation_overview.md)
- [Tutorial 9: Analysis](../tutorials/09_analyze.md)
