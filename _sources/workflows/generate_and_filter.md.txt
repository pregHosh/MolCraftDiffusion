# Workflow Example: Generate and Filter

This is the simplest end-to-end workflow. You start from a pre-trained diffusion model, generate a broad pool of molecules, and then filter the output based on validity, uniqueness, or downstream criteria.

## When to Use This

Use this workflow when:

- you want a fast baseline,
- you do not yet have a task-specific fine-tuned model,
- your first goal is to understand what the pre-trained model can already produce.

## Conceptual Flow

```text
[Pre-trained diffusion model]
            |
            v
   [Generate molecules]
            |
            v
   [Analyze and filter]
            |
            v
      [Candidate set]
```

## What This Gives You

This workflow keeps the first pass simple. You use the pre-trained model to produce a broad set of structures, then use analysis and filtering to decide which molecules are worth keeping.

## Typical Outcome

You usually obtain:

- a sense of the chemical space already covered by the model,
- a baseline validity and diversity profile,
- an initial candidate pool for more focused follow-up.

## Where to Go Next

- [Tutorial 5: Generation Overview](../tutorials/05_generation_overview.md)
- [Tutorial 8: Property Prediction and Evaluation](../tutorials/08_eval_predict.md)
- [Tutorial 9: Analysis](../tutorials/09_analyze.md)
