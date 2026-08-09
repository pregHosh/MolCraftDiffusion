# Workflow Example: End-to-End Design Loop

This workflow combines generation with downstream analysis and prioritisation. It is the most complete pattern in the framework and is the closest to how candidates are handled in a real design campaign.

## Conceptual Flow

```text
[Generate molecules]
        |
        v
[Analyse outputs]
        |
        v
[Filter valid molecules]
        |
        v
[Optimise structures]
        |
        v
[Evaluate with xTB or predictors]
        |
        v
[Rank candidates]
```

## What This Adds Beyond Generation

Generation alone is usually not enough. This workflow separates:

- molecules that are merely generated,
- molecules that are chemically reasonable,
- molecules that remain promising after optimisation and evaluation.

That distinction is what turns a raw sample set into a candidate-selection pipeline.

## Example Decision Table

| Stage | Main question | Typical outcome |
| :--- | :--- | :--- |
| Generate | What can the model propose? | Raw molecule pool |
| Analyse | Are the structures valid and diverse? | Cleaned subset |
| Optimise | Do the structures remain sensible after relaxation? | More reliable geometries |
| Evaluate | Do they satisfy the target property or descriptor? | Scored candidates |
| Rank | Which molecules deserve follow-up? | Final shortlist |

## When to Use This

Use this workflow when:

- you need a ranked candidate list rather than a raw sample set,
- downstream physics- or chemistry-based checks matter,
- you want to compare candidates consistently across multiple filters.

## Where to Go Next

- [Tutorial 8: Property Prediction and Evaluation](../tutorials/08_eval_predict.md)
- [Tutorial 9: Analysis](../tutorials/09_analyze.md)
- [Applications](../applications/index.md)
