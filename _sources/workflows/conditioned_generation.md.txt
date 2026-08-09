# Workflow Example: Conditioned Generation

This workflow is for cases where you do not just want plausible molecules, but molecules that satisfy a more specific design objective. In MolCraftDiffusion, the conditioning can be either **property-based** or **structure-based**.

## Two Main Modes

| Mode | Question it answers | Typical method |
| :--- | :--- | :--- |
| Property-conditioned | "Can I bias generation towards a target value or regime?" | CFG, GG, or hybrid guidance |
| Structure-conditioned | "Can I preserve one part of the molecule while redesigning or extending another?" | Inpainting or outpainting |

## Conceptual Flow

```text
                 [Base or adapted model]
                         |
          +--------------+--------------+
          |                             |
          v                             v
 [Property guidance]           [Structure guidance]
          |                             |
          v                             v
[Property-directed generation] [Structure-directed generation]
```

## What Changes with Conditioning

This workflow adds a design constraint to generation. Rather than only sampling plausible molecules, you guide the model towards a target property profile or a target structural pattern.

## Typical Outcome

This is often the right workflow when you want:

- molecules around a known scaffold,
- candidates with a target property profile,
- a more directed search than unconditional generation can provide.

## Where to Go Next

- [Tutorial 6: Structure-Guided Generation](../tutorials/06_structure_guided.md)
- [Tutorial 7: Property-Directed Generation](../tutorials/07_property_directed.md)
- [Applications](../applications/index.md)
