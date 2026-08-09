# Application: Local Chemical Space Exploration

This workflow uses **inpainting** to explore structural variants around a known reference molecule while keeping the rest of the 3D scaffold fixed. It is useful when you want to **probe a local neighbourhood** of chemical space rather than generate fully unconstrained molecules.

---

## When to Use This

::::{tip}
Use local exploration when you want to:

- **preserve** a known active or synthetically meaningful scaffold,
- **vary** only a specific substituent, ring system, or side chain,
- **compare** how controlled structural edits affect downstream properties.
::::

In the MolCraftDiffusion paper, this idea is used to explore variants around a reference ligand while retaining the chemically important core.

---

## Conceptual Workflow

```text
[Reference XYZ structure]
          |
          v
[Select atoms to regenerate]
          |
          v
  [Inpainting: fix scaffold,
   redraw selected region]
          |
          v
[Compare variants: diversity,
 validity, downstream properties]
```

The key idea is not to search all of chemical space. Instead, you **stay near a known structure** and deliberately vary only the region that matters for your design question.

---

## Why This Matters

::::{note}
Local exploration is often the right mode when a full generative search is too broad. It helps answer questions such as:

- What happens if I **replace one substituent** while preserving the rest of the molecule?
- How much **structural diversity** can I obtain around a validated scaffold?
- Which **local edits** improve the property of interest without moving too far from a known design?
::::

---

## Where to Go Next

::::{seealso}
For the actual workflow, use:

- [Tutorial 6: Structure-Guided Generation](../tutorials/06_structure_guided.md) — inpainting mechanics
- [Tutorial 9: Analysis](../tutorials/09_analyze.md) — comparing the generated variants

For a concrete starting template, the corresponding example config is
[`docs/cfg_examples/gen_inpaint.yaml`](../cfg_examples/gen_inpaint.yaml).
::::
