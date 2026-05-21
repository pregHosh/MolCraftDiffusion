# Tutorial 10: Generation Parameter Sweeps

This tutorial explains how to run generation sweeps with the packaged sweep module:

```text
MolecularDiffusion.runmodes.generate.sweep
```

The command-line entry point is:

```bash
MolCraftDiff generate-sweep <sweep_config.yaml>
```

The repository root also keeps a compatibility shim:

```bash
python sweep.py <sweep_config.yaml>
```

Both commands call the same implementation.

Use sweeps when you want to optimize controlled generation settings for **inpainting**, **outpainting**, **classifier-free guidance (CFG)**, **gradient guidance (GG)**, or **hybrid CFG/GG** without manually launching each run.

---

## 1. What a Sweep Does

A sweep repeatedly runs:

```bash
MolCraftDiff generate <base_config> <Hydra overrides...>
```

For each parameter combination, the sweep creates a run-specific output directory and then calls your evaluation workflow:

```bash
bash <eval_script> <run_output_dir>
```

After the workflow finishes, the sweep reads registered CSV files from the run output directory and appends one row to:

```text
<sweep_dir>/summary.csv
```

This gives you a single table containing the generation parameters and the metrics for each run.

---

## 2. Minimal Command

Start with a dry-run. This prints the exact generation and evaluation commands without writing files:

```bash
MolCraftDiff generate-sweep configs/sweep_example.yaml --dry-run --max-runs 1
```

Equivalent legacy command:

```bash
python sweep.py configs/sweep_example.yaml --dry-run --max-runs 1
```

Then launch real runs:

```bash
MolCraftDiff generate-sweep configs/sweep_example.yaml --max-runs 5
```

Useful options:

| Option | Meaning |
| :--- | :--- |
| `--dry-run` | Print commands only; no generation, eval, logs, summary, or copied config. |
| `--max-runs N` | Run at most `N` new parameter combinations. |
| `--skip-gen` | Skip generation and only run evaluation/collection. |
| `--skip-eval` | Skip evaluation and only run generation/collection. |
| `--retry-failed` | Retry rows marked failed or interrupted in `summary.csv`. |

---

## 3. Sweep Config Anatomy

A complete commented template is available at:

```text
configs/sweep_example.yaml
```

The main fields are:

```yaml
base_config: configs/lilas_cfg/gen_3_nto_op_cfg_2.yaml
eval_script: workflow_hybrid_sf_a3_2.sh
sweep_dir: sweep_results/example_sweep
name: example_sweep
```

`base_config` is the normal generation YAML you would pass to `MolCraftDiff generate`.

`eval_script` is a shell workflow that receives the run output directory as its only argument:

```bash
bash workflow_hybrid_sf_a3_2.sh sweep_results/example_sweep/<run_name>
```

`sweep_dir` contains:

```text
sweep_results/example_sweep/
├── sweep_config.yaml
├── summary.csv
└── example_sweep_<parameter_tag>/
```

Each run directory is also passed to generation as:

```text
interference.output_path=<run_output_dir>
```

---

## 4. Varying Parameters

Use `parameters` for scalar Hydra overrides:

```yaml
parameters:
  interference.condition_configs.cfg_scale: [1.5, 2.0, 2.5]
  interference.condition_configs.outpaint_cfgs.t_start: "0.55:0.95:0.20"
  interference.condition_configs.outpaint_cfgs.spread: [0.5, 1.0, 1.5]
  diffusion_steps: [50, 300]
```

The range string is inclusive. This:

```yaml
"0.55:0.95:0.20"
```

expands to:

```text
0.55, 0.75, 0.95
```

Use `list_parameters` when the Hydra value itself is a list:

```yaml
list_parameters:
  interference.mol_size:
    - [23, 45]
    - [23, 55]
    - [23, 65]
  interference.negative_target_values:
    - [-10, 0, -1]
    - [-15, 0, -1]
```

Every scalar value and every list-valued choice is combined by Cartesian product.

Use `extra_overrides` for settings applied to every run:

```yaml
extra_overrides:
  - "interference.num_generate=60"
  - "interference.batch_size=2"
```

Override order is:

1. varied sweep parameters,
2. generated `interference.output_path`,
3. `extra_overrides`.

Later overrides win if the same Hydra key appears more than once.

---

## 5. Controlled Generation Parameters to Sweep

### Inpainting

For inpainting, sweep the parameters that control how much of the masked region changes and how strongly it is constrained:

```yaml
base_config: configs/my_inpaint.yaml

parameters:
  interference.condition_configs.inpaint_cfgs.denoising_strength: [0.4, 0.6, 0.8, 1.0]
  interference.condition_configs.inpaint_cfgs.constraint_strength: [0.6, 0.8, 1.0]
  interference.condition_configs.inpaint_cfgs.scale_factor: [1.0, 1.1, 1.2]
  diffusion_steps: [300, 600]
```

Use this when you want to find the best balance between scaffold retention and novelty.

### Outpainting

For outpainting, sweep seed placement and denoising length:

```yaml
base_config: configs/my_outpaint.yaml

parameters:
  interference.condition_configs.outpaint_cfgs.t_start: [0.55, 0.75, 0.95]
  interference.condition_configs.outpaint_cfgs.seed_dist: [1.0, 1.5, 2.0]
  interference.condition_configs.outpaint_cfgs.min_dist: [1.5, 2.0, 2.5]
  interference.condition_configs.outpaint_cfgs.spread: [0.5, 1.0, 1.5]
  interference.condition_configs.outpaint_cfgs.constraint_strength: [0.6, 0.8]
```

Use this when some runs clash with the scaffold, fail to connect, or generate too little diversity.

### CFG

For classifier-free guidance, the central parameter is `cfg_scale`:

```yaml
base_config: docs/cfg_examples/gen_cfg.yaml

parameters:
  interference.condition_configs.cfg_scale: [0.5, 1.0, 1.5, 2.0, 3.0]
  diffusion_steps: [300, 600]

list_parameters:
  interference.target_values:
    - [3.0, 1.5]
    - [3.2, 1.4]
  interference.negative_target_values:
    - [1.0, 3.0]
    - [0.5, 3.5]
```

Use this to tune how strongly the conditional model is pushed toward the desired property regime.

### Gradient Guidance

For GG, sweep the gradient scale, clipping, and guidance window:

```yaml
base_config: docs/cfg_examples/gen_gradient_guidance.yaml

parameters:
  interference.condition_configs.gg_scale: [0.0001, 0.0005, 0.001]
  interference.condition_configs.max_norm: [0.0005, 0.001, 0.005]
  interference.condition_configs.guidance_at: [1, 50, 100]
  interference.condition_configs.guidance_stop: [0, 50]
  interference.condition_configs.n_backwards: [0, 1, 3]
```

Use this when guidance is too weak, unstable, or over-optimizes at the cost of molecular quality.

### Hybrid CFG/GG

For hybrid guidance, sweep both CFG and GG strengths:

```yaml
base_config: docs/cfg_examples/gen_hybrid_cfg_gg.yaml

parameters:
  interference.condition_configs.cfg_scale: [0.5, 1.0, 1.5]
  interference.condition_configs.gg_scale: [0.0001, 0.0005, 0.001]
  interference.condition_configs.max_norm: [0.0005, 0.001]
  interference.condition_configs.n_backwards: [0, 1, 3]
```

This is useful when CFG gives the correct broad direction but GG is needed to sharpen a numeric objective.

---

## 6. Registering Workflow CSVs

Your evaluation workflow should write CSVs under the run output directory:

```text
<run_output_dir>/optimized_xyz/merged_hits.csv
<run_output_dir>/postbuster_metrics.csv
<run_output_dir>/custom_metrics.csv
```

Register those files in `collect`:

```yaml
collect:
  - path: optimized_xyz/merged_hits.csv
    metrics:
      n_total:
        agg: len
      hit_rate_both:
        column: both_hit
        agg: mean
        round: 4
      n_both_hit:
        column: both_hit
        agg: sum

  - path: custom_metrics.csv
    metrics:
      mean_custom_score:
        column: score
        agg: mean
        round: 4
      n_custom_pass:
        column: passed
        agg: sum
```

`path` is always relative to the individual run output directory.

Supported aggregations:

| Aggregation | Meaning |
| :--- | :--- |
| `len` | Total number of rows. `column` is optional. |
| `mean` | Mean of a column after dropping missing values. |
| `sum` | Sum of a column after dropping missing values. |
| `count` | Non-null count of a column. |

You can drop sentinel values before aggregation:

```yaml
mean_eff_dihedral:
  column: eff_dihedral
  agg: mean
  exclude_value: -1.0
  round: 3
```

:::{important}
If `collect` is omitted, the sweep uses built-in defaults for:

- `optimized_xyz/merged_hits.csv`
- `postbuster_metrics.csv`
- `xyz_analysis.csv`
- `scscore_results.csv`
- `u.csv`

If `collect` is present, it replaces the built-in defaults. Include every metric you want in `summary.csv`.
:::

---

## 7. Bayesian Optimization Objective

Grid search evaluates combinations in deterministic order. Bayesian search chooses from the same candidate grid using previous rows in `summary.csv`.

```yaml
search:
  method: bayesian
  include_base_config: true
  objective:
    metric: hit_rate_both
    mode: max
  bayesian:
    n_initial: 3
    random_state: 0
```

The objective metric must be collected into `summary.csv`. For example, this objective:

```yaml
objective:
  metric: hit_rate_both
  mode: max
```

requires:

```yaml
collect:
  - path: optimized_xyz/merged_hits.csv
    metrics:
      hit_rate_both:
        column: both_hit
        agg: mean
```

For minimization:

```yaml
objective:
  metric: mean_rmsd
  mode: min
```

Bayesian mode requires Optuna. If Optuna is unavailable, install it or switch to:

```yaml
search:
  method: grid
```

---

## 8. Resume Behavior

Each completed or failed run is recorded in:

```text
<sweep_dir>/summary.csv
```

By default, any recorded configuration is skipped on the next launch:

```bash
MolCraftDiff generate-sweep configs/sweep_example.yaml --max-runs 10
```

To retry failed or interrupted rows:

```bash
MolCraftDiff generate-sweep configs/sweep_example.yaml --retry-failed --max-runs 10
```

The configuration identity is based on the swept parameter values, not the output directory name.

---

## 9. Practical Workflow

1. Create a normal generation config for the controlled generation mode.
2. Create a sweep config that points to it as `base_config`.
3. Add only the parameters you actually want to vary.
4. Make the eval workflow write CSV files under `<run_output_dir>`.
5. Register those CSV files and metrics under `collect`.
6. Start with:

```bash
MolCraftDiff generate-sweep configs/sweep_example.yaml --dry-run --max-runs 1
```

7. Launch a small batch:

```bash
MolCraftDiff generate-sweep configs/sweep_example.yaml --max-runs 3
```

8. Inspect:

```text
sweep_results/example_sweep/summary.csv
```

9. Continue the sweep when ready:

```bash
MolCraftDiff generate-sweep configs/sweep_example.yaml --max-runs 20
```

The sweep will skip configurations already present in `summary.csv`.
