# DEMO experiment runtime

The runtime standardizes experiment configuration, GPU selection, JSONL
recording, plotting, completion detection, and paper-table generation without
changing the core methods in `evo.py`, `MOEA/`, `egnn/`, or
`equivariant_diffusion/`.

## Supported environment

- WSL Ubuntu 18.04
- Conda environment: `cgm310`
- GPU visibility is set by the launcher before the child process imports
  PyTorch.

The shell entry points activate the environment automatically. Override the
environment name only when necessary:

```bash
DEMO_CONDA_ENV=cgm310 bash scripts/run_suite.sh --suite qm9_cmop --gpus 0
```

## One-command suites

```bash
# Validate command expansion without running a model.
bash scripts/run_suite.sh --suite qm9_cmop --gpus 0,1 --dry-run

# Run, skip exact completed runs, plot, and summarize.
bash scripts/run_suite.sh \
  --suite qm9_cmop \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize

# Run all five experiment families.
bash scripts/run_all_revision.sh --gpus 0,1,2,3

# List or select paper-table tasks.
bash scripts/run_suite.sh --suite qm9_mop --list-tasks
bash scripts/run_suite.sh \
  --suite qm9_mop \
  --task qm9_mop_saes_wo_co \
  --task qm9_mop_saes_wo_mt \
  --gpus 0,1
```

Available suites:

- `qm9_single`
- `qm9_mop`
- `qm9_cmop`
- `qm9_struct_cmop`
- `docking_mop`

Every suite is a JSON document under `configs/suites/`. A task invokes either a
legacy script through `entrypoint` or a refactored module through `module`.
Paper-table tasks declare their diffusion `backbone` and an environment mapping
such as `{"DEMO_USE_EDM": "1"}`. Directly invoking a legacy script still keeps
that script's original backbone default; the suite override only makes a paper
command explicit and reproducible.
Files whose contents define an experiment, such as a custom CMOP problem JSON,
belong in the task's `identity_files` list. Their SHA-256 hashes participate in
the run id, so changing an objective or constraint cannot be mistaken for an
already completed run.

`--task TASK_ID` is repeatable and filters a suite without creating another
configuration file. `--list-tasks` reports task ids, method labels, backbones,
and known duplicate-command notes.

The exact mapping for Tables 1–5, imported ablations, duplicate commands, and
the reviewer QM9-CMOP experiment is documented in
[`PAPER_EXPERIMENT_COMMANDS_ZH.md`](PAPER_EXPERIMENT_COMMANDS_ZH.md).

## Legacy-reference consistency

The imported Table 2, Table 3, and Table 5 ablations can be compared with the
provided sibling `GeoLDM` reference directory:

```bash
python -m demo_runtime.consistency --reference-root ../GeoLDM
```

The audit normalizes only launcher-managed GPU visibility, the explicit
backbone environment adapter (while preserving the old default), and the
unused optimizer initialization in the duplicate Top-N reference. Other
differences are reported as mismatches.

## Result contract

Each run has a stable identity derived from its task, method, seed, budget,
checkpoint hashes, configuration, and Git commit:

```text
results/<suite>/<problem>/<method>/seed_<seed>_<hash>/
  manifest.json
  metrics.jsonl
  population.jsonl
  lineage.jsonl
  events.jsonl
  stdout.log
  stderr.log
  .complete
```

JSONL is the canonical source for every metric. PNG, PDF, CSV, Markdown, and
LaTeX files are derived artifacts and can be regenerated:

```bash
bash scripts/plot_suite.sh results/qm9_cmop
bash scripts/summarize_suite.sh \
  results/qm9_cmop \
  --suite-config configs/suites/qm9_cmop.json
```

A run is skipped only when `.complete` contains the exact run id and the last
event is `run_end` with `status=completed`. Interrupted and failed runs are not
silently treated as complete.

## Custom QM9-CMOP interface

`configs/problems/qm9_frontier_alignment.json` is an example, not a hard-coded
problem. A new QM9-CMOP is created by copying the file and changing:

- objectives and min/max directions;
- canonical units and fixed HV bounds;
- direct or derived properties;
- inequality thresholds;
- equality targets, tolerances, and scales;
- population size, generations, noise policy, and selection policy.

Supported derived-property operations are `mean`, `sum`, and `difference`.
Supported constraints are:

```json
{
  "name": "minimum_gap",
  "kind": "inequality",
  "property": "gap",
  "operator": ">=",
  "threshold": 6.0,
  "scale": 1.0,
  "unit": "eV"
}
```

```json
{
  "name": "frontier_alignment",
  "kind": "equality",
  "property": "frontier_midpoint",
  "target": -4.0,
  "tolerance": 0.15,
  "scale": 0.15,
  "unit": "eV"
}
```

Run a config directly:

```bash
python -m demo_runtime.qm9_cmop \
  --problem-config configs/problems/qm9_frontier_alignment.json \
  --validate-config
```

The QM9-CMOP runtime uses only the bundled EGNN property predictors. It does
not call xTB or DFT. Energy predictions are converted from the legacy meV
representation to canonical eV before objectives and constraints are evaluated.
The record metadata identifies the source as `egnn_property_predictor`.

## Summary tables

`summary.jsonl` contains one record per problem, method, and metric. Derived
tables report mean, standard deviation, 95% confidence interval, median, range,
and completed-run count. LaTeX tables bold the best method and underline the
second-best according to the metric direction declared by the suite.

Incomplete runs are excluded rather than padded with their last generation.
