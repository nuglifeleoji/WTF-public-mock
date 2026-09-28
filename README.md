# Why Do Self-Improving Agents Fail?

Cross-task memory experiments with ACE, Agent Workflow Memory, ReasoningBank,
and Hermes on AppWorld, BrowseComp, SWE-bench Lite, tau2, and Terminal-Bench.

- [Methods](Methods/README.md): memory implementations and their shared interface.
- [Task selections](Task/README.md): the ordered Set1 and scale-up task lists.
- [Running benchmarks](Task/RUNNING.md): installation, data preparation, and execution.

## Run an experiment

Run these commands from the repository root, in the selected benchmark's
environment. Install dependencies and prepare data once, as described in the
[running guide](Task/RUNNING.md). For example, for tau2:

```bash
python -m pip install -e '.[tau2]'
python -m Task.prepare tau2
python run_experiment.py tau2 --check-tasks
```

The short command is `python run_experiment.py BENCHMARK METHOD`. It loads
`configs/BENCHMARK.yaml` and defaults to the full Set1 selection. Export provider
credentials in your shell, or append `--env-file .env` to a run; `.env.example`
lists the keys. Examples:

```bash
python run_experiment.py tau2 ace
python run_experiment.py appworld awm
python run_experiment.py browsecomp rb
python run_experiment.py swebench hermes
python run_experiment.py terminalbench baseline
```

Methods are `ace` (the default), `awm`, `rb` / `reasoning_bank`, `hermes`, and
`baseline`. AWM and ReasoningBank also require
`python -m pip install -e '.[embeddings]'`. Baseline runs without cross-task
memory or reflection.

Select a scale-up subset without writing a JSON path:

```bash
python run_experiment.py tau2 ace --tasks scale-up/airline-set1
python run_experiment.py appworld awm --tasks scale-up/challenge-set1
python run_experiment.py browsecomp rb --tasks scale-up/subset3
python run_experiment.py swebench hermes --tasks scale-up/django
python run_experiment.py terminalbench baseline --tasks scale-up/set2
```

List all 30 selections, or just one benchmark's selections:

```bash
python run_experiment.py --list
python run_experiment.py appworld --list
```

The list labels AppWorld's canonical and additional random subsets separately.
`set1` is the initial full selection; `scale-up/set1` is a distinct, smaller
selection where available.

Preview a command, test one task, or select another model:

```bash
python run_experiment.py tau2 ace --tasks scale-up/airline-set1 --dry-run
python run_experiment.py tau2 ace --env-file .env --max-tasks 1
python run_experiment.py tau2 hermes --model openrouter/anthropic/claude-sonnet-4.6
```

`--dry-run` shows the selected IDs, model, provider, and output directory without
loading datasets or calling a model. `--check-tasks` loads real task definitions
without model calls. `--max-tasks N` takes the first N IDs in the published order.
An explicit `--model` infers its provider from the model name; use `--provider`
to override routing. It changes the agent/reflection model; benchmark user and
judge models retain their configured settings.

Short commands create a distinct `runs/<benchmark>_<selection>_<method>_<model>_<UTC time>/`
directory each time. Supply `--run-id NAME` for a fixed name; nonempty output
directories are never overwritten. Each run saves its resolved config, task log,
summary, and memory snapshots.

These commands use runnable example settings. They do not encode every original
paper configuration (model-specific budgets, retrieval settings, and installed
agents varied between experiments). Customize options with `--set`, or keep the
explicit config interface for a fully specified experiment:

```bash
python run_experiment.py --config configs/tau2.yaml \
  --task-file Task/scale_up_tasks/tau2/tau2_airline_set1_30.json \
  --method hermes --run-id hermes_airline_set1 --env-file .env
```

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The offline suite checks task integrity, benchmark execution boundaries, model
request construction, and the complete memory loop using fake model clients.
Benchmark-specific loading and environment checks are described in the running
guide. Full benchmark execution requires its environment and live model access.
