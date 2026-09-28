# Running the task selections

Each benchmark adapter loads the actual task definition behind an ID, injects
memory into the agent, executes the task, and obtains its score. The shared
runner then reflects, updates memory, and saves logs and state snapshots.
The JSON files in `Set1_tasks/` and `scale_up_tasks/` select tasks and their order.

Run commands below from the repository root. Use a separate virtual environment
for each benchmark: in particular, AppWorld requires Pydantic 1 while the pinned
tau2 runtime requires Pydantic 2. Installing all benchmark extras together is
unsupported. Python 3.11 is suitable for AppWorld; Python 3.12 was used to validate
the other adapters.

After preparation, use `python run_experiment.py BENCHMARK METHOD` for Set1 or
add `--tasks scale-up/NAME` for a published subset. `python run_experiment.py --list`
shows all names. Short commands automatically choose the matching example config
and a new output directory. See the [command examples](../README.md#run-an-experiment).
The explicit `--config` commands below remain available for custom configurations.

## AppWorld

```bash
python3.11 -m venv .venv-appworld
source .venv-appworld/bin/activate
python -m pip install -e '.[appworld]'
python -m Task.prepare appworld
python run_experiment.py --config configs/appworld.yaml --check-tasks
python run_experiment.py --config configs/appworld.yaml --env-file .env --max-tasks 1
```

Preparation runs `appworld install` and downloads official data under
`data/appworld/data`. The adapter uses the bundled ACE ReAct prompt (source and
license in `Task/assets/`) and evaluates the completed environment with AppWorld.
Existing data can be supplied with `--set benchmark_options.data_dir=/path/to/data`.

The original environment referenced an unavailable editable AppWorld
`0.1.4.dev0` checkout. This public runtime uses `appworld==0.1.3.post1`, verified
to load all 300 listed AppWorld entries, render their prompts, and execute and
evaluate a task with a static agent. Exact score parity with the unavailable
checkout has not been established.

Official setup: <https://github.com/StonyBrookNLP/appworld>.
Protected AppWorld data is downloaded locally using its official bundle flow;
decrypted databases and task ground truth are excluded from version control.

## tau2

```bash
python3.12 -m venv .venv-tau2
source .venv-tau2/bin/activate
python -m pip install -e '.[tau2]'
python -m Task.prepare tau2
python run_experiment.py --config configs/tau2.yaml --check-tasks
python run_experiment.py --config configs/tau2.yaml --env-file .env --max-tasks 1
```

The package and data checkout are pinned to
`0ed2fd8d830a20657d89ae9c2efcc94838aa7129`. Preparation applies the four verified
files in `Task/assets/tau2_overlay/`: the original experiment used modified
airline/retail tasks and user-simulator guidance. The overlay retains these
changes and includes source hashes and the upstream MIT license. Simulations,
voice assets, and old run outputs are excluded. A fresh upstream checkout alone
does not reproduce this experiment data.

The adapter uses tau2's user simulator, tools, and official reward evaluation.
The sample config routes agent and user models through OpenRouter. Upstream
natural-language assertion evaluation also uses an OpenAI judge; set
`OPENAI_API_KEY` as well as `OPENROUTER_API_KEY`. Existing experiment data can be
supplied using `--set benchmark_options.data_dir=/path/to/tau2-data`.

Official implementation: <https://github.com/sierra-research/tau2-bench>.

## BrowseComp

```bash
python -m pip install -e '.[browsecomp]'
python -m Task.prepare browsecomp
python run_experiment.py --config configs/browsecomp.yaml --check-tasks
python run_experiment.py --config configs/browsecomp.yaml --env-file .env --max-tasks 1
```

Install Java 21 and set `JAVA_HOME` before building or searching the Pyserini
index. Preparation downloads/decrypts BrowseComp-Plus locally, downloads the
corpus, and builds a BM25 index. Corpus and index are multi-gigabyte resources.
The adapter runs the search-tool conversation and grades the final answer.

Individual preparation commands support custom paths and Hugging Face revisions:

```bash
python -m Task.prepare_browsecomp_data --help
python -m Task.prepare_browsecomp_corpus --help
python -m Task.prepare_browsecomp_qrels --help
```

The original BrowseComp Hugging Face revisions were not recorded. For a frozen
download, supply `--revision` to each relevant preparation command. Existing
resources are configured with `benchmark_options.data_path`, `corpus_path`, and
`corpus_index_path`. The local BM25 backend is available for small offline
fixtures; the sample experiment config uses Pyserini.

Datasets: <https://huggingface.co/datasets/Tevatron/browsecomp-plus> and
<https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus>.

## SWE-bench Lite

```bash
python -m pip install -e '.[swebench]'
python -m Task.prepare swebench
python run_experiment.py --config configs/swebench.yaml --check-tasks
python -m modal setup
python run_experiment.py --config configs/swebench.yaml --env-file .env --max-tasks 1
```

Java 21 and Git are required for the official retrieval/prompt pipeline. The
dataset is pinned to the original cached revision
`6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`. The adapter checks out each task's base
commit, retrieves repository context, generates a patch, and calls the official
SWE-bench harness. The sample uses Modal, matching the original experiment setup.
Preparation also restores the original package's small Modal compatibility
patch (cgroup initialization and sandbox file transfer), after checking the
installed version and file hashes.
For local Docker evaluation, set `--set benchmark_options.use_modal=false` and
provide a running Docker daemon with sufficient disk space.

`--set benchmark_options.solve_only=true` generates predictions without scoring.
Unscored outcomes are logged separately and do not count as evaluated failures
or update memory. Official implementation: <https://github.com/SWE-bench/SWE-bench>.

## Terminal-Bench

```bash
python -m pip install -e '.[terminalbench]'
python -m Task.prepare terminalbench
python run_experiment.py --config configs/terminalbench.yaml --check-tasks
python -m modal setup
python run_experiment.py --config configs/terminalbench.yaml --env-file .env --max-tasks 1
```

Task definitions are pinned to Terminal-Bench 2 commit
`53ff2b87d621bdb97b455671f2bd9728b7d86c11`. Harbor `0.4.0` launches the installed
agent, injects memory into the task instruction, runs the task environment, and
reads its verifier reward. The sample uses OpenCode through OpenRouter and Modal,
matching the scale-up configuration. Set `benchmark_options.environment=docker`
to use a configured local Docker daemon instead.

Implementation: <https://github.com/harbor-framework/harbor>.
Tasks: <https://github.com/harbor-framework/terminal-bench-2>.

Preparation restores two original Harbor fixes: Modal directory creation and
Claude Code token forwarding. They are recorded as exact replacements in
`Task/assets/runtime_patches/` with upstream licenses and before/after hashes.
Only the pinned package versions are supported; unexpected local edits cause
an error rather than being overwritten. These package fixes can also be applied
explicitly with `python -m Task.patch_runtime swebench` or
`python -m Task.patch_runtime terminalbench` in the corresponding environment.

## Configuration and results

`configs/` contains one runnable example per benchmark. Top-level fields select
the method, model/provider, task file, output directory, retrieval count, and
snapshot interval. `method_options` and `benchmark_options` map directly to the
selected class constructors. Nested CLI overrides use `--set`, for example:

```bash
python run_experiment.py --config configs/tau2.yaml --method awm \
  --set method_options.induction_window=3 --run-id awm_tau2 --env-file .env
```

Install the `embeddings` extra for AWM/ReasoningBank. Their default embedder is
`all-MiniLM-L6-v2`, which downloads weights on first use. Hermes uses the original
runner's keyword retrieval. `--load-state` is specified as `--set load_state=...`;
use a new run ID and a task selection containing the intended remaining tasks.

Outputs include `config.json`, `log.jsonl`, `summary.json`, and per-step
`state_snapshots/`, plus benchmark artifacts. The runner's cost fields retain
historical estimates and do not include every external model call; use provider
billing records for actual cost.

The public tests make no live model calls. Additional migration checks loaded
every published selection from actual local resources; AppWorld execution and
SWE-bench/BrowseComp retrieval were also exercised locally. Full live-model
accuracy and Docker/Modal/Harbor evaluations were not rerun during migration.
