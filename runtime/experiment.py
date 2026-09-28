"""CLI and factories connecting published task selections to the experiment loop."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

import yaml

from Methods import AWMState, HermesState, PlaybookState, ReasoningBankState
from Methods.base import AgentState
from .llm import EmbeddingClient, LLMClient, _infer_backend
from .presets import automatic_run_id, config_path, resolve_tasks, task_presets
from .runner import RunConfig, Runner


BENCHMARKS = {
    "appworld": ("appworld", "AppWorldBenchmark"),
    "browsecomp": ("browsecomp", "BrowseCompBenchmark"),
    "swebench": ("swebench", "SWEBenchBenchmark"),
    "tau2": ("tau2", "Tau2Benchmark"),
    "terminalbench": ("terminalbench", "TerminalBenchBenchmark"),
}
METHODS = {
    "ace": PlaybookState,
    "awm": AWMState,
    "reasoning_bank": ReasoningBankState,
    "hermes": HermesState,
}


class NoMemoryState(AgentState):
    def retrieve(self, task_description, k=5):
        return ""

    def update(self, reflection):
        pass

    def serialize(self):
        return '{"type": "baseline"}'

    def load_serialized(self, payload):
        if json.loads(payload).get("type") != "baseline":
            raise ValueError("Expected baseline snapshot")

    def stats(self):
        return {"type": "baseline", "approx_tokens": 0}


def load_selection(path: str, max_tasks: int | None = None) -> dict:
    """Load and validate a published JSON selection before starting a run."""
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = selection["task_ids"]
    if not ids or not all(isinstance(x, str) and x.strip() == x and x for x in ids):
        raise ValueError("Task IDs must be nonempty strings")
    if len(ids) != len(set(ids)) or selection["n_tasks"] != len(ids):
        raise ValueError("Task selection has duplicate IDs or an incorrect count")
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:12]
    if selection["sha256_12"] != digest:
        raise ValueError("Task selection checksum mismatch")
    benchmark = selection["benchmark"]
    selection["benchmark"] = "swebench" if benchmark == "swebench_lite" else benchmark
    if selection["benchmark"] not in BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    if max_tasks is not None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        selection["task_ids"] = ids[:max_tasks]
    return selection


def make_benchmark(selection: dict, config: dict, llm=None, *, checking=False):
    benchmark = selection["benchmark"]
    module, class_name = BENCHMARKS[benchmark]
    cls = getattr(importlib.import_module(f"Task.benchmarks.{module}"), class_name)
    options = dict(config.get("benchmark_options") or {})
    if any(key in options for key in ("task_ids", "max_tasks", "num_tasks")):
        raise ValueError("Select task IDs using task_file and max_tasks, not benchmark_options")
    options["task_ids"] = selection["task_ids"]
    output = Path(config["output_dir"]) / config["run_id"]
    if benchmark == "appworld":
        split = selection.get("split", "test_normal")
        if options.get("split", split) != split:
            raise ValueError("AppWorld split must match the task selection")
        options.update(split=split, experiment_name=config["run_id"])
    elif benchmark == "tau2":
        options.setdefault("domain", selection.get("domain", "retail"))
        options.setdefault("task_split_name", selection.get("split", "base"))
    elif benchmark == "swebench":
        options.setdefault("split", selection.get("split", "test"))
        options.setdefault("dataset_name", selection.get("dataset", "princeton-nlp/SWE-bench_Lite"))
        options.setdefault("predictions_path", str(output / "predictions.json"))
        options.setdefault("model_name", config["run_id"])
    elif benchmark == "browsecomp":
        options.setdefault("run_output_dir", None if checking else str(output / "benchmark"))
        options.setdefault("run_id", config["run_id"])
        if llm is not None and not options.get("judge_model"):
            options["judge_client"] = llm
    elif benchmark == "terminalbench":
        options.setdefault("run_output_dir", None if checking else str(output / "benchmark"))
        options.setdefault("run_id", config["run_id"])
        options.setdefault("agent_name", "opencode")
        model = config["model"]
        provider = _infer_backend(model, config.get("provider"))
        options.setdefault("model_name", model if model.startswith(provider + "/") else f"{provider}/{model}")
    return cls(**options)


def _provider_env(config):
    """Expose a custom key variable to benchmark SDKs used in this process."""
    provider = _infer_backend(config["model"], config.get("provider"))
    canonical = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                 "openrouter": "OPENROUTER_API_KEY"}[provider]
    source = config.get("api_key_env") or canonical
    if not os.environ.get(source):
        raise ValueError(f"Missing credential environment variable: {source}")
    os.environ[canonical] = os.environ[source]
    if provider == "openrouter":
        os.environ["ANTHROPIC_AUTH_TOKEN"] = os.environ[source]
        os.environ["ANTHROPIC_BASE_URL"] = "https://openrouter.ai/api"
        os.environ["ANTHROPIC_API_KEY"] = ""


def run(config: dict) -> dict:
    selection = load_selection(config["task_file"], config.get("max_tasks"))
    benchmark_name = selection["benchmark"]
    method = config["method"]
    if config.get("dry_run"):
        return {"benchmark": benchmark_name, "method": method,
                "model": config["model"],
                "provider": _infer_backend(config["model"], config.get("provider")),
                "task_file": config["task_file"], "run_id": config["run_id"],
                "run_dir": str(Path(config["output_dir"]) / config["run_id"]),
                "n_tasks": len(selection["task_ids"]), "task_ids": selection["task_ids"]}
    if config.get("check_tasks"):
        benchmark = make_benchmark(selection, config, checking=True)
        tasks = benchmark.get_tasks()
        _check_loaded_order(tasks, selection)
        return {"benchmark": benchmark_name, "n_tasks": len(tasks), "tasks_loaded": True}

    output = Path(config["output_dir"]) / config["run_id"]
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Run directory already contains files: {output}; choose a new run_id")
    _provider_env(config)
    llm = LLMClient(model=config["model"], temperature=config.get("temperature", 0.0),
                    provider=config.get("provider"), base_url=config.get("base_url"),
                    api_key_env=config.get("api_key_env"), extra_body=config.get("extra_body"))
    embedder = None
    # The original runner used embeddings for AWM/RB and keyword retrieval for Hermes.
    if method in {"awm", "reasoning_bank"}:
        embedder = EmbeddingClient(config["embedding_model"])
    if method == "baseline":
        state = NoMemoryState()
    else:
        state = METHODS[method](llm=llm, embedder=embedder, **(config.get("method_options") or {}))
    if config.get("load_state"):
        state.load_serialized(Path(config["load_state"]).read_text())
    benchmark = make_benchmark(selection, config, llm)
    tasks = benchmark.get_tasks()
    _check_loaded_order(tasks, selection)
    state_type = "playbook" if method == "ace" else method
    run_config = RunConfig(
        run_id=config["run_id"], state_type=state_type, benchmark=benchmark_name,
        model=config["model"], temperature=config.get("temperature", 0.0),
        log_dir=config["output_dir"], snapshot_interval=config.get("snapshot_interval", 1),
        retrieval_k=config.get("retrieval_k", 1 if method == "reasoning_bank" else 5),
        max_state_tokens=(config.get("method_options") or {}).get("max_tokens", 50_000),
        eval_only=method == "baseline" or config.get("eval_only", False),
        load_state=config.get("load_state"), task_file=config["task_file"], settings=config,
    )
    runner = Runner(state, benchmark, llm, run_config)
    try:
        return runner.run()
    finally:
        runner.logger.close()


def _check_loaded_order(tasks, selection):
    if [task["task_id"] for task in tasks] != selection["task_ids"]:
        raise ValueError("Loaded benchmark tasks do not match the selection's IDs and order")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: python run_experiment.py tau2 ace --tasks scale-up/airline-set1",
    )
    parser.add_argument("benchmark", nargs="?", choices=BENCHMARKS,
                        help="Load this benchmark's example config (defaults to Set1)")
    parser.add_argument("memory_method", nargs="?", choices=[*METHODS, "rb", "baseline"],
                        metavar="method", help="Memory method for the short command (default: ace)")
    parser.add_argument("--tasks", help="Short task selection name; see --list")
    parser.add_argument("--list", action="store_true", help="List task selections without running")
    parser.add_argument("--config", help="YAML runtime configuration")
    parser.add_argument("--task-file", help="Published JSON task selection")
    parser.add_argument("--method", choices=[*METHODS, "rb", "baseline"])
    parser.add_argument("--model")
    parser.add_argument("--provider", choices=["openai", "anthropic", "openrouter"])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--env-file", help="Explicit local credentials file to load")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection/config only")
    parser.add_argument("--check-tasks", action="store_true", help="Load real tasks without model calls")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value; supports benchmark_options.key=value")
    args = parser.parse_args(argv)
    if args.list:
        presets = [p for p in task_presets() if not args.benchmark or p["benchmark"] == args.benchmark]
        print(f"{'Benchmark':<15} {'--tasks':<40} {'Count':>5}  Group")
        for preset in presets:
            print(f"{preset['benchmark']:<15} {preset['tasks']:<40} "
                  f"{preset['n_tasks']:>5}  {preset['selection_group']}")
        return presets
    if args.benchmark and args.config:
        parser.error("Use either a benchmark short command or --config")
    if args.tasks and not args.benchmark:
        parser.error("--tasks requires a benchmark; use --task-file with custom configs")
    if args.tasks and args.task_file:
        parser.error("Use either --tasks or --task-file")
    if args.memory_method and args.method:
        parser.error("Specify the method either positionally or with --method")
    config = {"method": "ace", "model": "gpt-5.4", "run_id": "run",
              "output_dir": "runs", "embedding_model": "all-MiniLM-L6-v2"}
    source = config_path(args.benchmark) if args.benchmark else args.config
    if source:
        try:
            loaded = yaml.safe_load(Path(source).read_text())
        except (OSError, yaml.YAMLError) as exc:
            parser.error(str(exc))
        if not isinstance(loaded, dict):
            parser.error("Config must be a YAML mapping")
        config.update(loaded)
    if args.benchmark:
        try:
            config["task_file"] = resolve_tasks(args.benchmark, args.tasks or "set1")
        except ValueError as exc:
            parser.error(str(exc))
    if args.memory_method:
        config["method"] = args.memory_method
    for key in ("task_file", "method", "model", "provider", "max_tasks", "run_id", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for override in args.set:
        key, separator, value = override.partition("=")
        if not separator:
            parser.error("--set requires KEY=VALUE")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(value)
    overridden = {value.partition("=")[0] for value in args.set}
    if args.benchmark and (args.model or "model" in overridden) and not (args.provider or "provider" in overridden):
        # A new model should not inherit the example's provider accidentally.
        config.pop("provider", None)
    config["dry_run"] = args.dry_run or config.get("dry_run", False)
    config["check_tasks"] = args.check_tasks or config.get("check_tasks", False)
    allowed = {
        "task_file", "method", "model", "provider", "base_url", "api_key_env",
        "extra_body", "temperature", "max_tasks", "run_id", "output_dir",
        "embedding_model", "method_options", "benchmark_options", "retrieval_k",
        "snapshot_interval", "eval_only", "load_state", "dry_run", "check_tasks",
    }
    unknown = config.keys() - allowed
    if unknown:
        parser.error(f"Unknown config fields: {', '.join(sorted(unknown))}")
    config["method"] = "reasoning_bank" if config["method"] == "rb" else config["method"]
    if not config.get("task_file"):
        parser.error("--task-file or task_file in the config is required")
    if config["method"] not in {*METHODS, "baseline"}:
        parser.error("Unsupported method")
    if args.benchmark:
        try:
            selection = load_selection(config["task_file"], config.get("max_tasks"))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if selection["benchmark"] != args.benchmark:
            parser.error("Task selection does not match the requested benchmark")
        if args.run_id is None and "run_id" not in overridden:
            config["run_id"] = automatic_run_id(args.benchmark, selection, config)
    if config.get("snapshot_interval", 1) < 1:
        parser.error("snapshot_interval must be positive")
    if Path(config["run_id"]).name != config["run_id"] or config["run_id"] in {"", ".", ".."}:
        parser.error("run_id must be a single directory name")
    if args.env_file:
        from dotenv import load_dotenv
        if not Path(args.env_file).is_file():
            parser.error("env_file does not exist")
        load_dotenv(args.env_file, override=False)
    # Some optional benchmark libraries otherwise search parent directories for .env.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    try:
        result = run(config)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"{exc}\nSee Task/RUNNING.md for benchmark setup.\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
