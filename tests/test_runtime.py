"""Exercise the public entry point and full memory loop without API calls."""

import json
from pathlib import Path

import numpy as np
import pytest

from runtime import experiment


ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    def __init__(self, model, **kwargs):
        self.model = model

    def generate(self, prompt, max_tokens=2048, system=None):
        if system and "memory curator" in system:
            return "# Memory Item 1\n## Title Verify\n## Description General\n## Content Check the result."
        if "Hermes' skill self-improvement" in prompt:
            return '{"actions": []}'
        if "workflows" in prompt.lower():
            return '{"workflows": []}'
        if "master curator" in prompt:
            return '{"operations": [{"type": "ADD", "section": "others", "content": "Verify results."}]}'
        return '{"key_insight": "Verify results."}'


class FakeEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def embed(self, texts):
        return np.ones((len(texts), 1), dtype=np.float32)

    def top_k_indices(self, query, candidates, k):
        return list(range(min(k, len(candidates))))


class FakeBenchmark:
    def __init__(self, ids):
        self.ids = ids
        self.contexts = []

    def get_tasks(self):
        return [{"task_id": tid, "description": f"Solve {tid}"} for tid in self.ids]

    def solve(self, task, context, llm):
        self.contexts.append(context)
        return "A real adapter executes here.", {"outcome": "success", "score": 1.0, "details": "Passed"}


@pytest.mark.parametrize("method", ["ace", "awm", "reasoning_bank", "hermes", "baseline"])
def test_cli_runs_selected_tasks_and_saves_state(method, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-placeholder")
    monkeypatch.setattr(experiment, "LLMClient", FakeLLM)
    monkeypatch.setattr(experiment, "EmbeddingClient", FakeEmbedder)
    benchmarks = []

    def factory(selection, config, llm=None, checking=False):
        benchmark = FakeBenchmark(selection["task_ids"])
        benchmarks.append(benchmark)
        return benchmark

    monkeypatch.setattr(experiment, "make_benchmark", factory)
    result = experiment.main([
        "--task-file", str(ROOT / "Task/Set1_tasks/browsecomp_tasks.json"),
        "--method", method, "--max-tasks", "2", "--output-dir", str(tmp_path),
        "--run-id", method,
    ])
    assert result["n_tasks"] == 2 and result["accuracy"] == 1.0
    run_dir = tmp_path / method
    records = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    assert [row["task_id"] for row in records] == benchmarks[0].ids
    snapshots = sorted((run_dir / "state_snapshots").glob("*.json"))
    assert len(snapshots) == 2
    assert "offline-test-placeholder" not in (run_dir / "config.json").read_text()
    if method == "baseline":
        assert benchmarks[0].contexts == ["", ""]
        assert all(row["reflection_text"] == "" for row in records)
    else:
        assert json.loads(snapshots[-1].read_text())["update_count"] == 2


def test_all_published_selections_load_without_benchmark_dependencies():
    selections = [*ROOT.glob("Task/Set1_tasks/*_tasks.json"),
                  *ROOT.glob("Task/scale_up_tasks/*/*.json")]
    assert len(selections) == 30
    for path in selections:
        result = experiment.load_selection(str(path), max_tasks=1)
        assert len(result["task_ids"]) == 1


def test_checksum_corruption_is_rejected(tmp_path):
    data = json.loads((ROOT / "Task/Set1_tasks/appworld_tasks.json").read_text())
    data["task_ids"].reverse()
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        experiment.load_selection(str(path))


def test_check_tasks_loads_real_adapter_data_without_llm(tmp_path, monkeypatch):
    selection_path = ROOT / "Task/Set1_tasks/browsecomp_tasks.json"
    ids = experiment.load_selection(str(selection_path), 2)["task_ids"]
    data_path = tmp_path / "questions.jsonl"
    data_path.write_text("\n".join(json.dumps({"query_id": tid, "query": "Synthetic question?", "answer": "Synthetic answer"}) for tid in reversed(ids)))

    def forbidden(*args, **kwargs):
        raise AssertionError("A loading check must not construct a model client")

    monkeypatch.setattr(experiment, "LLMClient", forbidden)
    result = experiment.main([
        "--task-file", str(selection_path), "--max-tasks", "2", "--check-tasks",
        "--set", f"benchmark_options.data_path={data_path}",
        "--output-dir", str(tmp_path / "runs"),
    ])
    assert result["tasks_loaded"] is True and result["n_tasks"] == 2
    assert not (tmp_path / "runs").exists()


def test_terminalbench_default_agent_matches_provider_model(tmp_path):
    selection = experiment.load_selection(str(ROOT / "Task/Set1_tasks/terminalbench_tasks.json"), 1)
    benchmark = experiment.make_benchmark(selection, {
        "model": "gpt-5.4", "output_dir": str(tmp_path), "run_id": "loading",
    }, checking=True)
    assert benchmark.agent_name == "opencode"
    assert benchmark.model_name == "openai/gpt-5.4"
    assert benchmark.run_output_dir is None


def test_config_dry_run_cannot_accidentally_start_execution(tmp_path, monkeypatch):
    config = tmp_path / "dry.yaml"
    config.write_text(f"task_file: {ROOT / 'Task/Set1_tasks/tau2_tasks.json'}\ndry_run: true\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry_run must not construct a benchmark")

    monkeypatch.setattr(experiment, "make_benchmark", forbidden)
    assert experiment.main(["--config", str(config)])["n_tasks"] == 100
    assert experiment.main(["--task-file", str(ROOT / "Task/Set1_tasks/tau2_tasks.json"),
                            "--set", "dry_run=true"])["n_tasks"] == 100


def test_unscored_tasks_do_not_count_as_failures_or_update_memory(tmp_path):
    from Methods import PlaybookState
    from runtime.runner import RunConfig, Runner

    benchmark = FakeBenchmark(["unscored"])
    benchmark.solve = lambda *args: ("No harness report", {"outcome": "unknown", "score": 0.0})
    state = PlaybookState(FakeLLM("test"))
    config = RunConfig(run_id="unscored", state_type="playbook", benchmark="swebench",
                       model="test", log_dir=str(tmp_path))
    summary = Runner(state, benchmark, FakeLLM("test"), config).run()
    assert summary["n_tasks"] == 1 and summary["n_evaluated_tasks"] == 0
    assert summary["outcome_counts"] == {"unknown": 1}
    assert state.stats()["update_count"] == 0
