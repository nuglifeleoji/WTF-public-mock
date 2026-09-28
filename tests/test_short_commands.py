"""Check concise experiment selectors without datasets or live model calls."""

import json
from pathlib import Path

import pytest

from runtime import experiment
from runtime.llm import _infer_backend


ROOT = Path(__file__).resolve().parents[1]
METHODS = ["ace", "awm", "reasoning_bank", "hermes", "baseline"]


def _expected_presets():
    """Independent selector expectations, including the additional AppWorld sets."""
    presets = []
    for benchmark in ("appworld", "browsecomp", "swebench", "tau2", "terminalbench"):
        stem = "swebench_lite" if benchmark == "swebench" else benchmark
        presets.append((benchmark, "set1", ROOT / f"Task/Set1_tasks/{stem}_tasks.json"))
    scale_sets = {
        "appworld": ["normal_set1_30", "normal_set2_30", "challenge_set1_30",
                     "challenge_set2_30", "test_normal_subset1_30",
                     "test_normal_subset2_30", "test_challenge_subset1_30",
                     "test_challenge_subset2_30"],
        "browsecomp": [f"subset{i}_30" for i in range(1, 6)],
        "swebench": ["django_30", "matplotlib_23", "sklearn_23", "sympy_30"],
        "tau2": [f"{domain}_set{i}_30" for domain in ("airline", "retail", "telecom")
                 for i in (1, 2)],
        "terminalbench": ["set1_30", "set2_30"],
    }
    for benchmark, sets in scale_sets.items():
        for stem in sets:
            selector = "scale-up/" + stem.rsplit("_", 1)[0].replace("_", "-")
            path = ROOT / f"Task/scale_up_tasks/{benchmark}/{benchmark}_{stem}.json"
            presets.append((benchmark, selector, path))
    return presets


PRESETS = _expected_presets()


@pytest.fixture(autouse=True)
def repository_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.mark.parametrize("benchmark,selector,path", PRESETS)
@pytest.mark.parametrize("method", METHODS)
def test_every_short_task_and_method_runs_metadata_only(benchmark, selector, path, method, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A dry run must not create model clients or benchmark adapters")

    monkeypatch.setattr(experiment, "LLMClient", forbidden)
    monkeypatch.setattr(experiment, "EmbeddingClient", forbidden)
    monkeypatch.setattr(experiment, "make_benchmark", forbidden)
    result = experiment.main([benchmark, method, "--tasks", selector, "--dry-run"])
    selection = json.loads(path.read_text())
    assert result["benchmark"] == benchmark
    assert result["method"] == method
    assert result["n_tasks"] == selection["n_tasks"]
    assert result["task_ids"] == selection["task_ids"]


def test_catalog_covers_each_selection_once():
    from runtime.presets import task_presets

    catalog = task_presets()
    assert len(catalog) == 30
    actual = {(p["benchmark"], p["tasks"]): p for p in catalog}
    assert len(actual) == 30
    assert set(actual) == {(benchmark, selector) for benchmark, selector, _ in PRESETS}
    for benchmark, selector, path in PRESETS:
        row = actual[benchmark, selector]
        source = json.loads(path.read_text())
        assert Path(row["task_file"]).resolve() == path
        assert row["n_tasks"] == source["n_tasks"]
        if source.get("selection_group"):
            assert row["selection_group"] == source["selection_group"]


def _captured_config(monkeypatch, argv):
    monkeypatch.setattr(experiment, "run", lambda config: dict(config))
    return experiment.main(argv)


def test_benchmark_alone_chooses_ace_set1_and_benchmark_options(monkeypatch):
    config = _captured_config(monkeypatch, ["tau2"])
    assert config["method"] == "ace"
    assert Path(config["task_file"]).resolve() == ROOT / "Task/Set1_tasks/tau2_tasks.json"
    assert config["benchmark_options"]["data_dir"] == "data/tau2-bench/data"
    assert config["benchmark_options"]["max_steps"] == 30
    assert _infer_backend(config["model"], config.get("provider")) == "openrouter"


def test_rb_alias_and_explicit_run_id(monkeypatch):
    config = _captured_config(monkeypatch, ["swebench", "rb", "--tasks", "scale-up/django",
                                           "--run-id", "rb_django_repeat2"])
    assert config["method"] == "reasoning_bank"
    assert config["run_id"] == "rb_django_repeat2"
    assert Path(config["task_file"]).name == "swebench_django_30.json"


def test_shortcut_allocates_distinct_descriptive_run_ids(monkeypatch):
    first = _captured_config(monkeypatch, ["tau2", "hermes", "--tasks", "scale-up/airline-set1"])
    second = _captured_config(monkeypatch, ["tau2", "hermes", "--tasks", "scale-up/airline-set1"])
    assert first["run_id"] != second["run_id"]
    for config in (first, second):
        name = config["run_id"]
        assert "tau2" in name and "hermes" in name and "airline" in name
        assert "/" not in name


@pytest.mark.parametrize("benchmark,model,provider", [
    ("tau2", "claude-sonnet-4-6", "anthropic"),
    ("terminalbench", "gpt-5.4", "openai"),
    ("browsecomp", "openrouter/deepseek/deepseek-v4-pro", "openrouter"),
])
def test_model_override_infers_provider(benchmark, model, provider, monkeypatch):
    config = _captured_config(monkeypatch, [benchmark, "--model", model])
    assert _infer_backend(config["model"], config.get("provider")) == provider


def test_explicit_provider_overrides_model_inference(monkeypatch):
    config = _captured_config(monkeypatch, ["tau2", "--model", "anthropic/claude-sonnet-4.6",
                                           "--provider", "openrouter"])
    assert config["provider"] == "openrouter"


def test_shortcut_retains_runtime_overrides(monkeypatch):
    config = _captured_config(monkeypatch, ["appworld", "awm", "--max-tasks", "2",
                                           "--set", "benchmark_options.max_react_steps=7",
                                           "--set", "method_options.induction_window=4"])
    assert config["max_tasks"] == 2
    assert config["benchmark_options"]["max_react_steps"] == 7
    assert config["method_options"]["induction_window"] == 4


@pytest.mark.parametrize("argv", [
    ["unknown", "ace", "--dry-run"],
    ["tau2", "unknown", "--dry-run"],
    ["tau2", "--tasks", "scale-up/django", "--dry-run"],
    ["tau2", "--tasks", "scale-up/not-a-set", "--dry-run"],
    ["tau2", "--config", "configs/tau2.yaml", "--dry-run"],
    ["tau2", "--tasks", "set1", "--task-file", "Task/Set1_tasks/tau2_tasks.json", "--dry-run"],
])
def test_invalid_or_conflicting_shortcuts_fail_before_execution(argv, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid command must not execute")

    monkeypatch.setattr(experiment, "run", forbidden)
    with pytest.raises(SystemExit) as exc:
        experiment.main(argv)
    assert exc.value.code != 0


@pytest.mark.parametrize("argv", [["--list"], ["tau2", "--list"]])
def test_list_never_executes_an_experiment(argv, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Listing must not execute")

    monkeypatch.setattr(experiment, "run", forbidden)
    experiment.main(argv)
    output = capsys.readouterr().out
    assert "set1" in output and "scale-up/airline-set1" in output
    if argv[0] == "tau2":
        assert "scale-up/django" not in output
