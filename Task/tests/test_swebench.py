from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from Task.benchmarks.swebench import (
    SWEBenchBenchmark,
    _build_official_prompt,
    _evaluate_with_harness,
    _is_retryable_swebench_modal_error,
    _normalize_hunk_headers,
    _build_system_prompt,
    _parse_harness_report,
    _playbook_has_bullets,
)


class FakeSWELLM:
    model = "o4"

    def __init__(self) -> None:
        self._message_calls = 0

    def generate_messages(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> str:
        if system and "You will be provided with a partial code base" in system:
            return """```diff
--- a/sample.py
+++ b/sample.py
@@ -1,2 +1,2 @@
 def target():
-    return 1
+    return 2
```"""

        self._message_calls += 1
        if self._message_calls == 1:
            return """```bash
python3 - <<'PY'
from pathlib import Path
p = Path("sample.py")
src = p.read_text()
p.write_text(src.replace("return 1", "return 2"))
PY
git diff
```
DONE"""
        return "DONE"


def _make_instance(base_commit: str) -> dict:
    return {
        "instance_id": "pytest-dev__pytest-00001",
        "repo": "pytest-dev/pytest",
        "base_commit": base_commit,
        "problem_statement": "Update the return value in sample.py.",
        "hints_text": "Inspect sample.py and make a minimal change.",
    }


def _init_git_repo(repo_path: Path) -> str:
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "sample.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "sample.py"], cwd=repo_path, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class _TempDirOverride:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self) -> str:
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class SWEBenchBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        # Optional harness/Java dependencies stay outside the offline test suite.
        # Only their API boundaries are faked; adapter code runs unchanged.
        modules = {}
        for name in (
            "swebench", "swebench.harness", "swebench.harness.modal_eval",
            "swebench.inference", "swebench.inference.make_datasets",
            "swebench.inference.make_datasets.bm25_retrieval",
            "swebench.inference.make_datasets.utils", "tiktoken", "datasets",
        ):
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(modules[parent], child, module)
        modules["swebench.harness.modal_eval"].validate_modal_credentials = mock.Mock()
        bm25 = modules["swebench.inference.make_datasets.bm25_retrieval"]
        bm25.make_index = mock.Mock()
        bm25.search = mock.Mock()
        bm25.DOCUMENT_ENCODING_FUNCTIONS = {"file_name_and_contents": mock.Mock()}
        modules["swebench.inference.make_datasets.utils"].extract_diff = (
            lambda response: re.search(r"```diff\n(.*?)```", response, re.S).group(1)
        )
        modules["tiktoken"].get_encoding = lambda name: SimpleNamespace(encode=lambda text, **kwargs: list(text))
        modules["datasets"].load_dataset = mock.Mock()
        self.optional_modules = modules
        patcher = mock.patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_task_ids_keep_requested_order_and_override_limit(self) -> None:
        instances = [dict(_make_instance("abc"), instance_id=task_id) for task_id in ["a", "b", "c"]]
        self.optional_modules["datasets"].load_dataset.return_value = instances
        benchmark = SWEBenchBenchmark(task_ids=["c", "a"], max_tasks=1)
        self.assertEqual([t["task_id"] for t in benchmark.get_tasks()], ["c", "a"])
        self.assertIs(benchmark.get_tasks(), benchmark.get_tasks())

    def test_missing_task_id_is_an_error(self) -> None:
        self.optional_modules["datasets"].load_dataset.return_value = [_make_instance("abc")]
        with self.assertRaisesRegex(ValueError, "missing-task"):
            SWEBenchBenchmark(task_ids=["missing-task"]).get_tasks()

    def test_dataset_revision_is_forwarded_only_when_configured(self) -> None:
        loader = self.optional_modules["datasets"].load_dataset
        loader.return_value = [_make_instance("abc")]
        SWEBenchBenchmark(dataset_revision="dataset-commit").get_tasks()
        loader.assert_called_once_with(
            "princeton-nlp/SWE-bench_Lite", split="test", revision="dataset-commit"
        )
        loader.reset_mock()
        SWEBenchBenchmark().get_tasks()
        loader.assert_called_once_with("princeton-nlp/SWE-bench_Lite", split="test")

    def test_rendered_context_for_all_four_methods_is_included(self) -> None:
        from Methods import AWMState, HermesState, PlaybookState, ReasoningBankState
        from Methods.awm import Workflow
        from Methods.hermes import HermesSkill
        from Methods.reasoning_bank import MemoryItem, ReasoningBankEntry

        llm = mock.Mock()
        ace = PlaybookState(llm)
        self.assertFalse(_playbook_has_bullets(ace.retrieve("patch")))
        ace.load_serialized(json.dumps({"type": "playbook", "playbook": "[code-00001] Inspect the diff."}))
        awm = AWMState(llm)
        awm._workflows["patch"] = Workflow("patch", "Inspect the diff.", ["Read the source."], "wf-00001")
        embedder = SimpleNamespace(top_k_indices=lambda query, candidates, k: [0])
        rb = ReasoningBankState(llm, embedder)
        rb._entries = [ReasoningBankEntry("patch", "success", [MemoryItem("rb-00001", "Patch", "Inspect the diff.", "Read the source.")])]
        hermes = HermesState(llm)
        hermes._skills["patch"] = HermesSkill("hm-00001", "patch", "Inspect the diff.", "Read the source.")
        for state in (ace, awm, rb, hermes):
            with self.subTest(method=type(state).__name__):
                context = state.retrieve("patch")
                self.assertTrue(_playbook_has_bullets(context))
                self.assertIn(context, _build_system_prompt(_make_instance("abc"), context))

    def test_normalize_hunk_headers_preserves_bare_multifile_headers(self) -> None:
        patch = "\n".join(
            [
                "--- a/astropy/io/ascii/docs.py",
                "+++ b/astropy/io/ascii/docs.py",
                "@@ -184,7 +186,8 @@ WRITE_KWARG_TYPES = {",
                '     "names": "list-like",',
                '     "include_names": "list-like",',
                '     "exclude_names": "list-like",',
                '+    "header_rows": "list-like",',
                '     "fast_writer": (bool, str),',
                '     "overwrite": (bool),',
                " }",
                "--- a/astropy/io/ascii/ui.py",
                "+++ b/astropy/io/ascii/ui.py",
                "@@ -858,4 +858,5 @@ extra_writer_pars = (",
                '     "include_names",',
                '     "exclude_names",',
                '     "strip_whitespace",',
                '+    "header_rows",',
                " )",
            ]
        )

        normalized = _normalize_hunk_headers(patch)

        self.assertIn("@@ -184,6 +186,7 @@ WRITE_KWARG_TYPES = {", normalized)
        self.assertIn(
            "--- a/astropy/io/ascii/ui.py\n+++ b/astropy/io/ascii/ui.py",
            normalized,
        )


    def test_evaluate_modal_uses_official_report_locations(self) -> None:
        instance = _make_instance("abc123")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_ctx = _TempDirOverride(tmpdir)
            report_path = (
                Path(tmpdir)
                / "logs"
                / "run_evaluation"
                / "unit_pytest_dev__pytest_00001"
                / "unit"
                / instance["instance_id"]
                / "report.json"
            )

            def fake_run(*args, **kwargs):
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps({instance["instance_id"]: {"resolved": True}}),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with mock.patch(
                "swebench.harness.modal_eval.validate_modal_credentials",
                return_value=None,
            ), mock.patch(
                "Task.benchmarks.swebench.tempfile.TemporaryDirectory",
                return_value=temp_ctx,
            ), mock.patch(
                "Task.benchmarks.swebench.subprocess.run",
                side_effect=fake_run,
            ):
                result = _evaluate_with_harness(
                    instance=instance,
                    patch="diff --git a/sample.py b/sample.py",
                    run_id="unit",
                    dataset_name="princeton-nlp/SWE-bench_Lite",
                    split="test",
                    model_name="unit",
                    use_modal=True,
                )

        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["score"], 1.0)
        self.assertIn("modal", result["details"])

    def test_harness_error_ids_are_scored_failures(self) -> None:
        instance = _make_instance("abc123")

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "unit.unit_pytest_dev__pytest_00001.json"
            report_path.write_text(
                json.dumps(
                    {
                        "resolved_ids": [],
                        "unresolved_ids": [],
                        "error_ids": [instance["instance_id"]],
                        "schema_version": 2,
                    }
                ),
                encoding="utf-8",
            )

            resolved, status = _parse_harness_report(
                report_path, instance["instance_id"]
            )

        self.assertFalse(resolved)
        self.assertEqual(status, "error")

    def test_modal_infra_error_report_is_retried(self) -> None:
        instance = _make_instance("abc123")
        safe_iid = instance["instance_id"].replace("/", "_").replace("-", "_")
        harness_run_id = f"unit_{safe_iid}"
        calls = 0

        def fake_run(*args, **kwargs):
            nonlocal calls
            calls += 1
            eval_path = Path(kwargs["cwd"])
            report_path = eval_path / f"unit.{harness_run_id}.json"
            if calls == 1:
                report_path.write_text(
                    json.dumps(
                        {
                            "resolved_ids": [],
                            "unresolved_ids": [],
                            "error_ids": [instance["instance_id"]],
                            "schema_version": 2,
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="Error creating sandbox: internal server error",
                    stderr="modal.exception.FilesystemExecutionError",
                )

            report_path.write_text(
                json.dumps(
                    {
                        "resolved_ids": [instance["instance_id"]],
                        "unresolved_ids": [],
                        "error_ids": [],
                        "schema_version": 2,
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        self.assertTrue(
            _is_retryable_swebench_modal_error(
                "Error creating sandbox: internal server error"
            )
        )
        self.assertFalse(
            _is_retryable_swebench_modal_error(
                "Could not deserialize remote exception\n"
                ">>>>> Patch Apply Failed:\n"
                "error: patch failed: astropy/io/ascii/docs.py:184"
            )
        )
        self.assertFalse(
            _is_retryable_swebench_modal_error(
                "Could not deserialize remote exception\n"
                "failed to run builder command \"/bin/bash /root/setup_repo.sh\": "
                "ERROR: Project file:///testbed uses a build backend that is "
                "missing the 'build_editable' hook"
            )
        )

        with mock.patch(
            "swebench.harness.modal_eval.validate_modal_credentials",
            return_value=None,
        ), mock.patch(
            "Task.benchmarks.swebench.subprocess.run",
            side_effect=fake_run,
        ), mock.patch("Task.benchmarks.swebench.time.sleep", return_value=None):
            result = _evaluate_with_harness(
                instance=instance,
                patch="diff --git a/sample.py b/sample.py",
                run_id="unit",
                dataset_name="princeton-nlp/SWE-bench_Lite",
                split="test",
                model_name="unit",
                use_modal=True,
            )

        self.assertEqual(calls, 2)
        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["score"], 1.0)

    def test_build_official_prompt_uses_dict_readmes_and_selected_files(self) -> None:
        instance = _make_instance("abc123")
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / "sample.py").write_text("def target():\n    return 1\n", encoding="utf-8")

            with mock.patch(
                "swebench.inference.make_datasets.bm25_retrieval.make_index",
                return_value=Path(tmpdir) / "index",
            ), mock.patch(
                "swebench.inference.make_datasets.bm25_retrieval.search",
                return_value={"hits": [{"docid": "sample.py", "score": 1.0}]},
            ):
                prompt, files = _build_official_prompt(instance, str(repo_path), tmpdir)

        self.assertIn("<issue>", prompt)
        self.assertIn("[start of sample.py]", prompt)
        self.assertEqual(files, ["sample.py"])

    def test_solve_only_generates_patch_and_saves_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            base_commit = _init_git_repo(repo_path)
            instance = _make_instance(base_commit)
            predictions_path = Path(tmpdir) / "predictions.json"
            llm = FakeSWELLM()

            benchmark = SWEBenchBenchmark(
                solve_only=True,
                predictions_path=str(predictions_path),
                model_name="swebench_smoke",
            )

            task = {
                "task_id": instance["instance_id"],
                "description": instance["problem_statement"],
                "_swebench_instance": instance,
            }

            official_prompt = (
                "You will be provided with a partial code base and an issue statement explaining a problem to resolve.\n"
                "<issue>\nUpdate the return value in sample.py.\n</issue>\n"
                "<code>\n[start of sample.py]\ndef target():\n    return 1\n[end of sample.py]\n</code>\n"
                "Respond below:"
            )
            with mock.patch("Task.benchmarks.swebench._clone_repo", return_value=str(repo_path)), mock.patch(
                "Task.benchmarks.swebench._build_official_prompt",
                return_value=(official_prompt, ["sample.py"]),
            ):
                trajectory, result = benchmark.solve(task, "Prefer minimal patches.", llm)

            saved = json.loads(predictions_path.read_text(encoding="utf-8"))
            self.assertEqual(result["outcome"], "pending")
            self.assertIn("Patch generated", result["details"])
            self.assertIn("sample.py", trajectory)
            self.assertEqual(saved[0]["instance_id"], instance["instance_id"])
            self.assertIn("return 2", saved[0]["model_patch"])

    def test_react_backend_executes_edits_in_a_local_fixture_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            instance = _make_instance(_init_git_repo(repo_path))
            predictions_path = Path(tmpdir) / "predictions.json"
            benchmark = SWEBenchBenchmark(
                solver_backend="react", solve_only=True,
                predictions_path=str(predictions_path),
            )
            task = {
                "task_id": instance["instance_id"],
                "description": instance["problem_statement"],
                "_swebench_instance": instance,
            }
            with mock.patch("Task.benchmarks.swebench._clone_repo", return_value=str(repo_path)):
                trajectory, result = benchmark.solve(task, "", FakeSWELLM())
            self.assertEqual(result["outcome"], "pending")
            self.assertIn("return 2", (repo_path / "sample.py").read_text())
            self.assertIn("ASSISTANT:", trajectory)
            saved = json.loads(predictions_path.read_text())
            self.assertIn("+    return 2", saved[0]["model_patch"])



if __name__ == "__main__":
    unittest.main()
