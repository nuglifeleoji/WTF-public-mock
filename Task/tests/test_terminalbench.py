from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from Task.benchmarks.terminalbench import TerminalBenchBenchmark


class FakeTerminalBenchLLM:
    model = "anthropic/claude-haiku-4-5"


def _write_terminalbench_task(repo_root: Path, task_name: str = "cancel-async-tasks") -> Path:
    task_dir = repo_root / task_name
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text(
        "Create `/app/run.py` with an async helper that runs tasks concurrently and preserves cleanup.",
        encoding="utf-8",
    )
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                'version = "1.0"',
                "",
                "[metadata]",
                'difficulty = "hard"',
                'category = "software-engineering"',
                'tags = ["async", "python"]',
                "",
                "[verifier]",
                "timeout_sec = 900.0",
            ]
        ),
        encoding="utf-8",
    )
    return task_dir


class TerminalBenchBenchmarkTests(unittest.TestCase):
    def test_task_ids_keep_requested_order_and_override_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            for task_id in ("a", "b", "c"):
                _write_terminalbench_task(repo_root, task_id)
            benchmark = TerminalBenchBenchmark(
                repo_path=tmpdir, repo_url=None, task_ids=["c", "a"],
                max_tasks=1, run_output_dir=None,
            )
            self.assertEqual([t["task_id"] for t in benchmark.get_tasks()], ["c", "a"])

    def test_missing_task_id_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_terminalbench_task(Path(tmpdir), "a")
            benchmark = TerminalBenchBenchmark(
                repo_path=tmpdir, repo_url=None, task_ids=["a", "missing-task"],
                run_output_dir=None,
            )
            with self.assertRaisesRegex(ValueError, "missing-task"):
                benchmark.get_tasks()

    def test_infrastructure_failure_retries_then_uses_verified_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "tasks"
            _write_terminalbench_task(repo_root, "a")
            benchmark = TerminalBenchBenchmark(
                repo_path=str(repo_root), repo_url=None, agent_name="claude-code",
                run_output_dir=str(Path(tmpdir) / "runs"), harbor_infra_retries=1,
            )
            attempts = []

            def fake_harbor_run(args, **kwargs):
                trial_name = args[args.index("--trial-name") + 1]
                attempts.append(trial_name)
                trial_dir = Path(args[args.index("--trials-dir") + 1]) / trial_name
                trial_dir.mkdir(parents=True)
                payload = {
                    "verifier_result": {"rewards": {"reward": 1.0}},
                    "exception_info": {"exception_type": "FilesystemExecutionError"} if len(attempts) == 1 else None,
                }
                (trial_dir / "result.json").write_text(json.dumps(payload))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch("Task.benchmarks.terminalbench.subprocess.run", side_effect=fake_harbor_run), mock.patch(
                "Task.benchmarks.terminalbench.time.sleep"
            ) as sleep:
                _, result = benchmark.solve(benchmark.get_tasks()[0], "", FakeTerminalBenchLLM())
            self.assertEqual(len(set(attempts)), 2)
            sleep.assert_called_once_with(15)
            self.assertEqual(result["outcome"], "success")
            self.assertEqual(result["score"], 1.0)


    def test_get_tasks_reads_local_terminalbench_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "terminal-bench-2"
            _write_terminalbench_task(repo_root, "cancel-async-tasks")
            _write_terminalbench_task(repo_root, "regex-log")
            benchmark = TerminalBenchBenchmark(
                repo_path=str(repo_root),
                repo_url=None,
                max_tasks=1,
                run_output_dir=str(Path(tmpdir) / "runs"),
                run_id="unit",
            )

            tasks = benchmark.get_tasks()

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["task_id"], "cancel-async-tasks")
            self.assertIn("async helper", tasks[0]["description"])
            self.assertEqual(tasks[0]["difficulty"], "hard")
            self.assertEqual(tasks[0]["category"], "software-engineering")
            self.assertEqual(tasks[0]["tags"], ["async", "python"])

    def test_solve_runs_harbor_trial_and_injects_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "terminal-bench-2"
            _write_terminalbench_task(repo_root)
            benchmark = TerminalBenchBenchmark(
                repo_path=str(repo_root),
                repo_url=None,
                environment="modal",
                agent_name="claude-code",
                run_output_dir=str(Path(tmpdir) / "runs"),
                run_id="unit",
            )
            task = benchmark.get_tasks()[0]
            llm = FakeTerminalBenchLLM()
            observed_instruction: dict[str, str] = {}

            def fake_harbor_run(args, capture_output, text, timeout, **kwargs):
                task_path = Path(args[args.index("--path") + 1])
                observed_instruction["text"] = (task_path / "instruction.md").read_text(encoding="utf-8")
                trial_name = args[args.index("--trial-name") + 1]
                trials_dir = Path(args[args.index("--trials-dir") + 1])
                trial_dir = trials_dir / f"{trial_name}__claude-code__attempt-1"
                (trial_dir / "logs" / "agent").mkdir(parents=True, exist_ok=True)
                (trial_dir / "logs" / "verifier").mkdir(parents=True, exist_ok=True)
                (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
                (trial_dir / "verifier").mkdir(parents=True, exist_ok=True)
                (trial_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "verifier_result": {
                                "rewards": {"reward": 1.0},
                                "stdout": "All checks passed.",
                                "stderr": "",
                            },
                            "exception_info": None,
                        }
                    ),
                    encoding="utf-8",
                )
                (trial_dir / "agent" / "trajectory.json").write_text(
                    json.dumps(
                        {
                            "episodes": [
                                {
                                    "steps": [
                                        {"source": "user", "content": "Solve the task."},
                                        {"source": "assistant", "content": "Edited /app/run.py."},
                                        {"source": "verifier", "content": "Reward: 1.0"},
                                    ]
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                (trial_dir / "verifier" / "test-stdout.txt").write_text(
                    "All checks passed.",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with mock.patch("Task.benchmarks.terminalbench.subprocess.run", side_effect=fake_harbor_run):
                trajectory, result = benchmark.solve(
                    task,
                    context="Inspect the verifier reward before concluding the task.",
                    llm=llm,
                )

            self.assertIn("Additional guidance from prior terminal-bench attempts", observed_instruction["text"])
            self.assertIn("Inspect the verifier reward", observed_instruction["text"])
            self.assertEqual(result["outcome"], "success")
            self.assertEqual(result["score"], 1.0)
            self.assertIn("terminalbench harness (modal)", result["details"])
            self.assertIn("verifier_stdout=All checks passed.", result["details"])
            self.assertIn("ASSISTANT:\nEdited /app/run.py.", trajectory)
            self.assertTrue((Path(tmpdir) / "runs" / "unit").exists())

    def test_solve_sanitizes_nul_in_context_and_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "terminal-bench-2"
            _write_terminalbench_task(repo_root, "break-filter-js-from-html")
            benchmark = TerminalBenchBenchmark(
                repo_path=str(repo_root),
                repo_url=None,
                environment="modal",
                agent_name="opencode",
                model_name="openrouter/anthropic/claude-sonnet-4.6",
                run_output_dir=str(Path(tmpdir) / "runs"),
                run_id="nul-sanitization",
            )
            task = benchmark.get_tasks()[0]
            observed_instruction: dict[str, str] = {}

            def fake_harbor_run(args, capture_output, text, timeout, **kwargs):
                task_path = Path(args[args.index("--path") + 1])
                instruction = (task_path / "instruction.md").read_text(encoding="utf-8")
                observed_instruction["text"] = instruction
                self.assertNotIn("\x00", instruction)
                self.assertIn(r"\x00onerror", instruction)

                trial_name = args[args.index("--trial-name") + 1]
                trials_dir = Path(args[args.index("--trials-dir") + 1])
                trial_dir = trials_dir / f"{trial_name}__opencode__attempt-1"
                (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
                (trial_dir / "verifier").mkdir(parents=True, exist_ok=True)
                (trial_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "verifier_result": {"rewards": {"reward": 1.0}},
                            "exception_info": None,
                        }
                    ),
                    encoding="utf-8",
                )
                (trial_dir / "agent" / "trajectory.json").write_text(
                    json.dumps(
                        {
                            "episodes": [
                                {
                                    "steps": [
                                        {
                                            "source": "assistant",
                                            "content": "Candidate: <img src=x \x00onerror=alert(1)>",
                                        }
                                    ]
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with mock.patch("Task.benchmarks.terminalbench.subprocess.run", side_effect=fake_harbor_run):
                trajectory, result = benchmark.solve(
                    task,
                    context="Use candidate <img src=x \x00onerror=alert(1)>",
                    llm=FakeTerminalBenchLLM(),
                )

            self.assertIn(r"\x00onerror", observed_instruction["text"])
            self.assertNotIn("\x00", trajectory)
            self.assertIn(r"\x00onerror", trajectory)
            self.assertEqual(result["outcome"], "success")

    def test_solve_wraps_remote_codex_harbor_call_without_changing_result_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "terminal-bench-2"
            _write_terminalbench_task(repo_root, "regex-log")
            harbor_bin_dir = Path(tmpdir) / "harbor-bin"
            harbor_bin_dir.mkdir(parents=True, exist_ok=True)
            fake_harbor = harbor_bin_dir / "harbor"
            fake_python = harbor_bin_dir / "python"
            fake_harbor.write_text("", encoding="utf-8")
            fake_python.write_text("", encoding="utf-8")

            benchmark = TerminalBenchBenchmark(
                repo_path=str(repo_root),
                repo_url=None,
                environment="modal",
                agent_name="codex",
                harbor_bin=str(fake_harbor),
                run_output_dir=str(Path(tmpdir) / "runs"),
                run_id="unit",
            )
            task = benchmark.get_tasks()[0]
            llm = FakeTerminalBenchLLM()

            def fake_harbor_run(args, capture_output, text, timeout, **kwargs):
                self.assertEqual(args[0], str(fake_python))
                self.assertTrue(args[1].endswith("helpers/harbor_terminalbench_wrapper.py"))
                self.assertEqual(kwargs["env"]["WTF_FILTER_CODEX_AGENT_UPLOADS"], "1")

                trial_name = args[args.index("--trial-name") + 1]
                trials_dir = Path(args[args.index("--trials-dir") + 1])
                trial_dir = trials_dir / f"{trial_name}__codex__attempt-1"
                (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
                (trial_dir / "verifier").mkdir(parents=True, exist_ok=True)
                (trial_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "verifier_result": {"rewards": {"reward": 1.0}},
                            "exception_info": None,
                        }
                    ),
                    encoding="utf-8",
                )
                (trial_dir / "agent" / "trajectory.json").write_text(
                    json.dumps(
                        {
                            "episodes": [
                                {
                                    "steps": [
                                        {"source": "assistant", "content": "Solved regex-log."},
                                    ]
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with mock.patch("Task.benchmarks.terminalbench.subprocess.run", side_effect=fake_harbor_run):
                trajectory, result = benchmark.solve(task, context="", llm=llm)

            self.assertEqual(result["outcome"], "success")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["agent_name"], "codex")
            self.assertEqual(result["environment"], "modal")
            self.assertIn("ASSISTANT:\nSolved regex-log.", trajectory)



if __name__ == "__main__":
    unittest.main()
