"""AppWorld adapter contract tests; no optional environment or API is required."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from Task.benchmarks import appworld


def raw_task(task_id="a"):
    return NS(
        id=task_id, instruction="Count my playlists.",
        supervisor=dict(first_name="Test", last_name="User", email="test@example.com", phone_number="0"),
        app_descriptions={"spotify": "Music"},
    )


class AppWorldTests(unittest.TestCase):
    def test_import_and_constructor_do_not_require_appworld(self):
        with patch.dict(sys.modules, {"appworld": None}):
            appworld.AppWorldBenchmark(data_dir="/unused/data")

    def test_selection_preserves_order_and_reports_missing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            (data / "datasets").mkdir(parents=True)
            (data / "datasets/dev.txt").write_text("a\nb\nc\n", encoding="utf-8")
            task_type = Mock(load=Mock(side_effect=raw_task))
            modules = {"appworld.task": NS(Task=task_type)}
            with patch.dict(sys.modules, modules), patch.object(
                appworld.AppWorldBenchmark, "_configure_paths", return_value=NS(data=str(data))
            ):
                bench = appworld.AppWorldBenchmark(task_ids=["c", "a", "c"], max_tasks=1)
                self.assertEqual([t["task_id"] for t in bench.get_tasks()], ["c", "a", "c"])
                self.assertIs(bench.get_tasks(), bench.get_tasks())
                with self.assertRaisesRegex(ValueError, "missing"):
                    appworld.AppWorldBenchmark(task_ids=["a", "missing"]).get_tasks()
                self.assertEqual(appworld.AppWorldBenchmark(task_ids=[]).get_tasks(), [])
                self.assertEqual(len(appworld.AppWorldBenchmark(max_tasks=2).get_tasks()), 2)

    def test_packaged_prompt_renders_without_ground_truth(self):
        with patch.dict(os.environ, {}, clear=True):
            messages, count = appworld._build_react_messages(raw_task(), "Remember pagination.")
        self.assertEqual(count, len(messages))
        self.assertIn("Remember pagination.", messages[0]["content"])
        self.assertTrue(messages[-1]["content"].endswith("Task: Count my playlists.\n\n"))
        self.assertNotIn("{{", appworld._messages_to_text(messages))

    def test_explicit_prompt_override_is_honored_and_missing_path_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.txt"
            prompt.write_text("USER:\nTask: {{ input_str }}", encoding="utf-8")
            with patch.dict(os.environ, {"APPWORLD_REACT_PROMPT_PATH": str(prompt)}):
                self.assertEqual(appworld._build_react_messages(raw_task(), "")[0][0]["content"],
                                 "Task: Count my playlists.\n\n")
            with patch.dict(os.environ, {"APPWORLD_REACT_PROMPT_PATH": str(prompt) + ".missing"}):
                with self.assertRaises(FileNotFoundError):
                    appworld._load_official_prompt_template()

    def test_code_extraction_stops_at_first_block_and_closes_partial(self):
        code, response = appworld._extract_code_and_fix("hi\n```python\nprint(1)\n```\n```python\nbad\n```")
        self.assertEqual(code, "print(1)")
        self.assertNotIn("bad", response)
        self.assertEqual(appworld._extract_code_and_fix("```python\nprint(2)")[1],
                         "```python\nprint(2)\n```")
        self.assertEqual(appworld._extract_code_and_fix("no code"), ("", "no code"))

    def test_history_trimming_keeps_task_and_does_not_mutate_source(self):
        messages = [
            {"role": "user", "content": "Task: x\n"},
            {"role": "assistant", "content": "code\n"},
            {"role": "user", "content": "Output:" + "x" * 500},
            {"role": "assistant", "content": "next\n"},
            {"role": "user", "content": "Output:small\n"},
        ]
        with patch.object(appworld, "_MAX_PROMPT_HISTORY_CHARS", 100):
            trimmed = appworld._trimmed_messages(messages, 1)
            self.assertIn("Task: x", trimmed[0]["content"])
            self.assertNotIn("x" * 500, appworld._messages_to_text(trimmed))
            self.assertIn("x" * 500, appworld._messages_to_text(messages))
            with self.assertRaises(ValueError):
                appworld._trimmed_messages([{"role": "user", "content": "Task: " + "x" * 200}], 1)

    def test_solver_runs_code_observations_and_uses_environment_score(self):
        tracker = NS(success=False, num_tests=2, pass_count=1,
                     to_dict=lambda: {"failures": [{"requirement": "answer mismatch"}]})
        world = Mock()
        world.__enter__ = Mock(return_value=world)
        world.__exit__ = Mock(return_value=False)
        world.execute.side_effect = ["x" * 21000, "finished"]
        world.task_completed.side_effect = [False, True]
        world.evaluate.return_value = (tracker, "extra")
        factory = Mock(return_value=world)
        llm = Mock()
        llm.generate_messages.side_effect = ["```python\nprint(1)\n```", "```python\nprint(2)"]
        with patch.dict(sys.modules, {"appworld": NS(AppWorld=factory)}), patch.object(
            appworld.AppWorldBenchmark, "_configure_paths"
        ):
            bench = appworld.AppWorldBenchmark(max_react_steps=3, experiment_name="test")
            trajectory, result = bench.solve({"_appworld_task": raw_task()}, "memory", llm)
        self.assertEqual(result, {"outcome": "failure", "score": 0.5, "details": "answer mismatch"})
        self.assertEqual([c.args[0] for c in world.execute.call_args_list], ["print(1)", "print(2)"])
        self.assertIn("REST NOT SHOWN FOR BREVITY", trajectory)
        self.assertEqual(llm.generate_messages.call_count, 2)
        factory.assert_called_once_with(task_id="a", experiment_name="test")
        world.__exit__.assert_called_once()

    def test_execution_error_produces_failure(self):
        with patch.dict(sys.modules, {"appworld": NS(AppWorld=Mock(side_effect=RuntimeError("offline failure")))}), \
             patch.object(appworld.AppWorldBenchmark, "_configure_paths"):
            _, result = appworld.AppWorldBenchmark().solve({"_appworld_task": raw_task()}, "", Mock())
        self.assertEqual(result["score"], 0)
        self.assertIn("RuntimeError: offline failure", result["details"])


if __name__ == "__main__":
    unittest.main()
