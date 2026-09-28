"""tau2 task selection and simulation handoff without network access."""

import os
import sys
import unittest
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from Task.benchmarks.tau2 import Tau2Benchmark, _CONTEXT, _resolve_task_set_name


def raw_task(task_id):
    return NS(id=task_id, user_scenario=NS(instructions="Synthetic request"))


class Registry:
    def __init__(self):
        self.agents = {}

    def get_task_sets(self):
        return ["airline", "retail", "telecom", "telecom-workflow"]

    def get_agents(self):
        return list(self.agents)

    def register_agent(self, constructor, name):
        self.agents[name] = constructor

    def get_agent_constructor(self, name):
        return self.agents[name]


class Tau2Tests(unittest.TestCase):
    def test_experiment_overlay_contains_exact_manifested_assets(self):
        root = Path(__file__).resolve().parents[1] / "assets" / "tau2_overlay"
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["upstream_commit"], "0ed2fd8d830a20657d89ae9c2efcc94838aa7129")
        expected = {
            "tau2/domains/airline/tasks.json", "tau2/domains/retail/tasks.json",
            "tau2/user_simulator/simulation_guidelines.md",
            "tau2/user_simulator/simulation_guidelines_tools.md",
        }
        self.assertEqual({item["path"] for item in manifest["files"]}, expected)
        for item in manifest["files"]:
            with self.subTest(path=item["path"]):
                contents = (root / item["path"]).read_bytes()
                self.assertEqual(len(contents), item["bytes"])
                self.assertEqual(hashlib.sha256(contents).hexdigest(), item["sha256"])
                self.assertNotEqual(item["sha256"], item["upstream_sha256"])
        self.assertEqual({str(p.relative_to(root)) for p in (root / "tau2").rglob("*") if p.is_file()}, expected)

    def test_selection_preserves_mixed_domain_order_and_duplicates(self):
        fetch = Mock(side_effect=lambda domain, task_ids=None, **kw: [raw_task(i) for i in reversed(task_ids)])
        with patch.dict(sys.modules, {"tau2.run": NS(get_tasks=fetch), "tau2.registry": NS(registry=Registry())}):
            bench = Tau2Benchmark(task_ids=["airline/2", "retail/0", "airline/1", "airline/2"])
            tasks = bench.get_tasks()
            self.assertEqual([t["task_id"] for t in tasks], ["airline/2", "retail/0", "airline/1", "airline/2"])
            self.assertEqual(fetch.call_args_list[0].kwargs["task_ids"], ["2", "1"])
            self.assertIs(bench.get_tasks(), tasks)
            self.assertEqual(Tau2Benchmark(task_ids=[]).get_tasks(), [])

    def test_selection_rejects_missing_id_and_ambiguous_domain(self):
        with patch.dict(sys.modules, {"tau2.run": NS(get_tasks=lambda *a, **k: []),
                                      "tau2.registry": NS(registry=Registry())}):
            with self.assertRaisesRegex(ValueError, "airline/missing"):
                Tau2Benchmark(task_ids=["airline/missing"]).get_tasks()
        with self.assertRaisesRegex(ValueError, "no domain prefix"):
            Tau2Benchmark(domain={"retail": 2}, task_ids=["0"])
        with self.assertRaises(ValueError):
            Tau2Benchmark(task_ids=["retail/"])

    def test_split_resolution_supports_workflows(self):
        with patch.dict(sys.modules, {"tau2.registry": NS(registry=Registry())}):
            self.assertEqual(_resolve_task_set_name("airline", "base"), "airline")
            self.assertEqual(_resolve_task_set_name("telecom", "workflow"), "telecom-workflow")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                _resolve_task_set_name("airline", "unknown")

    def test_data_path_is_set_before_import_and_cannot_silently_change(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"tau2.utils.utils": None}):
            bench = Tau2Benchmark(data_dir="/tmp/tau2/data")
            self.assertNotIn("TAU2_DATA_DIR", os.environ)
            bench._configure_paths()
            self.assertEqual(os.environ["TAU2_DATA_DIR"], "/tmp/tau2/data")
        with patch.dict(sys.modules, {"tau2.utils.utils": NS(DATA_DIR=Path("/tmp/other/data"))}):
            with self.assertRaisesRegex(ValueError, "different data_dir"):
                Tau2Benchmark(data_dir="/tmp/tau2/data")._configure_paths()

    def test_simulation_receives_context_evaluation_and_model_options(self):
        registry = Registry()
        captured = []
        class Agent:
            @property
            def system_prompt(self):
                return "Base policy"
        def run_task(**kwargs):
            captured.append(kwargs)
            prompt = registry.get_agent_constructor(kwargs["agent"])().system_prompt
            self.assertEqual(prompt, "Base policy\n<retrieved_knowledge>\nuse memory\n</retrieved_knowledge>")
            return NS(
                messages=[NS(role="assistant", content="done", tool_calls=None)],
                reward_info=NS(reward=1, reward_breakdown={}, db_check=None,
                               communicate_checks=[], env_assertions=[]),
                agent_cost=.03, user_cost=.02,
            )
        modules = {"tau2.agent.llm_agent": NS(LLMAgent=Agent),
                   "tau2.evaluator.evaluator": NS(EvaluationType=NS(ALL_WITH_NL_ASSERTIONS="all_with_nl")),
                   "tau2.registry": NS(registry=registry),
                   "tau2.run": NS(get_tasks=Mock(), run_task=run_task)}
        llm = NS(model="gpt-4o", extra_body={"reasoning": {"effort": "low"}, "verbosity": "low"},
                 base_url="https://example.com/v1")
        with patch.dict(sys.modules, modules):
            trajectory, result = Tau2Benchmark(max_steps=7).solve(
                {"task_id": "airline/3", "_domain": "airline", "_tau2_task": raw_task("3")}, "use memory", llm)
        self.assertEqual(trajectory, "ASSISTANT:\ndone")
        self.assertEqual(result["outcome"], "success")
        self.assertAlmostEqual(result["cost_usd"], .05)
        self.assertEqual(captured[0]["evaluation_type"], "all_with_nl")
        self.assertEqual(captured[0]["llm_agent"], "openai/gpt-4o")
        self.assertEqual(captured[0]["max_steps"], 7)
        self.assertEqual(captured[0]["llm_args_agent"], {
            "parallel_tool_calls": False, "reasoning_effort": "low", "verbosity": "low",
            "api_base": "https://example.com/v1",
        })
        self.assertEqual(_CONTEXT.get(), "")

    def test_serialized_task_recovers_its_domain_and_clears_context_after_failure(self):
        registry = Registry()
        fetch = Mock(return_value=[raw_task("3")])
        class Agent:
            @property
            def system_prompt(self):
                return "policy"
        modules = {"tau2.agent.llm_agent": NS(LLMAgent=Agent),
                   "tau2.evaluator.evaluator": NS(EvaluationType=NS(ALL_WITH_NL_ASSERTIONS="all")),
                   "tau2.registry": NS(registry=registry),
                   "tau2.run": NS(get_tasks=fetch, run_task=Mock(side_effect=RuntimeError("simulation failed")))}
        with patch.dict(sys.modules, modules):
            _, result = Tau2Benchmark().solve({"task_id": "airline/3"}, "private context", NS(model="gpt-4o"))
        fetch.assert_called_once_with("airline", task_ids=["3"])
        self.assertEqual(result["outcome"], "failure")
        self.assertIn("simulation failed", result["details"])
        self.assertEqual(_CONTEXT.get(), "")


if __name__ == "__main__":
    unittest.main()
