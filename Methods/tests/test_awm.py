"""Offline regression tests for workflow induction, retrieval, and snapshots."""

import json
import unittest

from Methods.awm import AWMState


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else '{"workflows": []}'


def workflow(name, docstring="Reusable procedure", steps=None):
    return {
        "name": name,
        "docstring": docstring,
        "steps": steps if steps is not None else ["Open {resource}", "Read {field}"],
    }


def response(*workflows):
    return json.dumps({"workflows": workflows})


def success(task_id):
    return {
        "outcome": "success",
        "task_description": f"Task {task_id}",
        "trajectory": f"Trace {task_id}",
        "metadata": {"task_id": task_id},
    }


class AWMTests(unittest.TestCase):
    def test_success_only_induction_uses_window_and_recent_history(self):
        llm = FakeLLM()
        state = AWMState(llm, induction_window=2, induction_history_size=2)
        self.assertEqual(state.retrieve("any task"), "")
        state.update({"outcome": "failure", "trajectory": "failed-trajectory"})
        state.update({"outcome": "success", "trajectory": " "})
        state.update(success("first"))
        self.assertEqual(llm.prompts, [])
        state.update(success("second"))
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("Trace first", llm.prompts[0])
        self.assertIn("Trace second", llm.prompts[0])
        self.assertNotIn("failed-trajectory", llm.prompts[0])
        state.update(success("third"))
        state.update(success("fourth"))
        self.assertEqual(len(llm.prompts), 2)
        self.assertNotIn("Trace first", llm.prompts[1])
        self.assertNotIn("Trace second", llm.prompts[1])
        self.assertIn("Trace third", llm.prompts[1])
        self.assertIn("Trace fourth", llm.prompts[1])
        self.assertEqual(state.stats()["successful_updates"], 5)
        self.assertEqual(state.stats()["update_count"], 6)

    def test_revision_preserves_id_and_merges_sources(self):
        llm = FakeLLM(
            response(workflow("Lookup Orders", "First version")),
            response(workflow("lookup_orders", "Updated version", ["Select {id}", "Read status"])),
        )
        state = AWMState(llm, induction_window=1)
        state.update(success("first"))
        state.update(success("second"))
        self.assertIn("[wf-00001] lookup_orders", llm.prompts[1])
        rendered = state.retrieve("orders")
        self.assertIn("[wf-00001] lookup_orders", rendered)
        self.assertIn("Updated version", rendered)
        self.assertIn("1. Select {id}", rendered)
        saved = json.loads(state.serialize())["workflows"]["lookup_orders"]
        self.assertEqual(saved["revision"], 2)
        self.assertEqual(saved["source_task_ids"], ["first", "second"])
        self.assertEqual(saved["usage_count"], 1)
        self.assertEqual(state.stats()["total_added"], 1)
        self.assertEqual(state.stats()["total_revised"], 1)

    def test_semantic_retrieval_uses_name_and_description(self):
        calls = []

        class Embedder:
            def top_k_indices(self, query, candidates, k):
                calls.append((query, candidates, k))
                return [1, -1, 99]

        state = AWMState(
            FakeLLM(response(workflow("orders", "Order history"), workflow("stock", "Inventory lookup"))),
            embedder=Embedder(),
            induction_window=1,
        )
        state.update(success("first"))
        rendered = state.retrieve("quantity", k=1)
        self.assertEqual(calls, [("quantity", ["orders\nOrder history", "stock\nInventory lookup"], 1)])
        self.assertIn("[wf-00002] stock", rendered)
        self.assertNotIn("[wf-00001]", rendered)

    def test_keyword_retrieval_handles_missing_or_failed_embeddings(self):
        class BrokenEmbedder:
            def top_k_indices(self, *args):
                raise RuntimeError("unavailable")

        for embedder in (None, BrokenEmbedder()):
            with self.subTest(embedder=type(embedder).__name__):
                state = AWMState(
                    FakeLLM(response(workflow("orders", "Order history"), workflow("stock", "Inventory lookup"))),
                    embedder=embedder,
                    induction_window=1,
                )
                state.update(success("first"))
                self.assertIn("[wf-00002] stock", state.retrieve("inventory", k=1))
                self.assertIn("[wf-00001] orders", state.retrieve("", k=1))

    def test_snapshot_roundtrip_resets_pending_induction_history(self):
        state = AWMState(FakeLLM(response(workflow("orders"))), induction_window=2)
        for task_id in ("first", "second", "pending"):
            state.update(success(task_id))
        state.retrieve("orders")
        payload = state.serialize()
        llm = FakeLLM()
        restored = AWMState(llm, induction_window=2)
        restored.load_serialized(payload)
        self.assertEqual(restored.serialize(), payload)
        self.assertEqual(restored.stats(), state.stats())
        restored.update(success("new-first"))
        self.assertEqual(llm.prompts, [])
        restored.update(success("new-second"))
        self.assertEqual(len(llm.prompts), 1)
        self.assertNotIn("Trace pending", llm.prompts[0])
        with self.assertRaises(ValueError):
            restored.load_serialized('{"type": "another-method"}')

    def test_library_limits_evict_oldest_workflows(self):
        workflows = response(workflow("first"), workflow("second"), workflow("third"))
        state = AWMState(FakeLLM(workflows), induction_window=1, max_workflows=2)
        state.update(success("task"))
        self.assertEqual(state.stats()["workflow_names"], ["second", "third"])
        limited = AWMState(FakeLLM(workflows), induction_window=1, max_tokens=0)
        limited.update(success("task"))
        self.assertEqual(limited.retrieve("any"), "")
        self.assertEqual(limited.stats()["approx_tokens"], 0)

    def test_invalid_induction_and_short_workflows_are_rejected(self):
        state = AWMState(
            FakeLLM("invalid JSON", response(workflow("too_short", steps=["One step"]), workflow("valid"))),
            induction_window=1,
        )
        state.update(success("first"))
        self.assertEqual(state.stats()["induction_failures"], 1)
        state.update(success("second"))
        self.assertEqual(state.stats()["workflow_names"], ["valid"])


if __name__ == "__main__":
    unittest.main()
