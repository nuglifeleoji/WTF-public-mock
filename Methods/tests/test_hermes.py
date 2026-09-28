"""Offline regression tests for the Hermes skill lifecycle."""

import json
import unittest

from Methods.hermes import HermesState


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        return response if isinstance(response, str) else json.dumps(response)


class FakeEmbedder:
    def __init__(self, fail=False):
        self.fail = fail

    def top_k_indices(self, query, candidates, k):
        if self.fail:
            raise RuntimeError("offline embedding failure")
        return list(reversed(range(len(candidates))))[:k]


def create(name="files", body="Inspect paths before editing."):
    return dict(action="create", name=name, description=name, body=body)


def update(state, task="Inspect files", outcome="success"):
    state.update(dict(task_description=task, outcome=outcome,
                      reflection_text="Preserve the reusable procedure.",
                      metadata={"task_id": task}))


def skills(state):
    return json.loads(state.serialize())["skills"]


class HermesTests(unittest.TestCase):
    def test_action_lifecycle_and_snapshot(self):
        replacement = "---\nname: files\ndescription: Safe file edits\n---\n\nRead before writing."
        llm = FakeLLM(
            {"actions": [create()]},
            {"actions": [
                dict(action="edit", name="files", content=replacement),
                dict(action="patch", name="files", old_string="Read before", new_string="Inspect before"),
                dict(action="write_file", name="files", file_path="references/check.md", file_content="check old"),
                dict(action="patch", name="files", file_path="references/check.md", old_string="old", new_string="new"),
            ]},
            {"actions": [
                dict(action="remove_file", name="files", file_path="references/check.md"),
                dict(action="delete", name="files", absorbed_into="new-files"),
            ]},
        )
        state = HermesState(llm)
        update(state)
        update(state, task="Second task", outcome="failure")
        skill = skills(state)["files"]
        self.assertIn("Inspect before writing.", skill["content"])
        self.assertEqual(skill["support_files"], {"references/check.md": "check new"})
        self.assertEqual((skill["version"], skill["patch_count"]), (5, 4))
        self.assertEqual(skill["source_task_ids"], ["Inspect files", "Second task"])
        self.assertIn("check new", state.retrieve("files"))

        restored = HermesState(FakeLLM())
        restored.load_serialized(state.serialize())
        self.assertEqual(restored.serialize(), state.serialize())
        self.assertEqual(restored.retrieve("files"), state.retrieve("files"))
        self.assertEqual(restored.stats(), state.stats())
        update(state, task="Third task")
        self.assertEqual(state.retrieve("files"), "")
        self.assertEqual(skills(state)["files"]["absorbed_into"], "new-files")
        self.assertEqual(state.stats()["num_archived_skills"], 1)
        self.assertEqual(state.stats()["num_support_files"], 0)
        for key, value in {"create_count": 1, "edit_count": 1, "patch_count": 2,
                           "write_file_count": 1, "remove_file_count": 1,
                           "delete_count": 1, "action_failures": 0}.items():
            self.assertEqual(state.stats()[key], value)

    def test_retrieval_discloses_index_and_selected_bodies(self):
        actions = [create("files", "File procedure."), create("network", "Network procedure.")]
        for embedder, expected, omitted in [
            (FakeEmbedder(), "Network procedure.", "File procedure."),
            (None, "File procedure.", "Network procedure."),
            (FakeEmbedder(fail=True), "File procedure.", "Network procedure."),
        ]:
            with self.subTest(embedder=embedder):
                state = HermesState(FakeLLM({"actions": actions}), embedder=embedder)
                update(state)
                rendered = state.retrieve("files", k=1)
                self.assertIn("files: files", rendered)
                self.assertIn("network: network", rendered)
                self.assertIn(expected, rendered)
                self.assertNotIn(omitted, rendered)
                self.assertEqual(sum(s["use_count"] for s in skills(state).values()), 1)
        index_only = HermesState(FakeLLM({"actions": actions}), retrieve_full_skills=False)
        update(index_only)
        self.assertNotIn("## Loaded Skill Bodies", index_only.retrieve("files"))

    def test_invalid_paths_and_ambiguous_patches_are_rejected(self):
        bad_paths = ["", "/references/absolute.md", "references/../secret", "references", "private/test.md"]
        attempts = [dict(action="write_file", name="files", file_path=p, file_content="test") for p in bad_paths]
        attempts += [dict(action="patch", name="files", old_string="repeat", new_string="changed")]
        llm = FakeLLM({"actions": [create(body="repeat repeat")]}, {"actions": attempts},
                      {"actions": [dict(action="patch", name="files", old_string="repeat",
                                        new_string="changed", replace_all=True)]})
        state = HermesState(llm)
        update(state)
        before = skills(state)["files"]["content"]
        update(state)
        self.assertEqual(skills(state)["files"]["content"], before)
        self.assertEqual(skills(state)["files"]["support_files"], {})
        self.assertEqual(state.stats()["action_failures"], len(attempts))
        update(state)
        self.assertIn("changed changed", skills(state)["files"]["content"])

    def test_budgets_archive_oldest_and_limit_actions(self):
        actions = [create("first"), create("second"), create("third")]
        state = HermesState(FakeLLM({"actions": actions}), max_skills=1, max_actions_per_update=2)
        update(state)
        self.assertEqual(state.stats()["skill_names"], ["second"])
        self.assertEqual(state.stats()["create_count"], 2)
        self.assertTrue(skills(state)["first"]["archived"])
        state = HermesState(FakeLLM({"actions": [create()]}), max_tokens=1)
        update(state)
        self.assertEqual(state.stats()["num_skills"], 0)

    def test_pending_and_malformed_updates_do_not_change_skills(self):
        llm = FakeLLM("not JSON")
        state = HermesState(llm)
        update(state, outcome="pending")
        self.assertEqual(llm.prompts, [])
        update(state)
        self.assertEqual(state.stats()["parse_failures"], 1)
        self.assertEqual(skills(state), {})
        with self.assertRaises(ValueError):
            state.load_serialized('{"type": "other"}')


if __name__ == "__main__":
    unittest.main()
