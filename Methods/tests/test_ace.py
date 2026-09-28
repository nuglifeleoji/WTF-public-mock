"""Offline checks for ACE curation, complete retrieval, and budget pruning."""

import json

from Methods.ace import PlaybookState


class Curator:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, prompt, max_tokens=2048, system=None):
        self.prompts.append(prompt)
        return next(self.responses)


def additions(*contents):
    return json.dumps({"operations": [
        {"type": "ADD", "section": "verification_checklist", "content": text}
        for text in contents
    ]})


def test_curates_reflection_and_retrieves_all_rules():
    llm = Curator(additions("Check pagination.", "Verify the output schema."))
    state = PlaybookState(llm)
    state.update({
        "task_description": "List every record",
        "trajectory": "Only the first page was returned.",
        "outcome": "failure",
        "reflection_json": {"root_cause_analysis": "Pagination was missed."},
    })
    context = state.retrieve("An unrelated task", k=1)
    assert "[vc-00001] Check pagination." in context
    assert "[vc-00002] Verify the output schema." in context
    assert "Pagination was missed." in llm.prompts[0]
    assert "Only the first page was returned." in llm.prompts[0]


def test_snapshot_restores_rules_and_continues_ids():
    original = PlaybookState(Curator(additions("First rule.")))
    original.update({})
    restored = PlaybookState(Curator(additions("Second rule.")))
    restored.load_serialized(original.serialize())
    assert restored.serialize() == original.serialize()
    restored.update({})
    assert "[vc-00002] Second rule." in restored.retrieve("")
    assert restored.stats()["update_count"] == 2


def test_budget_prunes_oldest_multiline_bullet_as_a_unit():
    old = "Old rule.\n```python\nold_example = 1\n```\n" + "x" * 320
    new = "New rule. " + "y" * 320
    state = PlaybookState(Curator(additions(old), additions(new)), max_tokens=200)
    state.update({})
    assert "old_example" in state.retrieve("")
    state.update({})
    context = state.retrieve("")
    assert "old_example" not in context
    assert "Old rule" not in context
    assert "```" not in context
    assert "New rule" in context
    assert state.stats()["approx_tokens"] <= 200


def test_invalid_output_and_unsupported_operations_do_not_replace_rules():
    payload = json.dumps({"operations": [
        {"type": "DELETE", "section": "verification_checklist"},
        {"type": "ADD", "section": "unrecognized", "content": "Useful lesson."},
    ]})
    state = PlaybookState(Curator("Not JSON", payload))
    initial = state.retrieve("")
    state.update({})
    assert state.retrieve("") == initial
    state.update({})
    assert "## OTHERS\n[others-00001] Useful lesson." in state.retrieve("")
