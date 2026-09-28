"""Offline checks for extraction, retrieval, checkpoints, and FIFO pruning."""

import json

import numpy as np
import pytest

from Methods.reasoning_bank import (
    FAILED_EXTRACTION_SYSTEM,
    RETRIEVAL_PREAMBLE,
    SUCCESSFUL_EXTRACTION_SYSTEM,
    ReasoningBankState,
    _parse_memory_items,
)


def memory(title="Read the final page", content="Follow pagination to the end."):
    return (
        f"# Memory Item 1\n## Title {title}\n"
        f"## Description Check all matching records.\n## Content {content}\n"
    )


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompt, *, max_tokens, system):
        self.calls.append((prompt, system, max_tokens))
        return next(self.responses)


class FakeEmbedder:
    def __init__(self):
        self.candidates = []

    def embed(self, texts):
        vectors = np.array(
            [[text.lower().count(word) for word in ("alpha", "beta")] for text in texts],
            dtype=np.float32,
        )
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1)

    def top_k_indices(self, query, candidates, k):
        self.candidates = list(candidates)
        scores = self.embed(candidates) @ self.embed([query])[0]
        return np.argsort(scores)[::-1][:k].tolist()


def reflection(query="alpha", outcome="success"):
    return {
        "task_description": query,
        "trajectory": "Read the listing and follow the next-page link.",
        "outcome": outcome,
        "reflection_json": {"key_insight": "Check the entire listing."},
        "metadata": {"task_id": query},
    }


@pytest.mark.parametrize(
    "outcome, system",
    [("success", SUCCESSFUL_EXTRACTION_SYSTEM), ("failure", FAILED_EXTRACTION_SYSTEM)],
)
def test_extracts_outcome_specific_memories_and_caps_items(outcome, system):
    llm = FakeLLM(memory() + memory("Extra item"))
    state = ReasoningBankState(llm, FakeEmbedder(), max_items_per_trajectory=1)
    state.update(reflection(outcome=outcome))

    prompt, actual_system, token_limit = llm.calls[0]
    assert actual_system == system.format(max_items=1)
    assert "Key Insight: Check the entire listing." in prompt
    assert token_limit == 2048
    assert state.stats()["num_items"] == 1
    assert state.stats()[f"{outcome}_entries"] == 1
    assert state.retrieve("alpha") == (
        RETRIEVAL_PREAMBLE + "\n\n"
        "[rb-00001] Title: Read the final page\n"
        "Description: Check all matching records.\n"
        "Content: Follow pagination to the end.\n"
    )


def test_retrieval_ranks_task_queries_and_keeps_failure_items():
    embedder = FakeEmbedder()
    state = ReasoningBankState(
        FakeLLM(memory("alpha strategy"), memory("beta strategy")), embedder
    )
    state.update(reflection("beta", "success"))
    state.update(reflection("alpha", "failure"))

    assert "beta strategy" in state.retrieve("alpha")
    assert "alpha strategy" not in state.retrieve("alpha")
    rendered = state.retrieve("alpha", k=2)
    assert rendered.index("[rb-00002]") < rendered.index("[rb-00001]")
    assert embedder.candidates == ["beta", "alpha"]


def test_snapshot_round_trip_preserves_memories_and_next_id():
    state = ReasoningBankState(FakeLLM(memory()), FakeEmbedder())
    state.update(reflection())
    restored = ReasoningBankState(FakeLLM(memory()), FakeEmbedder())
    restored.load_serialized(state.serialize())

    assert restored.serialize() == state.serialize()
    assert restored.stats() == state.stats()
    assert restored.retrieve("alpha") == state.retrieve("alpha")
    assert json.loads(restored.serialize())["entries"][0]["embedding"] == [1.0, 0.0]
    restored.update(reflection("beta"))
    assert "[rb-00002]" in restored.retrieve("beta")


def test_budget_prunes_whole_oldest_entries():
    response = "# Memory Item 1\n## Title AAAA\n## Description BBBB\n## Content CCCC"
    state = ReasoningBankState(FakeLLM(response, response), FakeEmbedder(), max_tokens=3)
    state.update(reflection("alpha"))
    state.update(reflection("beta"))

    entries = json.loads(state.serialize())["entries"]
    assert [entry["query"] for entry in entries] == ["beta"]
    assert entries[0]["memory_items"][0]["id"] == "rb-00002"
    assert state.stats()["update_count"] == 2


def test_empty_pending_and_unparseable_updates():
    llm = FakeLLM("Unstructured output")
    state = ReasoningBankState(llm, FakeEmbedder())
    assert state.retrieve("alpha") == ""
    state.update(reflection(outcome="pending"))
    state.update({"task_description": "alpha", "outcome": "success"})
    assert llm.calls == []
    state.update(reflection())
    assert state.stats()["parse_failures"] == 1
    assert state.stats()["num_entries"] == 0


def test_parser_accepts_fences_colons_and_multiline_content():
    output = "```markdown\n" + memory().replace("## Title ", "## Title: ") + "More detail.\n```"
    items = _parse_memory_items(output)
    assert items == [{
        "title": "Read the final page",
        "description": "Check all matching records.",
        "content": "Follow pagination to the end. More detail.",
    }]
