"""Exercise BrowseComp loading, real local search, tool loops, and grading offline."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from Task.benchmarks.browsecomp import BrowseCompBenchmark, _BrowseCompSearchToolHandler
from Task.prepare_browsecomp_data import derive_key, transform_decrypt
from Task.retrieval.browsecomp import BrowseCompCorpusRetriever


ANSWER = "Explanation: The orchard grows fruit. [10]\nExact Answer: apples\nConfidence: 95%"
JUDGMENT = "extracted_final_answer: apples\nreasoning: Matches the answer.\ncorrect: yes\nconfidence: 95"


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    records = [
        {"docid": "10", "text": "The orchard grows apples.", "url": "https://example.org/orchard"},
        {"docid": "20", "text": "The garage repairs trucks.", "url": "https://example.org/garage"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    return BrowseCompCorpusRetriever(str(path), str(tmp_path / "index.pkl"), backend="local")


@pytest.fixture
def task():
    return {"task_id": "q1", "description": "What fruit does the orchard grow?", "_answer": "apples",
            "_evidence_docids": ["10"]}


def test_ordered_selection_missing_and_empty(tmp_path):
    path = tmp_path / "tasks.jsonl"
    records = [{"query_id": str(i), "query": f"Question {i}", "answer": f"Answer {i}",
                "evidence_docs": [{"docid": "10", "text": "Not retained in task objects"}]}
               for i in range(3)]
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    benchmark = BrowseCompBenchmark(data_path=str(path), task_ids=["2", "0"], max_tasks=1, run_output_dir=None)
    tasks = benchmark.get_tasks()
    assert [item["task_id"] for item in tasks] == ["2", "0"]
    assert tasks[0]["_answer"] == "Answer 2"
    assert tasks[0]["_evidence_docids"] == ["10"]
    assert "Not retained" not in json.dumps(tasks)
    assert BrowseCompBenchmark(data_path=str(path), task_ids=[], run_output_dir=None).get_tasks() == []
    with pytest.raises(ValueError, match="not found"):
        BrowseCompBenchmark(data_path=str(path), task_ids=["missing"], run_output_dir=None).get_tasks()
    with pytest.raises(ValueError, match="duplicates"):
        BrowseCompBenchmark(data_path=str(path), task_ids=["0", "0"], run_output_dir=None)


def test_local_index_roundtrip_and_search(corpus):
    assert corpus.search("orchard apples")[0]["docid"] == "10"
    assert corpus.search("") == []
    assert corpus.get_document("20")["text"] == "The garage repairs trucks."
    assert corpus.get_document("missing") is None
    reloaded = BrowseCompCorpusRetriever(str(corpus.corpus_path), str(corpus.index_path), auto_build=False)
    assert reloaded.search("orchard") == corpus.search("orchard")


def test_search_fallback_without_tokenizer_survives_repeated_calls(corpus):
    handler = _BrowseCompSearchToolHandler(corpus, k=5, snippet_max_tokens=2, include_get_document=True)
    # False is the cached state after the optional tokenizer cannot be loaded.
    handler._tokenizer = False
    for _ in range(2):
        result = json.loads(handler.execute_tool("search", {"query": "orchard"}))
        assert result[0]["snippet"] == "The orchard ..."
    assert json.loads(handler.execute_tool("get_document", {"docid": "10"}))["text"].endswith("apples.")
    assert "error" in json.loads(handler.execute_tool("get_document", {"docid": "missing"}))
    assert handler.tool_call_counts == {"search": 2, "get_document": 2}


class ResponseItem(SimpleNamespace):
    def model_dump(self, **kwargs):
        return vars(self).copy()


class Stream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self.message


def tool_llm(backend):
    """Return SDK-shaped fake responses that search once, then give an answer."""
    llm = SimpleNamespace(model=f"{backend}/mock-model", _api_model="mock-model", temperature=0.0, _backend=backend)
    if backend == "openai":
        search = ResponseItem(type="function_call", id="tool1", call_id="call1", name="search",
                              arguments='{"query": "orchard"}')
        final = ResponseItem(type="message", content=[{"type": "output_text", "text": ANSWER}])
        api = Mock(side_effect=[SimpleNamespace(output=[search], status="completed"),
                                SimpleNamespace(output=[final], status="completed")])
        llm._client = SimpleNamespace(responses=SimpleNamespace(create=api))
    elif backend == "openrouter":
        search = {"id": "tool1", "function": {"name": "search", "arguments": '{"query": "orchard"}'}}
        api = Mock(side_effect=[
            SimpleNamespace(choices=[{"message": {"content": "", "tool_calls": [search]}, "finish_reason": "tool_calls"}]),
            SimpleNamespace(choices=[{"message": {"content": ANSWER}, "finish_reason": "stop"}]),
        ])
        llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=api)))
    else:
        search = SimpleNamespace(type="tool_use", id="tool1", name="search", input={"query": "orchard"})
        final = SimpleNamespace(type="text", text=ANSWER)
        api = Mock(side_effect=[Stream(SimpleNamespace(content=[search], stop_reason="tool_use")),
                                Stream(SimpleNamespace(content=[final], stop_reason="end_turn"))])
        llm._client = SimpleNamespace(messages=SimpleNamespace(stream=api))
    return llm, api


@pytest.mark.parametrize("backend", ["openai", "openrouter", "anthropic"])
def test_tool_solver_judging_and_artifact(tmp_path, corpus, task, backend):
    judge = SimpleNamespace(generate=Mock(return_value=JUDGMENT))
    benchmark = BrowseCompBenchmark(retriever=corpus, judge_client=judge,
                                    run_output_dir=str(tmp_path / "runs"), run_id=backend)
    llm, api = tool_llm(backend)
    with patch.object(_BrowseCompSearchToolHandler, "_get_tokenizer", return_value=None):
        trajectory, result = benchmark.solve(task, "Use evidence from retrieved documents.", llm)
    assert result["outcome"] == "success", result
    assert result["score"] == 1.0
    assert result["retrieval_recall"] == 1.0
    assert result["citation_precision"] == 1.0
    assert result["citation_recall"] == 1.0
    assert result["tool_call_counts"] == {"search": 1}
    assert "TOOL:" in trajectory and ANSWER in trajectory
    assert api.call_count == 2
    assert api.call_args.kwargs["model"] == "mock-model"
    artifact = json.loads((tmp_path / "runs" / backend / "q1.json").read_text())
    assert artifact["query_id"] == "q1"
    assert artifact["status"] == "completed"
    assert artifact["retrieved_docids"] == ["10"]
    assert "_answer" not in artifact
    judge_prompt = judge.generate.call_args.args[0]
    assert "[correct_answer]: apples" in judge_prompt


@pytest.mark.parametrize("backend", ["openai", "openrouter", "anthropic"])
def test_iteration_exhaustion_is_not_a_success(corpus, task, backend):
    judge = SimpleNamespace(generate=Mock(return_value=JUDGMENT))
    benchmark = BrowseCompBenchmark(retriever=corpus, judge_client=judge, agent_max_iterations=1, run_output_dir=None)
    llm, _ = tool_llm(backend)
    with patch.object(_BrowseCompSearchToolHandler, "_get_tokenizer", return_value=None):
        _, result = benchmark.solve(task, "", llm)
    assert result["outcome"] == "failure"
    assert "max_iterations_exceeded" in result["details"]
    judge.generate.assert_not_called()


def test_grader_parse_failure_is_not_success(task):
    judge = SimpleNamespace(generate=Mock(return_value="I cannot grade this response."))
    result = BrowseCompBenchmark(judge_client=judge, run_output_dir=None).evaluate(task, ANSWER)
    assert result["outcome"] == "failure"
    assert "parse_error=True" in result["details"]


def test_decryption_preserves_task_ids():
    password = "synthetic test password"
    def encrypt(value):
        raw = value.encode("utf-8")
        return base64.b64encode(bytes(a ^ b for a, b in zip(raw, derive_key(password, len(raw))))).decode()
    encrypted = {"query_id": "123", "query": encrypt("A synthetic question"),
                 "evidence_docs": [{"docid": encrypt("10"), "text": encrypt("Synthetic evidence")} ]}
    result = transform_decrypt(encrypted, password, {"query_id"})
    assert result == {"query_id": "123", "query": "A synthetic question",
                      "evidence_docs": [{"docid": "10", "text": "Synthetic evidence"}]}


def test_preparation_pipeline_without_network(tmp_path, monkeypatch, capsys):
    """Download-shaped fixtures exercise the same CLI entry points as a fresh setup."""
    import sys
    from Task import prepare_browsecomp_corpus, prepare_browsecomp_data, prepare_browsecomp_qrels

    def encrypt(value):
        raw = value.encode("utf-8")
        key = derive_key(prepare_browsecomp_data.DEFAULT_CANARY, len(raw))
        return base64.b64encode(bytes(a ^ b for a, b in zip(raw, key))).decode()

    encrypted_task = {"query_id": "synthetic-1", "query": encrypt("Which\nfruit?"), "answer": encrypt("apples"),
                      "evidence_docs": [{"docid": encrypt("10")}], "gold_docs": [{"docid": encrypt("10")}]}
    document = {"docid": "10", "text": "The orchard grows apples.", "url": "https://example.org"}
    download = Mock(side_effect=[[encrypted_task], [document]])
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=download))
    task_file, queries = tmp_path / "tasks.jsonl", tmp_path / "queries.tsv"
    monkeypatch.setattr(sys, "argv", ["prepare-data", "--output", str(task_file), "--queries-tsv", str(queries),
                                      "--revision", "pinned-test-revision"])
    prepare_browsecomp_data.main()
    download.assert_called_with("Tevatron/browsecomp-plus", split="test", revision="pinned-test-revision")
    assert queries.read_text() == "synthetic-1\tWhich fruit?\n"

    corpus_file, index_file = tmp_path / "corpus.jsonl", tmp_path / "corpus.pkl"
    monkeypatch.setattr(sys, "argv", ["prepare-corpus", "--output", str(corpus_file), "--index", str(index_file),
                                      "--backend", "local", "--revision", "pinned-corpus-revision"])
    prepare_browsecomp_corpus.main()
    download.assert_called_with("Tevatron/browsecomp-plus-corpus", split="train", revision="pinned-corpus-revision")
    retriever = BrowseCompCorpusRetriever(str(corpus_file), str(index_file), auto_build=False)
    assert retriever.search("apples")[0]["docid"] == "10"
    loaded = BrowseCompBenchmark(data_path=str(task_file), run_output_dir=None).get_tasks()
    assert loaded[0]["_answer"] == "apples"
    assert loaded[0]["_evidence_docids"] == ["10"]

    qrels_dir = tmp_path / "qrels"
    monkeypatch.setattr(sys, "argv", ["prepare-qrels", "--input", str(task_file), "--output-dir", str(qrels_dir)])
    prepare_browsecomp_qrels.main()
    assert (qrels_dir / "queries.tsv").read_text() == "synthetic-1\tWhich fruit?\n"
    assert (qrels_dir / "qrel_evidence.txt").read_text() == "synthetic-1 0 10 1\n"
    assert (qrels_dir / "qrel_gold.txt").read_text() == "synthetic-1 0 10 1\n"
