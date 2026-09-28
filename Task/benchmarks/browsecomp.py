"""BrowseComp-Plus benchmark adapter.

This adapter loads decrypted BrowseComp query records, retrieves from the full
BrowseComp corpus through a local BM25-style retriever, asks the model to answer
with citations, judges the answer against the gold answer, and writes
official-style run JSON files for later external evaluation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .base import Benchmark
from ..retrieval.browsecomp import BrowseCompCorpusRetriever


_ANSWER_SYSTEM_PROMPT = """\
You are a deep research agent. Use only the provided documents to answer the
question. Follow the final response format exactly:

Explanation: {your explanation for your final answer. For this explanation
section only, cite your evidence documents inline by enclosing their docids in
square brackets [] at the end of sentences. For example, [20].}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}
"""


_GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()
_QUERY_TEMPLATE = """\
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search and get_document tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the search and get_document tools multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

_QUERY_TEMPLATE_NO_GET_DOCUMENT = """\
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

_QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION = """\
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

def _maybe_model_dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="python")
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError(f"Unsupported response item type: {type(obj)!r}")


def _get_field(obj: Any, *names: str, default: Any = 0) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _chat_completion_tool_definitions(tool_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chat_tools: list[dict[str, Any]] = []
    for tool_def in tool_defs:
        function_def = {
            "name": tool_def["name"],
            "description": tool_def.get("description", ""),
            "parameters": tool_def.get("parameters", {"type": "object"}),
        }
        if "strict" in tool_def:
            function_def["strict"] = tool_def["strict"]
        chat_tools.append({"type": "function", "function": function_def})
    return chat_tools


def _format_query(question: str, query_template: str | None) -> str:
    if query_template is None:
        return question
    if query_template == "QUERY_TEMPLATE":
        return _QUERY_TEMPLATE.format(Question=question)
    if query_template == "QUERY_TEMPLATE_NO_GET_DOCUMENT":
        return _QUERY_TEMPLATE_NO_GET_DOCUMENT.format(Question=question)
    if query_template == "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION":
        return _QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION.format(Question=question)
    raise ValueError(f"Unknown BrowseComp query template: {query_template}")


def _normalize_playbook_context(context: str) -> str:
    stripped = (context or "").strip()
    if not stripped:
        return ""
    meaningful_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if meaningful_lines and all(line.startswith("## ") for line in meaningful_lines):
        return ""
    return stripped


def _extract_retrieved_docids_from_result(result: list[dict[str, Any]]) -> list[str]:
    retrieved_docids_set: set[str] = set()
    for item in result:
        if item.get("type") != "tool_call":
            continue
        tool_name = str(item.get("tool_name") or "")
        if "search" not in tool_name.lower() and "retrieval" not in tool_name.lower():
            continue
        output = item.get("output")
        parsed = None
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
            except Exception:
                parsed = None
        elif isinstance(output, list):
            parsed = output

        if isinstance(parsed, list):
            for elem in parsed:
                if isinstance(elem, dict) and "docid" in elem:
                    retrieved_docids_set.add(str(elem["docid"]))
        elif isinstance(output, str):
            for match in re.findall(r'"docid"\s*:\s*"([^"]+)"', output):
                retrieved_docids_set.add(str(match))
            for match in re.findall(r'"docid"\s*:\s*(\d+)', output):
                retrieved_docids_set.add(str(match))
    return sorted(retrieved_docids_set)


class _BrowseCompSearchToolHandler:
    def __init__(
        self,
        retriever: BrowseCompCorpusRetriever,
        *,
        k: int,
        snippet_max_tokens: int,
        include_get_document: bool,
    ):
        self.retriever = retriever
        self.k = k
        self.snippet_max_tokens = snippet_max_tokens
        self.include_get_document = include_get_document
        self.tool_call_counts: dict[str, int] = {}
        self.retrieved_docids: set[str] = set()
        self._tokenizer = None

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "name": "search",
                "description": (
                    f"Perform a search on the BrowseComp-Plus corpus. Returns top-{self.k} "
                    "hits with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Targeted search query string.",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]
        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "name": "get_document",
                    "description": "Retrieve a full document by its docid.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve.",
                            }
                        },
                        "required": ["docid"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            )
        return tools

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1
        if tool_name == "search":
            return self._search(str(arguments["query"]))
        if tool_name == "get_document":
            return self._get_document(str(arguments["docid"]))
        raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str) -> str:
        candidates = self.retriever.search(query, k=self.k)
        results = []
        for cand in candidates:
            self.retrieved_docids.add(str(cand["docid"]))
            snippet = self._truncate_to_tokens(str(cand.get("text", "")))
            payload: dict[str, Any] = {
                "docid": str(cand["docid"]),
                "snippet": snippet,
            }
            if cand.get("score") is not None:
                payload["score"] = float(cand["score"])
            results.append(payload)
        return json.dumps(results, indent=2, ensure_ascii=False)

    def _get_document(self, docid: str) -> str:
        self.retrieved_docids.add(str(docid))
        doc = self.retriever.get_document(docid)
        if doc is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(doc, indent=2, ensure_ascii=False)

    def _truncate_to_tokens(self, text: str) -> str:
        compact = " ".join((text or "").split())
        if not compact or self.snippet_max_tokens <= 0:
            return compact

        # Avoid tokenizing arbitrarily long raw documents just to keep a short snippet.
        approx_char_window = max(self.snippet_max_tokens * 12, 512)
        if len(compact) > approx_char_window * 2:
            compact = compact[: approx_char_window * 2]

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            parts = compact.split()
            if len(parts) <= self.snippet_max_tokens:
                return compact
            return " ".join(parts[: self.snippet_max_tokens]) + " ..."

        encoded = tokenizer(
            compact,
            add_special_tokens=False,
            truncation=True,
            max_length=self.snippet_max_tokens,
            return_attention_mask=False,
        )
        tokens = encoded.get("input_ids", [])
        if len(tokens) < self.snippet_max_tokens and len(compact) <= approx_char_window:
            return compact
        truncated = tokenizer.decode(
            tokens,
            skip_special_tokens=True,
        )
        return " ".join(truncated.split()) + " ..."

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return None if self._tokenizer is False else self._tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        except Exception:
            self._tokenizer = False
        return None if self._tokenizer is False else self._tokenizer


def _extract_citations(response_text: str) -> list[str]:
    if not response_text:
        return []

    single_square = re.findall(r"\[(\d+)\]", response_text)
    multi_square = re.findall(r"\[([^\[\]]*?)\]", response_text)
    single_fullwidth = re.findall(r"【(\d+)】", response_text)
    multi_fullwidth = re.findall(r"【([^【】]*?)】", response_text)

    docids: set[str] = set(single_square) | set(single_fullwidth)
    for group in multi_square + multi_fullwidth:
        for docid in re.findall(r"\d+", group):
            docids.add(docid)
    return sorted(docids)


def _parse_judge_response(judge_response: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }
    if not judge_response:
        result["parse_error"] = True
        return result

    answer_match = re.search(
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not answer_match:
        answer_match = re.search(
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not answer_match:
        answer_match = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    reasoning_match = re.search(
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not reasoning_match:
        reasoning_match = re.search(
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not reasoning_match:
        reasoning_match = re.search(
            r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    correct_match = re.search(r"\*\*correct:\*\*\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(
            r"\*\*correct\*\*:\s*(yes|no)",
            judge_response,
            re.IGNORECASE,
        )
    if not correct_match:
        correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    confidence_match = re.search(
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?",
        judge_response,
        re.IGNORECASE,
    )
    if not confidence_match:
        confidence_match = re.search(
            r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?",
            judge_response,
            re.IGNORECASE,
        )
    if not confidence_match:
        confidence_match = re.search(
            r"confidence:\s*(\d+(?:\.\d+)?)\s*%?",
            judge_response,
            re.IGNORECASE,
        )
    if confidence_match:
        result["confidence"] = min(float(confidence_match.group(1)), 100.0)

    if result["correct"] is None:
        result["parse_error"] = True
    return result


def _compute_recall(predicted_docids: list[str], relevant_docids: list[str]) -> float:
    if not relevant_docids:
        return 0.0
    return len(set(predicted_docids) & set(relevant_docids)) / float(len(set(relevant_docids)))


def _compute_citation_metrics(cited_docids: list[str], relevant_docids: list[str]) -> dict:
    if not cited_docids:
        return {"precision": 0.0, "recall": 0.0}

    cited = set(cited_docids)
    relevant = set(relevant_docids)
    overlap = cited & relevant
    return {
        "precision": len(overlap) / float(len(cited)) if cited else 0.0,
        "recall": len(overlap) / float(len(relevant)) if relevant else 0.0,
    }


def _clean_text(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


class BrowseCompBenchmark(Benchmark):
    """Benchmark adapter for decrypted BrowseComp-Plus JSONL data."""

    def __init__(
        self,
        data_path: str = "data/browsecomp/browsecomp_plus_decrypted.jsonl",
        max_tasks: Optional[int] = None,
        retrieval_top_k: int = 5,
        max_doc_chars: int = 1600,
        corpus_path: str = "data/browsecomp/corpus.jsonl",
        corpus_index_path: str = "data/browsecomp/indexes/bm25",
        retriever_backend: str = "auto",
        run_output_dir: Optional[str] = "data/browsecomp/runs",
        run_id: str = "browsecomp_run",
        judge_model: Optional[str] = None,
        agent_max_output_tokens: int = 10_000,
        agent_max_iterations: int = 100,
        snippet_max_tokens: int = 512,
        include_get_document: bool = False,
        retriever_build_threads: int = 8,
        query_template: str = "QUERY_TEMPLATE_NO_GET_DOCUMENT",
        reasoning_effort: Optional[str] = None,
        judge_client=None,
        retriever: Optional[BrowseCompCorpusRetriever] = None,
        task_ids: Optional[list[str]] = None,
    ):
        self.data_path = Path(data_path)
        self.max_tasks = max_tasks
        # Pinned task_ids override max_tasks; preserve user-specified order so
        # runs are reproducible and Set2/Set3 splits stay in a documented order.
        self.task_ids: Optional[list[str]] = (
            [str(t) for t in task_ids] if task_ids is not None else None
        )
        if self.task_ids is not None and len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("BrowseComp task_ids must not contain duplicates")
        self.retrieval_top_k = retrieval_top_k
        self.max_doc_chars = max_doc_chars
        self.corpus_path = Path(corpus_path)
        self.corpus_index_path = Path(corpus_index_path)
        self.retriever_backend = retriever_backend
        self.run_id = run_id
        self.judge_model = judge_model
        self.agent_max_output_tokens = agent_max_output_tokens
        self.agent_max_iterations = agent_max_iterations
        self.snippet_max_tokens = snippet_max_tokens
        self.include_get_document = include_get_document
        self.retriever_build_threads = retriever_build_threads
        self.query_template = query_template
        self.reasoning_effort = reasoning_effort
        self._judge_client = judge_client
        self._retriever = retriever
        self._tasks: Optional[list[dict]] = None

        self.run_output_dir: Optional[Path]
        if run_output_dir:
            self.run_output_dir = Path(run_output_dir) / run_id
            self.run_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_output_dir = None

    def get_tasks(self) -> list[dict]:
        if self._tasks is not None:
            return self._tasks
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"BrowseComp data file not found: {self.data_path}. "
                "Run python -m Task.prepare_browsecomp_data first."
            )

        # Read all eligible records into a {query_id -> task_dict} map first;
        # then either filter+order by self.task_ids, or take the first
        # self.max_tasks. This keeps memory bounded (BrowseComp JSONL is ~830
        # rows), retaining only fields needed to solve and evaluate each task.
        wanted: Optional[set[str]] = set(self.task_ids) if self.task_ids is not None else None
        records: dict[str, dict] = {}
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                qid = str(record["query_id"])
                if wanted is not None and qid not in wanted:
                    continue
                records[qid] = {
                    "task_id": qid,
                    "description": str(record.get("query", "")),
                    "_answer": str(record.get("answer", "")),
                    "_evidence_docids": [
                        str(doc.get("docid", "")) for doc in (record.get("evidence_docs") or [])
                    ],
                }
                if self.task_ids is None and self.max_tasks is not None and len(records) >= self.max_tasks:
                    break

        tasks: list[dict]
        if self.task_ids is not None:
            missing = [tid for tid in self.task_ids if tid not in records]
            if missing:
                raise ValueError(
                    f"BrowseComp task_ids not found in {self.data_path}: {missing[:10]}"
                )
            # Preserve the order the user provided.
            tasks = [records[tid] for tid in self.task_ids]
        else:
            tasks = list(records.values())

        self._tasks = tasks
        return self._tasks

    def evaluate(self, task: dict, agent_output: str) -> dict:
        try:
            judge_client = self._ensure_judge_client()
            return self._score_answer(
                task=task,
                answer=agent_output,
                judge_client=judge_client,
            )
        except Exception as exc:
            return {
                "outcome": "failure",
                "score": 0.0,
                "details": f"Execution error: {type(exc).__name__}: {exc}",
            }

    def solve(self, task: dict, context: str, llm) -> tuple[str, dict]:
        try:
            if self._should_use_tool_agent(llm):
                return self._solve_with_tool_agent(task, context, llm)

            retriever = self._ensure_retriever()
            retrieved_docs = retriever.search(task["description"], k=self.retrieval_top_k)
            retrieved_docids = [doc["docid"] for doc in retrieved_docs]

            prompt = self._build_answer_prompt(task["description"], context, retrieved_docs)
            answer = llm.generate(prompt, max_tokens=2048, system=_ANSWER_SYSTEM_PROMPT)

            judge_client = self._ensure_judge_client(llm)
            result = self._score_answer(
                task=task,
                answer=answer,
                judge_client=judge_client,
                retrieved_docids=retrieved_docids,
            )
            self._write_run_artifact(
                query_id=task["task_id"],
                status="completed",
                result=[{"type": "output_text", "output": answer}],
                retrieved_docids=retrieved_docids,
                tool_call_counts={"search": 1},
                model=getattr(llm, "model", "unknown"),
            )

            trajectory = self._build_trajectory(
                task_description=task["description"],
                context=context,
                retrieved_docs=retrieved_docs,
                answer=answer,
                judge_text=result.get("judge_text", ""),
            )
            result.pop("judge_text", None)
            return trajectory, result
        except Exception as exc:
            return "", {
                "outcome": "failure",
                "score": 0.0,
                "details": f"Execution error: {type(exc).__name__}: {exc}",
            }

    def _should_use_tool_agent(self, llm) -> bool:
        return getattr(llm, "_backend", None) in ("openai", "anthropic", "openrouter")

    def _solve_with_tool_agent(self, task: dict, context: str, llm) -> tuple[str, dict]:
        retriever = self._ensure_retriever()
        tool_handler = _BrowseCompSearchToolHandler(
            retriever,
            k=self.retrieval_top_k,
            snippet_max_tokens=self.snippet_max_tokens,
            include_get_document=self.include_get_document,
        )
        query_prompt = self._build_tool_agent_query(task["description"], context)
        backend = getattr(llm, "_backend", None)
        if backend == "anthropic":
            agent_run = self._run_anthropic_tool_conversation(llm, query_prompt, tool_handler)
        elif backend == "openrouter":
            agent_run = self._run_chat_tool_conversation(llm, query_prompt, tool_handler)
        else:
            agent_run = self._run_tool_conversation(llm, query_prompt, tool_handler)
        normalized_results = agent_run["result"]
        answer = self._extract_final_output_text(normalized_results)
        status = str(agent_run.get("status", "completed"))
        retrieved_docids = list(agent_run.get("retrieved_docids", []))
        tool_call_counts = dict(agent_run.get("tool_call_counts", {}))
        usage = dict(agent_run.get("usage", {}))

        self._write_run_artifact(
            query_id=task["task_id"],
            status=status,
            result=normalized_results,
            retrieved_docids=retrieved_docids,
            tool_call_counts=tool_call_counts,
            model=getattr(llm, "model", "unknown"),
            usage=usage,
        )

        if status != "completed" or not answer.strip():
            trajectory = self._build_tool_trajectory(
                task_description=task["description"],
                context=context,
                normalized_results=normalized_results,
                judge_text="",
            )
            return trajectory, {
                "outcome": "failure",
                "score": 0.0,
                "cited_docids": [],
                "citation_precision": 0.0,
                "citation_recall": 0.0,
                "retrieved_docids": retrieved_docids,
                "tool_call_counts": tool_call_counts,
                "details": (
                    f"status={status} tool_calls={sum(tool_call_counts.values())} "
                    "Agent did not produce a completed final answer."
                ),
            }

        judge_client = self._ensure_judge_client(llm)
        result = self._score_answer(
            task=task,
            answer=answer,
            judge_client=judge_client,
            retrieved_docids=retrieved_docids,
        )
        result["tool_call_counts"] = tool_call_counts
        result["details"] = (
            f"tool_calls={sum(tool_call_counts.values())} "
            f"{result['details']}"
        )
        trajectory = self._build_tool_trajectory(
            task_description=task["description"],
            context=context,
            normalized_results=normalized_results,
            judge_text=result.get("judge_text", ""),
        )
        result.pop("judge_text", None)
        return trajectory, result

    def _ensure_retriever(self) -> BrowseCompCorpusRetriever:
        if self._retriever is None:
            self._retriever = BrowseCompCorpusRetriever(
                corpus_path=str(self.corpus_path),
                index_path=str(self.corpus_index_path),
                backend=self.retriever_backend,
                auto_build=True,
                build_threads=self.retriever_build_threads,
            )
        return self._retriever

    def _ensure_judge_client(self, llm=None):
        if self._judge_client is not None:
            return self._judge_client
        if self.judge_model and self.judge_model != getattr(llm, "model", None):
            from runtime.llm import LLMClient

            self._judge_client = LLMClient(model=self.judge_model, temperature=0.0)
        elif llm is not None:
            self._judge_client = llm
        else:
            raise RuntimeError(
                "BrowseComp evaluation requires a judge client. "
                "Pass judge_client or configure browsecomp_judge_model."
            )
        return self._judge_client

    def _run_tool_conversation(
        self,
        llm,
        query_prompt: str,
        tool_handler: _BrowseCompSearchToolHandler,
    ) -> dict[str, Any]:
        response_client = getattr(getattr(llm, "_client", None), "responses", None)
        if response_client is None:
            raise RuntimeError("OpenAI Responses API client is unavailable for BrowseComp.")

        input_messages: list[dict[str, Any]] = [{"role": "user", "content": query_prompt}]
        combined_output: list[Any] = []
        tool_outputs: dict[str, dict[str, Any]] = {}
        cumulative_usage = {
            "input_tokens": 0,
            "input_tokens_cached": 0,
            "output_tokens": 0,
            "included_reasoning_tokens": 0,
            "total_tokens": 0,
        }

        model_name = getattr(llm, "_api_model", None) or llm.model
        base_request: dict[str, Any] = {
            "model": model_name,
            "max_output_tokens": self.agent_max_output_tokens,
            "input": input_messages,
            "tools": tool_handler.get_tool_definitions(),
            "tool_choice": "auto",
            "truncation": "auto",
        }
        if not str(model_name).lower().startswith("o") and getattr(llm, "temperature", None) is not None:
            base_request["temperature"] = llm.temperature
        if self.reasoning_effort:
            base_request["reasoning"] = {"effort": self.reasoning_effort, "summary": "detailed"}

        for _ in range(self.agent_max_iterations):
            response = response_client.create(**base_request)
            output_items = list(getattr(response, "output", []) or [])
            combined_output.extend(output_items)
            usage = getattr(response, "usage", None)
            if usage is not None:
                cumulative_usage["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
                cumulative_usage["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
                cumulative_usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
                input_details = getattr(usage, "input_tokens_details", None)
                output_details = getattr(usage, "output_tokens_details", None)
                if input_details is not None:
                    cumulative_usage["input_tokens_cached"] += int(
                        getattr(input_details, "cached_tokens", 0) or 0
                    )
                if output_details is not None:
                    cumulative_usage["included_reasoning_tokens"] += int(
                        getattr(output_details, "reasoning_tokens", 0) or 0
                    )

            function_calls = [
                item for item in output_items if getattr(item, "type", None) == "function_call"
            ]
            if not function_calls:
                return self._normalize_tool_conversation(
                    status=str(getattr(response, "status", "completed")),
                    combined_output=combined_output,
                    tool_outputs=tool_outputs,
                    tool_call_counts=tool_handler.tool_call_counts,
                    retrieved_docids=sorted(tool_handler.retrieved_docids),
                    usage=cumulative_usage,
                )

            for item in output_items:
                serialized = _maybe_model_dump(item)
                serialized.pop("status", None)
                input_messages.append(serialized)

            for tool_call in function_calls:
                try:
                    args = json.loads(tool_call.arguments)
                    tool_result = tool_handler.execute_tool(tool_call.name, args)
                    tool_outputs[str(tool_call.id)] = {
                        "output": tool_result,
                        "status": "completed",
                        "error": None,
                    }
                    input_messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": tool_result,
                        }
                    )
                except Exception as exc:
                    error_msg = f"Error executing {tool_call.name}: {exc}"
                    tool_outputs[str(tool_call.id)] = {
                        "output": error_msg,
                        "status": "error",
                        "error": error_msg,
                    }
                    input_messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": error_msg,
                        }
                    )

            base_request["input"] = input_messages

        return self._normalize_tool_conversation(
            status="max_iterations_exceeded",
            combined_output=combined_output,
            tool_outputs=tool_outputs,
            tool_call_counts=tool_handler.tool_call_counts,
            retrieved_docids=sorted(tool_handler.retrieved_docids),
            usage=cumulative_usage,
        )

    def _run_chat_tool_conversation(
        self,
        llm,
        query_prompt: str,
        tool_handler: _BrowseCompSearchToolHandler,
    ) -> dict[str, Any]:
        chat = getattr(getattr(llm, "_client", None), "chat", None)
        chat_completions = getattr(chat, "completions", None)
        if chat_completions is None:
            raise RuntimeError("OpenAI-compatible Chat Completions client is unavailable for BrowseComp.")

        messages: list[dict[str, Any]] = [{"role": "user", "content": query_prompt}]
        normalized_results: list[dict[str, Any]] = []
        cumulative_usage = {
            "input_tokens": 0,
            "input_tokens_cached": 0,
            "output_tokens": 0,
            "included_reasoning_tokens": 0,
            "total_tokens": 0,
        }
        status = "completed"

        model_name = getattr(llm, "_api_model", None) or getattr(llm, "model", "unknown")
        tool_definitions = _chat_completion_tool_definitions(tool_handler.get_tool_definitions())

        for _ in range(self.agent_max_iterations):
            request: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "tools": tool_definitions,
                "tool_choice": "auto",
                "max_tokens": self.agent_max_output_tokens,
            }
            if getattr(llm, "temperature", None) is not None:
                request["temperature"] = llm.temperature
            if self.reasoning_effort:
                request["reasoning"] = {"effort": self.reasoning_effort}

            response = chat_completions.create(**request)
            usage = getattr(response, "usage", None)
            if usage is not None:
                input_tokens = int(_get_field(usage, "prompt_tokens", "input_tokens", default=0) or 0)
                output_tokens = int(
                    _get_field(usage, "completion_tokens", "output_tokens", default=0) or 0
                )
                total_tokens = int(
                    _get_field(usage, "total_tokens", default=input_tokens + output_tokens)
                    or input_tokens + output_tokens
                )
                input_details = _get_field(
                    usage,
                    "prompt_tokens_details",
                    "input_tokens_details",
                    default=None,
                )
                output_details = _get_field(
                    usage,
                    "completion_tokens_details",
                    "output_tokens_details",
                    default=None,
                )
                cumulative_usage["input_tokens"] += input_tokens
                cumulative_usage["output_tokens"] += output_tokens
                cumulative_usage["total_tokens"] += total_tokens
                if input_details is not None:
                    cumulative_usage["input_tokens_cached"] += int(
                        _get_field(input_details, "cached_tokens", default=0) or 0
                    )
                if output_details is not None:
                    cumulative_usage["included_reasoning_tokens"] += int(
                        _get_field(output_details, "reasoning_tokens", default=0) or 0
                    )

            choices = list(_get_field(response, "choices", default=[]) or [])
            if not choices:
                status = "empty_response"
                break

            choice = choices[0]
            message = _get_field(choice, "message", default=None)
            if message is None:
                status = "empty_response"
                break

            content = _get_field(message, "content", default=None) or ""
            if content.strip():
                normalized_results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": content,
                    }
                )

            tool_calls = list(_get_field(message, "tool_calls", default=[]) or [])
            if not tool_calls:
                finish_reason = _get_field(choice, "finish_reason", default=None)
                status = "completed" if finish_reason in (None, "stop") else str(finish_reason)
                break

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [],
            }
            for tool_call in tool_calls:
                function = _get_field(tool_call, "function", default=None)
                tool_call_id = str(_get_field(tool_call, "id", default=""))
                tool_name = str(_get_field(function, "name", default="") if function is not None else "")
                raw_arguments = (
                    _get_field(function, "arguments", default="{}") if function is not None else "{}"
                )
                arguments_text = (
                    json.dumps(raw_arguments)
                    if isinstance(raw_arguments, dict)
                    else str(raw_arguments or "{}")
                )

                assistant_message["tool_calls"].append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments_text,
                        },
                    }
                )

            messages.append(assistant_message)

            for tool_call in tool_calls:
                function = _get_field(tool_call, "function", default=None)
                tool_call_id = str(_get_field(tool_call, "id", default=""))
                tool_name = str(_get_field(function, "name", default="") if function is not None else "")
                raw_arguments = (
                    _get_field(function, "arguments", default="{}") if function is not None else "{}"
                )
                try:
                    args = (
                        dict(raw_arguments)
                        if isinstance(raw_arguments, dict)
                        else json.loads(str(raw_arguments or "{}"))
                    )
                    arguments_text = json.dumps(args, ensure_ascii=False)
                    tool_result = tool_handler.execute_tool(tool_name, args)
                except Exception as exc:
                    args = {}
                    arguments_text = (
                        json.dumps(raw_arguments, ensure_ascii=False)
                        if isinstance(raw_arguments, dict)
                        else str(raw_arguments or "{}")
                    )
                    tool_result = f"Error executing {tool_name}: {exc}"

                normalized_results.append(
                    {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "arguments": arguments_text,
                        "output": tool_result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result,
                    }
                )
        else:
            status = "max_iterations_exceeded"

        fallback_retrieved = _extract_retrieved_docids_from_result(normalized_results)
        return {
            "status": status,
            "result": normalized_results,
            "tool_call_counts": dict(tool_handler.tool_call_counts),
            "retrieved_docids": fallback_retrieved or sorted(tool_handler.retrieved_docids),
            "usage": cumulative_usage,
        }

    def _run_anthropic_tool_conversation(
        self,
        llm,
        query_prompt: str,
        tool_handler: _BrowseCompSearchToolHandler,
    ) -> dict[str, Any]:
        client = getattr(llm, "_client", None)
        messages_api = getattr(client, "messages", None)
        if messages_api is None:
            raise RuntimeError("Anthropic Messages API client is unavailable for BrowseComp.")

        tool_definitions = [
            {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "input_schema": tool_def["parameters"],
            }
            for tool_def in tool_handler.get_tool_definitions()
        ]

        messages: list[dict[str, Any]] = [{"role": "user", "content": query_prompt}]
        normalized_results: list[dict[str, Any]] = []
        cumulative_usage = {
            "input_tokens": 0,
            "input_tokens_cached": 0,
            "output_tokens": 0,
            "included_reasoning_tokens": 0,
            "total_tokens": 0,
        }
        status = "completed"

        for _ in range(self.agent_max_iterations):
            request: dict[str, Any] = {
                "model": getattr(llm, "_api_model", None) or llm.model,
                "max_tokens": self.agent_max_output_tokens,
                "messages": messages,
                "tools": tool_definitions,
            }
            if getattr(llm, "temperature", None) is not None:
                request["temperature"] = llm.temperature

            with messages_api.stream(**request) as stream:
                for _ in stream:
                    pass
                response = stream.get_final_message()

            usage = getattr(response, "usage", None)
            if usage is not None:
                in_tok = int(getattr(usage, "input_tokens", 0) or 0)
                out_tok = int(getattr(usage, "output_tokens", 0) or 0)
                cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
                cumulative_usage["input_tokens"] += in_tok
                cumulative_usage["output_tokens"] += out_tok
                cumulative_usage["input_tokens_cached"] += cached
                cumulative_usage["total_tokens"] += in_tok + out_tok

            content_blocks = list(getattr(response, "content", []) or [])
            assistant_content: list[dict[str, Any]] = []
            tool_uses: list[Any] = []
            for block in content_blocks:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text = getattr(block, "text", "") or ""
                    assistant_content.append({"type": "text", "text": text})
                    if text.strip():
                        normalized_results.append(
                            {
                                "type": "output_text",
                                "tool_name": None,
                                "arguments": None,
                                "output": text,
                            }
                        )
                elif block_type == "thinking":
                    thinking_text = getattr(block, "thinking", "") or ""
                    signature = getattr(block, "signature", None)
                    thinking_block: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": thinking_text,
                    }
                    if signature is not None:
                        thinking_block["signature"] = signature
                    assistant_content.append(thinking_block)
                    if thinking_text.strip():
                        normalized_results.append(
                            {
                                "type": "reasoning",
                                "tool_name": None,
                                "arguments": None,
                                "output": thinking_text,
                            }
                        )
                elif block_type == "tool_use":
                    tool_uses.append(block)
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "input": getattr(block, "input", {}) or {},
                        }
                    )

            if not tool_uses:
                status = (
                    "completed"
                    if str(getattr(response, "stop_reason", "end_turn")) in ("end_turn", "stop_sequence")
                    else str(getattr(response, "stop_reason", "completed"))
                )
                break

            messages.append({"role": "assistant", "content": assistant_content})

            tool_result_blocks: list[dict[str, Any]] = []
            for tool_use in tool_uses:
                tool_name = str(getattr(tool_use, "name", ""))
                tool_use_id = str(getattr(tool_use, "id", ""))
                raw_input = getattr(tool_use, "input", {}) or {}
                try:
                    args = dict(raw_input) if isinstance(raw_input, dict) else json.loads(str(raw_input))
                    tool_result = tool_handler.execute_tool(tool_name, args)
                    is_error = False
                except Exception as exc:
                    tool_result = f"Error executing {tool_name}: {exc}"
                    is_error = True

                normalized_results.append(
                    {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "arguments": json.dumps(args) if not is_error else json.dumps(raw_input if isinstance(raw_input, dict) else {}),
                        "output": tool_result,
                    }
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_result,
                        "is_error": is_error,
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})
        else:
            status = "max_iterations_exceeded"

        fallback_retrieved = _extract_retrieved_docids_from_result(normalized_results)
        return {
            "status": status,
            "result": normalized_results,
            "tool_call_counts": dict(tool_handler.tool_call_counts),
            "retrieved_docids": fallback_retrieved or sorted(tool_handler.retrieved_docids),
            "usage": cumulative_usage,
        }

    def _normalize_tool_conversation(
        self,
        *,
        status: str,
        combined_output: list[Any],
        tool_outputs: dict[str, dict[str, Any]],
        tool_call_counts: dict[str, int],
        retrieved_docids: list[str],
        usage: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        normalized_results: list[dict[str, Any]] = []
        for item in combined_output:
            payload = _maybe_model_dump(item)
            item_type = payload.get("type")
            if item_type == "function_call":
                item_id = str(payload.get("id"))
                normalized_results.append(
                    {
                        "type": "tool_call",
                        "tool_name": payload.get("name"),
                        "arguments": payload.get("arguments"),
                        "output": tool_outputs.get(item_id, {}).get("output"),
                    }
                )
            elif item_type == "reasoning":
                normalized_results.append(
                    {
                        "type": "reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": payload.get("summary"),
                    }
                )
            elif item_type == "message":
                text_parts = []
                for part in payload.get("content", []) or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text_parts.append(str(part.get("text", "")))
                text = "\n".join(chunk for chunk in text_parts if chunk).strip()
                if text:
                    normalized_results.append(
                        {
                            "type": "output_text",
                            "tool_name": None,
                            "arguments": None,
                            "output": text,
                        }
                    )

        normalized_retrieved_docids = _extract_retrieved_docids_from_result(normalized_results)

        return {
            "status": status,
            "result": normalized_results,
            "tool_call_counts": dict(tool_call_counts),
            "retrieved_docids": normalized_retrieved_docids or list(retrieved_docids),
            "usage": usage or {},
        }

    def _score_answer(
        self,
        task: dict,
        answer: str,
        judge_client,
        retrieved_docids: Optional[list[str]] = None,
    ) -> dict:
        judge_prompt = _GRADER_TEMPLATE.format(
            question=task["description"],
            response=answer,
            correct_answer=task.get("_answer", ""),
        )
        judge_text = judge_client.generate(judge_prompt, max_tokens=1024)
        judge_result = _parse_judge_response(judge_text)

        cited_docids = _extract_citations(answer)
        evidence_docids = task.get("_evidence_docids", [])
        citation_metrics = _compute_citation_metrics(cited_docids, evidence_docids)
        correct = bool(judge_result.get("correct")) and not judge_result.get("parse_error")

        detail_parts = [
            f"judge_correct={correct}",
            f"parse_error={judge_result.get('parse_error', False)}",
            f"confidence={judge_result.get('confidence', 'n/a')}",
            f"citation_precision={citation_metrics['precision']:.3f}",
            f"citation_recall={citation_metrics['recall']:.3f}",
        ]

        result = {
            "outcome": "success" if correct else "failure",
            "score": 1.0 if correct else 0.0,
            "cited_docids": cited_docids,
            "citation_precision": citation_metrics["precision"],
            "citation_recall": citation_metrics["recall"],
            "judge_text": judge_text,
        }

        if retrieved_docids is not None:
            retrieval_recall = _compute_recall(retrieved_docids, evidence_docids)
            detail_parts.insert(3, f"retrieval_recall={retrieval_recall:.3f}")
            result["retrieval_recall"] = retrieval_recall
            result["retrieved_docids"] = retrieved_docids

        result["details"] = " ".join(detail_parts)
        return result

    def _build_tool_agent_query(self, question: str, context: str) -> str:
        prompt = _format_query(question, self.query_template)
        playbook_context = _normalize_playbook_context(context)
        if not playbook_context:
            return prompt
        insertion = (
            "Additional guidance from prior tasks. Follow it only when it helps and does not "
            "conflict with the question or retrieved evidence.\n"
            f"{playbook_context}\n\nQuestion:"
        )
        return prompt.replace("Question:", insertion, 1)

    def _extract_final_output_text(self, normalized_results: list[dict[str, Any]]) -> str:
        for item in reversed(normalized_results):
            if item.get("type") == "output_text":
                return str(item.get("output", ""))
        return ""

    def _build_answer_prompt(self, question: str, context: str, docs: list[dict]) -> str:
        context_section = _normalize_playbook_context(context) or "N/A"
        doc_blocks = []
        for doc in docs:
            doc_blocks.append(
                "\n".join(
                    [
                        f"[{doc['docid']}]",
                        f"URL: {doc['url'] or 'N/A'}",
                        _clean_text(doc["text"], self.max_doc_chars),
                    ]
                )
            )

        return (
            f"Question:\n{question}\n\n"
            f"Retrieved Knowledge From Playbook:\n{context_section}\n\n"
            f"Retrieved BrowseComp Documents:\n\n" + "\n\n".join(doc_blocks) + "\n\n"
            "Answer the question using only the documents above. Follow this exact format:\n"
            "Explanation: {cite evidence docids inline in square brackets at the end of sentences}\n"
            "Exact Answer: {your succinct, final answer}\n"
            "Confidence: {your confidence score between 0% and 100% for your answer}"
        )

    def _build_trajectory(
        self,
        task_description: str,
        context: str,
        retrieved_docs: list[dict],
        answer: str,
        judge_text: str,
    ) -> str:
        doc_lines = []
        for doc in retrieved_docs:
            snippet = _clean_text(doc["text"], min(self.max_doc_chars, 500))
            doc_lines.append(f"[{doc['docid']}] {snippet}")
        retrieved = "\n".join(doc_lines) if doc_lines else "[none]"
        return (
            f"USER:\n{task_description}\n\n"
            f"SYSTEM:\nRetrieved playbook context:\n{context or 'N/A'}\n\n"
            f"TOOL:\nSearch results:\n{retrieved}\n\n"
            f"ASSISTANT:\n{answer}\n\n"
            f"JUDGE:\n{judge_text}"
        )

    def _build_tool_trajectory(
        self,
        task_description: str,
        context: str,
        normalized_results: list[dict[str, Any]],
        judge_text: str,
    ) -> str:
        parts = [
            f"USER:\n{task_description}",
            f"SYSTEM:\nRetrieved playbook context:\n{context or 'N/A'}",
        ]
        for item in normalized_results:
            item_type = item.get("type")
            if item_type == "reasoning":
                parts.append(f"ASSISTANT_REASONING:\n{item.get('output')}")
            elif item_type == "tool_call":
                parts.append(
                    "TOOL:\n"
                    f"name={item.get('tool_name')}\n"
                    f"arguments={item.get('arguments')}\n"
                    f"output={item.get('output')}"
                )
            elif item_type == "output_text":
                parts.append(f"ASSISTANT:\n{item.get('output')}")
        parts.append(f"JUDGE:\n{judge_text}")
        return "\n\n".join(parts)

    def _write_run_artifact(
        self,
        query_id: str,
        status: str,
        result: list[dict[str, Any]],
        retrieved_docids: list[str],
        tool_call_counts: dict[str, int],
        model: str,
        usage: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.run_output_dir is None:
            return
        payload = {
            "query_id": str(query_id),
            "tool_call_counts": tool_call_counts,
            "status": status,
            "retrieved_docids": retrieved_docids,
            "result": result,
            "usage": usage or {},
            "metadata": {
                "model": model,
                "reasoning": (
                    {"effort": self.reasoning_effort, "summary": "detailed"}
                    if self.reasoning_effort
                    else None
                ),
                "output_dir": str(self.run_output_dir),
                "max_tokens": self.agent_max_output_tokens,
                "benchmark": "browsecomp",
                "run_id": self.run_id,
            },
        }
        out_path = self.run_output_dir / f"{query_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
