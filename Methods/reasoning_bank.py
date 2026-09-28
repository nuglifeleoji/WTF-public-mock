"""ReasoningBank memory from successful and failed agent trajectories.

Adapted from Ouyang et al., "ReasoningBank: Scaling Agent Self-Evolving
with Reasoning Memory" (arXiv:2509.25140), and the reference implementation:
https://github.com/google-research/reasoning-bank.

Each completed task contributes structured title/description/content items.
Retrieval ranks stored task queries by embedding similarity and renders the
items from the top-k entries. Oldest entries are dropped when the memory
budget is exceeded.

This adaptation uses the benchmark outcome as its correctness signal instead
of an LLM judge. Pending outcomes are skipped; other non-success outcomes
use the failure extraction prompt. Optional reflection text supplements the
query and trajectory supplied to the extractor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .base import AgentState
from ._utils import approx_tokens
from .interfaces import EmbeddingClient, LLMClient


# ---------------------------------------------------------------------------
# Prompts — lifted verbatim from google-research/reasoning-bank
# (WebArena/prompts/memory_instruction.py), with only the `{max_items}`
# placeholder added so callers can tune the extraction cap.
# ---------------------------------------------------------------------------

SUCCESSFUL_EXTRACTION_SYSTEM = """\
You are an expert agent memory curator. You will be given a user query, the corresponding trajectory that represents **how an agent successfully accomplished the task**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most {max_items}* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing similar tasks in the future>
```
"""

FAILED_EXTRACTION_SYSTEM = """\
You are an expert agent memory curator. You will be given a user query, the corresponding trajectory that represents **how an agent attempted to resolve the task but failed**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most {max_items}* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to avoid such failures and successfully accomplishing similar tasks in the future>
```
"""

# Retrieval preamble is lifted verbatim from Appendix A.2 of the paper.
RETRIEVAL_PREAMBLE = (
    "Below are some memory items that I accumulated from past interaction "
    "from the environment that may be helpful to solve the task. You can use "
    "it when you feel it's relevant. In each step, please first explicitly "
    "discuss if you want to use each memory item or not, and then take action."
)

# Hard caps for the extractor's query / trajectory inputs so the prompt stays
# bounded regardless of how long the trajectory is.
_MAX_QUERY_CHARS = 4_000
_MAX_TRAJECTORY_CHARS = 60_000


def _truncate_middle(text: str, max_chars: int, *, label: str) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    marker = f"\n\n...[{label} truncated: omitted {omitted} chars]...\n\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return text[:max_chars]
    head = int(keep * 0.6)
    tail = keep - head
    return text[:head] + marker + text[-tail:]


# Parser for the markdown memory-item format the extractor is asked to emit.
# We accept a couple of small variations ("## Title: X", "## Title X", multi-
# line content) that LLMs produce in practice.
_ITEM_BLOCK_RE = re.compile(
    r"#\s*Memory\s*Item[^\n]*\n(?P<body>.*?)(?=\n#\s*Memory\s*Item|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"^##\s*(?P<key>Title|Description|Content)\s*:?\s*(?P<val>.*?)(?=\n##\s|\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)


def _parse_memory_items(raw: str) -> list[dict]:
    """Best-effort parse of the extractor output into list of item dicts."""
    if not raw:
        return []

    text = raw.strip()
    # Strip ```...``` fences so blocks inside them are parsed the same way.
    fence = re.match(r"```[a-zA-Z]*\s*(.*?)\s*```\s*\Z", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Also strip isolated fence lines mid-text ("```", "```markdown") since the
    # paper's format specimen itself wraps in fences.
    text = re.sub(r"^```[a-zA-Z]*\s*$", "", text, flags=re.MULTILINE).strip()

    items: list[dict] = []
    for block in _ITEM_BLOCK_RE.finditer(text):
        body = block.group("body")
        fields = {"title": "", "description": "", "content": ""}
        for sec in _SECTION_RE.finditer(body):
            key = sec.group("key").strip().lower()
            val = sec.group("val").strip()
            if key in fields:
                fields[key] = " ".join(val.split())
        # An item needs at least a title or content to be worth keeping.
        if fields["title"] or fields["content"]:
            items.append(fields)

    # Fallback: if the model ignored the Memory Item wrapper and emitted a
    # single block of ## Title / ## Description / ## Content, recover that too.
    if not items and "## Title" in text.title():
        fields = {"title": "", "description": "", "content": ""}
        for sec in _SECTION_RE.finditer(text):
            key = sec.group("key").strip().lower()
            val = sec.group("val").strip()
            if key in fields:
                fields[key] = " ".join(val.split())
        if fields["title"] or fields["content"]:
            items.append(fields)

    return items


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryItem:
    id: str                  # e.g. "rb-00001"
    title: str
    description: str
    content: str


@dataclass
class ReasoningBankEntry:
    """One stored experience: a query, its outcome, and extracted memories."""
    query: str
    outcome: str             # "success" | "failure" (anything non-success)
    memory_items: list[MemoryItem] = field(default_factory=list)
    task_id: Optional[str] = None
    embedding: Optional[list[float]] = None  # serialised representation of task-query vector


# ---------------------------------------------------------------------------
# ReasoningBankState
# ---------------------------------------------------------------------------


class ReasoningBankState(AgentState):
    """
    Memory module that stores structured reasoning items induced from an
    agent's own past successes and failures.  See module docstring for the
    end-to-end recipe.
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: EmbeddingClient,
        max_tokens: int = 50_000,
        max_items_per_trajectory: int = 3,
        extractor_max_tokens: int = 2048,
    ):
        if embedder is None:
            raise ValueError(
                "ReasoningBankState requires an EmbeddingClient for retrieval."
            )
        self._llm = llm
        self._embedder = embedder
        self._max_tokens = max_tokens
        self._max_items_per_trajectory = max_items_per_trajectory
        self._extractor_max_tokens = extractor_max_tokens
        self._entries: list[ReasoningBankEntry] = []
        self._next_id: int = 1
        self._update_count: int = 0
        self._parse_failures: int = 0

    # ------------------------------------------------------------------
    # AgentState interface
    # ------------------------------------------------------------------

    def retrieve(self, task_description: str, k: int = 1) -> str:
        if not self._entries:
            return ""

        # Select top-k entries by cosine similarity of the task-query embedding
        # against stored entry embeddings.  Fresh on-the-fly compute so the
        # user can swap embedders between runs without reindexing.
        queries = [e.query for e in self._entries]
        indices = self._embedder.top_k_indices(task_description, queries, k)
        if not indices:
            return ""

        retrieved_items = [
            item
            for idx in indices
            if 0 <= idx < len(self._entries)
            for item in self._entries[idx].memory_items
        ]
        if not retrieved_items:
            return ""

        lines: list[str] = [RETRIEVAL_PREAMBLE, ""]
        for item in retrieved_items:
            lines.append(f"[{item.id}] Title: {item.title}")
            if item.description:
                lines.append(f"Description: {item.description}")
            if item.content:
                lines.append(f"Content: {item.content}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def update(self, reflection: dict) -> None:
        self._update_count += 1

        outcome = (reflection.get("outcome") or "").lower()
        if outcome == "pending":
            # Benchmark couldn't decide — don't pollute memory with noise.
            return

        query = (reflection.get("task_description") or "").strip()
        trajectory = (reflection.get("trajectory") or "").strip()
        if not query or not trajectory:
            return

        # 1) Extract candidate memory items with the outcome-specific prompt.
        raw_items = self._extract_items(query, trajectory, outcome, reflection)
        if not raw_items:
            return

        # 2) Turn parsed dicts into MemoryItem objects with fresh bullet IDs.
        items: list[MemoryItem] = []
        for raw in raw_items[: self._max_items_per_trajectory]:
            bullet_id = f"rb-{self._next_id:05d}"
            self._next_id += 1
            items.append(
                MemoryItem(
                    id=bullet_id,
                    title=raw.get("title", "").strip(),
                    description=raw.get("description", "").strip(),
                    content=raw.get("content", "").strip(),
                )
            )
            print(f"  [ReasoningBank] ADD [{bullet_id}] {items[-1].title[:80]}")

        if not items:
            return

        # 3) Persist the task-query vector as part of the snapshot.
        try:
            embedding_vec = self._embedder.embed([query])[0]
            embedding = embedding_vec.astype(np.float32).tolist()
        except Exception as exc:
            print(f"  [ReasoningBank] embedding failed ({type(exc).__name__}); storing without cached vector")
            embedding = None

        metadata = reflection.get("metadata") or {}
        self._entries.append(
            ReasoningBankEntry(
                task_id=str(metadata.get("task_id") or metadata.get("instance_id") or ""),
                query=query,
                outcome="success" if outcome == "success" else "failure",
                memory_items=items,
                embedding=embedding,
            )
        )
        self._prune_to_budget()

    def serialize(self) -> str:
        return json.dumps(
            {
                "type": "reasoning_bank",
                "entries": [
                    {
                        "task_id": e.task_id,
                        "query": e.query,
                        "outcome": e.outcome,
                        "memory_items": [asdict(it) for it in e.memory_items],
                        "embedding": e.embedding,
                    }
                    for e in self._entries
                ],
                "next_id": self._next_id,
                "update_count": self._update_count,
                "parse_failures": self._parse_failures,
            },
            indent=2,
        )

    def stats(self) -> dict:
        total_items = sum(len(e.memory_items) for e in self._entries)
        text_chars = sum(
            len(it.title) + len(it.description) + len(it.content)
            for e in self._entries
            for it in e.memory_items
        )
        return {
            "type": "reasoning_bank",
            "num_entries": len(self._entries),
            "num_items": total_items,
            "update_count": self._update_count,
            "parse_failures": self._parse_failures,
            "approx_tokens": text_chars // 4,
            "success_entries": sum(1 for e in self._entries if e.outcome == "success"),
            "failure_entries": sum(1 for e in self._entries if e.outcome != "success"),
        }

    def load_serialized(self, payload: str) -> None:
        data = json.loads(payload)
        if data.get("type") != "reasoning_bank":
            raise ValueError(
                f"Expected reasoning_bank snapshot, got: {data.get('type')!r}"
            )

        self._entries = []
        for e in data.get("entries", []):
            items = [
                MemoryItem(
                    id=str(it.get("id", "")),
                    title=str(it.get("title", "")),
                    description=str(it.get("description", "")),
                    content=str(it.get("content", "")),
                )
                for it in e.get("memory_items", [])
            ]
            self._entries.append(
                ReasoningBankEntry(
                    task_id=e.get("task_id"),
                    query=str(e.get("query", "")),
                    outcome=str(e.get("outcome", "failure")),
                    memory_items=items,
                    embedding=e.get("embedding"),
                )
            )
        self._next_id = int(data.get("next_id", self._next_id))
        self._update_count = int(data.get("update_count", 0))
        self._parse_failures = int(data.get("parse_failures", 0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_items(
        self,
        query: str,
        trajectory: str,
        outcome: str,
        reflection: dict,
    ) -> list[dict]:
        """Run the extractor LLM and parse its markdown output."""
        system_template = (
            SUCCESSFUL_EXTRACTION_SYSTEM if outcome == "success" else FAILED_EXTRACTION_SYSTEM
        )
        system_prompt = system_template.format(max_items=self._max_items_per_trajectory)

        q = _truncate_middle(query, _MAX_QUERY_CHARS, label="query")
        traj = _truncate_middle(trajectory, _MAX_TRAJECTORY_CHARS, label="trajectory")

        # Include optional analysis explaining the observed outcome.
        rj = reflection.get("reflection_json") or {}
        why_parts: list[str] = []
        for field_name in ("error_identification", "root_cause_analysis", "correct_approach", "key_insight"):
            val = rj.get(field_name, "")
            if val:
                why_parts.append(f"{field_name.replace('_', ' ').title()}: {val}")
        why_text = "\n".join(why_parts) or (reflection.get("reflection_text", "") or "")
        why_text = why_text.strip()

        label = "succeeded" if outcome == "success" else "failed"
        user_prompt_parts = [f"Query: {q}", "", f"Trajectory:\n{traj}"]
        if why_text:
            user_prompt_parts.extend(
                ["", f"The task {label} because: {why_text}"]
            )
        user_prompt = "\n".join(user_prompt_parts)

        try:
            raw = self._llm.generate(
                user_prompt,
                max_tokens=self._extractor_max_tokens,
                system=system_prompt,
            )
        except Exception as exc:
            print(
                f"  [ReasoningBank] extractor call failed "
                f"({type(exc).__name__}: {exc}) — skipping this update"
            )
            return []

        items = _parse_memory_items(raw)
        if not items:
            self._parse_failures += 1
            print("  [ReasoningBank] extractor output did not match the memory-item format — skipping")
        return items

    def _prune_to_budget(self) -> None:
        """Drop oldest entries until we fit inside max_tokens."""
        def total_tokens() -> int:
            return sum(
                approx_tokens(it.title) + approx_tokens(it.description) + approx_tokens(it.content)
                for e in self._entries
                for it in e.memory_items
            )

        while total_tokens() > self._max_tokens and self._entries:
            dropped = self._entries.pop(0)
            for it in dropped.memory_items:
                print(f"  [ReasoningBank] PRUNE [{it.id}] {it.title[:60]}")
