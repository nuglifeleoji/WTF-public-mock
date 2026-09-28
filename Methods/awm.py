"""AWMState — Agent Workflow Memory (AWM)-style structured memory.

Reference:
  Wang et al., "Agent Workflow Memory" (arXiv:2409.07429)
  https://github.com/zorazrw/agent-workflow-memory

Core idea (paper §3):
- A *workflow* is a named, natural-language sub-routine induced from a past
  successful trajectory.
- Workflows are represented by three human-readable fields: ``name`` (snake_case
  title), ``docstring`` (one sentence describing when the workflow applies),
  and a body of procedural ``steps`` whose non-fixed elements are written as
  ``{descriptive_placeholders}``.
- Induction runs online: after every N successful tasks (``induction_window``),
  the curator LLM is shown the existing workflows plus the most recent
  successful trajectories and is asked to propose *new* workflows covering
  patterns that the library does not yet capture. Failure trajectories are
  discarded at the filter step (same as ``induce_prompt.py::main`` in the
  reference repo).
- Retrieval embeds ``name\\ndocstring`` and returns the top-k most similar
  workflows to the new task. This mirrors ``mind2web/workflow/retrieve.py``
  (FAISS + OpenAI embeddings) but accepts an injected ``EmbeddingClient``.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from ._utils import approx_tokens, extract_json
from .base import AgentState
from .interfaces import EmbeddingClient, LLMClient


# ---------------------------------------------------------------------------
# Induction prompt — adapted from webarena/prompt/instruction.txt, generalised
# for any text trajectory (not just web actions). The language mirrors the
# reference verbatim where possible and extends it with a ``## Existing
# Workflows`` slot so the curator can avoid duplicates (a property enforced
# in the reference only implicitly via the "do not overlap" sentence).
# ---------------------------------------------------------------------------

INDUCTION_INSTRUCTION = """\
Given a list of successfully-completed agent tasks, your task is to extract the common workflows to solve these tasks.
Each given task contains a natural language instruction, and a trajectory of reasoning/actions that solved the task. You need to find the repetitive subset of actions across multiple tasks, and extract each of them out as a workflow.
Each workflow should be a commonly-reused sub-routine of the tasks. Do not generate similar or overlapping workflows. Each workflow should have at least two steps. Represent the non-fixed elements (input text, argument strings, ids, user-supplied values) with descriptive variable names as shown in the example.
Keep the values of invariant elements (e.g., fixed button labels, fixed API names, fixed file paths) as they will share and stay invariant across tasks.
Try to generate as many workflows that can cover all the tasks in the input list — but NEVER duplicate a workflow that is already listed under "Existing Workflows"; only propose workflows that are *new* or a strictly-better version of an existing one."""

INDUCTION_ONE_SHOT = """\
## Example
Successful Task 1: Find the date when I made my first purchase on this site.
Trajectory:
<think>Open the account area to reach order history.</think>
<action>click("My Account")</action>
<think>Open the orders list.</think>
<action>click("My Orders")</action>
<think>Jump to the last page to see the oldest orders.</think>
<action>click("last page")</action>
<think>The oldest row shows the first purchase date.</think>
<action>send_msg_to_user("The earliest purchase date is {first_purchase_date}.")</action>

Successful Task 2: Get the order number of my most recent cancelled order.
Trajectory:
<think>Open the account area.</think>
<action>click("My Account")</action>
<think>Open the orders list.</think>
<action>click("My Orders")</action>
<think>Filter by status and read the order number of the most recent cancelled row.</think>
<action>send_msg_to_user("Your most recent cancelled order is {order_id}.")</action>

## Summary Workflows
Workflow 1 — find_all_orders
Docstring: Given that the user is signed in, open the orders list so downstream steps can read order rows.
1. Click "My Account" to open the account page.
2. Click "My Orders" to list past orders.

Workflow 2 — read_order_field
Docstring: Given that the orders list is open, locate the row matching {match_criteria} and return its {field_name}.
1. Scroll or paginate the orders table until the row matching {match_criteria} is visible.
2. Read the {field_name} column of that row and respond with its value.
"""

INDUCTION_PROMPT_TEMPLATE = """\
{instruction}

{one_shot}

## Existing Workflows (do not duplicate; only propose NEW or strictly-better ones):
{existing_workflows}

## Successful Task Trajectories (new material to induce from):
{trajectories}

## Summary Workflows
Respond with a PURE JSON object (no markdown, no code fences) of the form:
{{
  "workflows": [
    {{
      "name": "<snake_case_name>",
      "docstring": "<one sentence describing when the workflow applies>",
      "steps": ["<step 1>", "<step 2>", "<step 3 — optional>"]
    }}
  ]
}}
If the existing library already covers all patterns in the new trajectories, return an empty list."""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_NAME_NORMALISE_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _normalise_workflow_name(raw: str) -> str:
    """Return a snake_case, alphanumeric workflow identifier."""
    name = (raw or "").strip().lstrip("#").strip()
    # Drop leading "Workflow 1 —", "Workflow 1:", etc. prefixes.
    name = re.sub(r"^[Ww]orkflow\s*\d*\s*[-:—]\s*", "", name).strip()
    snake = _NAME_NORMALISE_RE.sub("_", name).strip("_").lower()
    return snake or "workflow"


def _format_workflow_block(wf_id: str, name: str, docstring: str, steps: list[str]) -> str:
    """Render the stable ID, name, applicability description, and steps."""
    body_steps = "\n".join(f"  {i + 1}. {str(step).strip()}" for i, step in enumerate(steps))
    docline = (docstring or "").strip()
    return (
        f"[{wf_id}] {name}\n"
        f"  Docstring: {docline}\n"
        f"  Steps:\n"
        f"{body_steps}"
    )


# ---------------------------------------------------------------------------
# Workflow dataclass
# ---------------------------------------------------------------------------


@dataclass
class Workflow:
    name: str                    # canonical snake_case identifier / dict key
    docstring: str               # one-sentence "when to use" description
    steps: list[str]             # parameterised procedural steps
    wf_id: str                   # "wf-00001" style id, stable once assigned
    created_at: float = field(default_factory=time.time)
    source_task_ids: list[str] = field(default_factory=list)
    usage_count: int = 0
    revision: int = 1

    def render(self) -> str:
        return _format_workflow_block(self.wf_id, self.name, self.docstring, self.steps)


# ---------------------------------------------------------------------------
# AWMState
# ---------------------------------------------------------------------------


class AWMState(AgentState):
    """Agent Workflow Memory — online workflow induction from successes.

    Parameters
    ----------
    llm:
        Curator LLM used for workflow induction (paper §3.2).
    embedder:
        Embedding client used to pick the top-k most relevant workflows at
        retrieval time. Required for semantic retrieval; if ``None`` we fall
        back to a keyword overlap score over ``name`` + ``docstring``.
    max_tokens:
        Token budget for the rendered workflow library. Older workflows
        are dropped when the budget is exceeded.
    induction_window:
        Number of accumulated successful tasks that trigger a new induction
        call. Matches the online loop in ``webarena/pipeline.py`` (run ->
        autoeval -> induce) — we induce every ``induction_window`` successes
        rather than every step to keep LLM costs bounded.
    induction_history_size:
        Sliding window of recent successes kept as material for each
        induction call.
    max_trajectory_chars:
        Per-trajectory truncation budget before we hand the buffer to the
        curator. Keeps induction prompts bounded across long trajectories.
    max_workflows:
        Hard cap on the library size; the oldest workflows are evicted
        first once the cap is exceeded.
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: EmbeddingClient | None = None,
        max_tokens: int = 50_000,
        induction_window: int = 3,
        induction_history_size: int = 10,
        max_trajectory_chars: int = 4000,
        max_workflows: int = 64,
    ):
        self._llm = llm
        self._embedder = embedder
        self._max_tokens = max_tokens
        self._induction_window = max(1, int(induction_window))
        self._induction_history_size = max(1, int(induction_history_size))
        self._max_trajectory_chars = max(256, int(max_trajectory_chars))
        self._max_workflows = max(1, int(max_workflows))

        self._workflows: dict[str, Workflow] = {}
        self._success_buffer: deque = deque(maxlen=self._induction_history_size)
        self._pending_successes: int = 0
        self._next_id: int = 1

        self._update_count: int = 0
        self._successful_updates: int = 0
        self._induction_calls: int = 0
        self._induction_failures: int = 0
        self._total_added: int = 0
        self._total_revised: int = 0

    # ------------------------------------------------------------------
    # AgentState interface
    # ------------------------------------------------------------------

    def retrieve(self, task_description: str, k: int = 5) -> str:
        if not self._workflows:
            return ""

        names = list(self._workflows.keys())
        k = max(1, min(int(k), len(names)))

        texts = [
            f"{self._workflows[n].name}\n{self._workflows[n].docstring}"
            for n in names
        ]
        if self._embedder is None:
            idxs = self._keyword_topk(task_description, texts, k)
        else:
            try:
                idxs = list(self._embedder.top_k_indices(task_description, texts, k))
            except Exception:
                idxs = self._keyword_topk(task_description, texts, k)

        blocks = []
        for i in idxs:
            if i < 0 or i >= len(names):
                continue
            wf = self._workflows[names[i]]
            wf.usage_count += 1
            blocks.append(wf.render())

        if not blocks:
            return ""

        header = "## WORKFLOWS (learned from past successful tasks)"
        sections = [header, "\n\n".join(blocks)]
        return "\n\n".join(sections)

    def update(self, reflection: dict) -> None:
        self._update_count += 1

        outcome = (reflection.get("outcome") or "").lower()
        # AWM learns from successes only. Failures are filtered out at the
        # induction gate — same as ``webarena/induce_prompt.py::main`` which
        # keeps only trajectories with a positive reward.
        if outcome != "success":
            return

        self._successful_updates += 1

        task_desc = reflection.get("task_description", "") or ""
        trajectory = reflection.get("trajectory", "") or ""
        if not trajectory.strip():
            # Nothing to induce from — still count as a success so
            # bookkeeping is consistent, but skip buffering.
            return

        metadata = reflection.get("metadata") or {}
        task_id = (
            metadata.get("task_id")
            or reflection.get("task_id")
            or f"success-{self._successful_updates}"
        )

        self._success_buffer.append(
            {
                "task_id": str(task_id),
                "task_description": task_desc,
                "trajectory": self._truncate_middle(
                    trajectory, self._max_trajectory_chars
                ),
            }
        )
        self._pending_successes += 1

        if self._pending_successes >= self._induction_window:
            self._pending_successes = 0
            self._induce_workflows()

        self._prune_to_budget()

    def serialize(self) -> str:
        return json.dumps(
            {
                "type": "awm",
                "workflows": {k: asdict(v) for k, v in self._workflows.items()},
                "next_id": self._next_id,
                "update_count": self._update_count,
                "successful_updates": self._successful_updates,
                "induction_calls": self._induction_calls,
                "induction_failures": self._induction_failures,
                "total_added": self._total_added,
                "total_revised": self._total_revised,
            },
            indent=2,
        )

    def load_serialized(self, payload: str) -> None:
        data = json.loads(payload)
        if data.get("type") != "awm":
            raise ValueError(f"Expected awm snapshot, got: {data.get('type')!r}")

        self._workflows = {}
        for key, raw in (data.get("workflows") or {}).items():
            if not isinstance(raw, dict):
                continue
            self._workflows[key] = Workflow(
                name=raw.get("name", key),
                docstring=raw.get("docstring", ""),
                steps=list(raw.get("steps") or []),
                wf_id=raw.get("wf_id", f"wf-{len(self._workflows) + 1:05d}"),
                created_at=float(raw.get("created_at", time.time())),
                source_task_ids=list(raw.get("source_task_ids") or []),
                usage_count=int(raw.get("usage_count", 0)),
                revision=int(raw.get("revision", 1)),
            )
        self._next_id = int(data.get("next_id", len(self._workflows) + 1))
        self._update_count = int(data.get("update_count", 0))
        self._successful_updates = int(data.get("successful_updates", 0))
        self._induction_calls = int(data.get("induction_calls", 0))
        self._induction_failures = int(data.get("induction_failures", 0))
        self._total_added = int(data.get("total_added", 0))
        self._total_revised = int(data.get("total_revised", 0))
        self._pending_successes = 0
        self._success_buffer.clear()

    def stats(self) -> dict:
        rendered = "\n\n".join(w.render() for w in self._workflows.values())
        return {
            "type": "awm",
            "num_workflows": len(self._workflows),
            "approx_tokens": approx_tokens(rendered),
            "update_count": self._update_count,
            "successful_updates": self._successful_updates,
            "induction_calls": self._induction_calls,
            "induction_failures": self._induction_failures,
            "total_added": self._total_added,
            "total_revised": self._total_revised,
            "workflow_names": list(self._workflows.keys()),
        }

    # ------------------------------------------------------------------
    # Induction
    # ------------------------------------------------------------------

    def _induce_workflows(self) -> None:
        if not self._success_buffer:
            return

        self._induction_calls += 1

        existing_block = (
            "\n\n".join(w.render() for w in self._workflows.values())
            if self._workflows
            else "(none yet)"
        )
        trajs_block = "\n\n".join(
            f"### Successful Task {i + 1}: {ex['task_description']}\n{ex['trajectory']}"
            for i, ex in enumerate(self._success_buffer)
        )
        prompt = INDUCTION_PROMPT_TEMPLATE.format(
            instruction=INDUCTION_INSTRUCTION,
            one_shot=INDUCTION_ONE_SHOT,
            existing_workflows=existing_block,
            trajectories=trajs_block,
        )

        try:
            raw = self._llm.generate(prompt, max_tokens=2048)
        except Exception as exc:  # pragma: no cover - network-ish error path
            print(f"  [AWM] induction LLM call failed: {type(exc).__name__}: {exc}")
            self._induction_failures += 1
            return

        data = extract_json(raw)
        if not isinstance(data, dict) or not isinstance(data.get("workflows"), list):
            print("  [AWM] induction: could not parse workflows JSON")
            self._induction_failures += 1
            return

        task_ids = [ex["task_id"] for ex in self._success_buffer]
        added = 0
        revised = 0
        for entry in data["workflows"]:
            if not isinstance(entry, dict):
                continue
            name = _normalise_workflow_name(str(entry.get("name", "")))
            if not name:
                continue
            docstring = _WS_RE.sub(" ", str(entry.get("docstring", ""))).strip()
            steps_raw = entry.get("steps")
            if not isinstance(steps_raw, list):
                continue
            steps = [str(s).strip() for s in steps_raw if str(s).strip()]
            if len(steps) < 2:
                # Paper constraint: "Each workflow should have at least two steps."
                continue

            existing = self._workflows.get(name)
            if existing is not None:
                existing.docstring = docstring or existing.docstring
                existing.steps = steps
                existing.source_task_ids = list(
                    dict.fromkeys(existing.source_task_ids + task_ids)
                )
                existing.revision += 1
                revised += 1
                continue

            wf_id = f"wf-{self._next_id:05d}"
            self._next_id += 1
            self._workflows[name] = Workflow(
                name=name,
                docstring=docstring,
                steps=steps,
                wf_id=wf_id,
                source_task_ids=list(task_ids),
            )
            added += 1

        self._total_added += added
        self._total_revised += revised
        print(
            f"  [AWM] induction: +{added} new, {revised} revised "
            f"(library now = {len(self._workflows)} workflows)"
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _keyword_topk(self, query: str, candidates: list[str], k: int) -> list[int]:
        """Simple token-overlap scorer used when no embedder is available."""
        tokens = set(re.findall(r"\w+", (query or "").lower()))
        if not tokens:
            return list(range(min(k, len(candidates))))
        scored: list[tuple[int, int]] = []
        for i, text in enumerate(candidates):
            cand_tokens = set(re.findall(r"\w+", text.lower()))
            scored.append((len(tokens & cand_tokens), i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [i for _, i in scored[:k]]

    @staticmethod
    def _truncate_middle(text: str, max_chars: int) -> str:
        if not text or len(text) <= max_chars:
            return text
        if max_chars <= 120:
            return text[:max_chars]
        omitted = len(text) - max_chars
        marker = f"\n...[trajectory truncated: omitted {omitted} chars]...\n"
        keep = max_chars - len(marker)
        head = int(keep * 0.6)
        tail = keep - head
        return text[:head] + marker + text[-tail:]

    def _prune_to_budget(self) -> None:
        # 1. enforce workflow count cap
        if len(self._workflows) > self._max_workflows:
            by_age = sorted(
                self._workflows.items(),
                key=lambda kv: (kv[1].created_at, kv[1].wf_id),
            )
            while len(self._workflows) > self._max_workflows:
                name, _ = by_age.pop(0)
                self._workflows.pop(name, None)

        # 2. enforce token budget
        rendered = "\n\n".join(w.render() for w in self._workflows.values())
        if approx_tokens(rendered) <= self._max_tokens:
            return
        by_age = sorted(
            self._workflows.items(),
            key=lambda kv: (kv[1].created_at, kv[1].wf_id),
        )
        while by_age:
            name, wf = by_age.pop(0)
            self._workflows.pop(name, None)
            rendered = "\n\n".join(w.render() for w in self._workflows.values())
            if approx_tokens(rendered) <= self._max_tokens:
                return
