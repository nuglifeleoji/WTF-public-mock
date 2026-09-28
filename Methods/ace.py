"""ACE-style playbook memory, adapted from ace-agent/ace-appworld.

The curator adds numbered rules to a structured playbook after each task.
Retrieval returns the complete playbook; oldest bullets are pruned at the
storage token budget. The caller supplies the completed task's reflection.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .base import AgentState
from ._utils import approx_tokens, extract_json
from .interfaces import EmbeddingClient, LLMClient

_SECTIONS: list[tuple[str, str]] = [
    ("strategies_and_hard_rules",           "## STRATEGIES AND HARD RULES"),
    ("apis_to_use_for_specific_information","## APIS TO USE FOR SPECIFIC INFORMATION"),
    ("useful_code_snippets_and_templates",  "## USEFUL CODE SNIPPETS AND TEMPLATES"),
    ("common_mistakes_and_correct_strategies","## COMMON MISTAKES AND CORRECT STRATEGIES"),
    ("problem_solving_heuristics_and_workflows","## PROBLEM SOLVING HEURISTICS AND WORKFLOWS"),
    ("verification_checklist",              "## VERIFICATION CHECKLIST"),
    ("troubleshooting_and_pitfalls",        "## TROUBLESHOOTING AND PITFALLS"),
    ("others",                              "## OTHERS"),
]

# Mapping: normalised slug → markdown header
_SLUG_TO_HEADER: dict[str, str] = {slug: header for slug, header in _SECTIONS}

# Mapping: normalised slug → bullet-ID prefix (matches reference)
_SLUG_TO_PREFIX: dict[str, str] = {
    "strategies_and_hard_rules":              "shr",
    "apis_to_use_for_specific_information":   "api",
    "useful_code_snippets_and_templates":     "uc",
    "common_mistakes_and_correct_strategies": "cm",
    "problem_solving_heuristics_and_workflows":"psw",
    "verification_checklist":                 "vc",
    "troubleshooting_and_pitfalls":           "tp",
    "others":                                 "others",
}

_INITIAL_PLAYBOOK = "\n\n".join(
    f"{header}\n" for _, header in _SECTIONS
)

CURATOR_PROMPT = """\
You are a master curator of knowledge. Your job is to identify what new insights \
should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you create will be used to help answer similar questions.
- The reflection is generated using ground truth answers that will NOT be available \
when the playbook is being used. So you need to come up with content that can aid \
the playbook user to create predictions that likely align with ground truth.

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy — if similar advice already exists, only add content that is a perfect complement
- Do NOT regenerate the entire playbook — only provide the additions needed
- Focus on quality over quantity — a focused, well-organised playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation, if there is no new content to add, return an empty list for operations
- Be concise and specific — each addition should be actionable
- For coding tasks, explicitly curate the output format/schema of APIs used when unclear \
(e.g., `apis.blah.show_contents()` returns a list of content_ids (strings), not content objects)

**Task Context (the actual task instruction):**
{question_context}

**Current Playbook:**
{current_playbook}

**Agent Conversation / Generated Attempt (latest attempt, with reasoning):**
{trajectory}

**Current Reflections (principles and strategies that helped or hindered the current task):**
{guidebook}

**Available sections:**
- strategies_and_hard_rules
- apis_to_use_for_specific_information
- useful_code_snippets_and_templates
- common_mistakes_and_correct_strategies
- problem_solving_heuristics_and_workflows
- verification_checklist
- troubleshooting_and_pitfalls
- others

**RESPONSE FORMAT — Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process]",
  "operations": [
    {{
      "type": "ADD",
      "section": "verification_checklist",
      "content": "[New checklist item or API schema clarification...]"
    }}
  ]
}}"""


_BULLET_RE = re.compile(r"^\[([\w]+-(\d{5}))\]")


def _normalize_section(raw: str) -> str:
    """Normalise a curator-returned section name to a slug key."""
    return (
        raw.strip()
        .lower()
        .replace(" ", "_")
        .replace("&", "and")
        .rstrip(":")
    )

def _next_global_id(playbook_text: str) -> int:
    """Return one above the highest numeric ID in the playbook."""
    nums = [int(m.group(2)) for m in (_BULLET_RE.match(l.strip()) for l in playbook_text.splitlines()) if m]
    return max(nums, default=0) + 1

def _apply_operations(playbook_text: str, operations: list[dict], next_id: int) -> tuple[str, int]:
    """Apply curator ADD operations. Returns (updated_playbook_text, next_id)."""
    lines = playbook_text.split("\n")

    # Map slug → line index of its header
    header_to_lineidx: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        for slug, header in _SECTIONS:
            if stripped == header:
                header_to_lineidx[slug] = i

    # Build per-section bullet lists (preserves operation order within a section)
    slug_to_bullets: dict[str, list[str]] = {}
    for op in operations:
        if op.get("type") != "ADD":
            continue
        slug = _normalize_section(op.get("section", "others"))
        if slug not in _SLUG_TO_HEADER:
            print(f"  [Curator] Unknown section '{op.get('section')}' → routing to 'others'")
            slug = "others"
        prefix = _SLUG_TO_PREFIX[slug]
        bullet_id = f"{prefix}-{next_id:05d}"
        next_id += 1
        content = op.get("content", "").strip()
        # Strip any bullet-ID prefix the curator may have echoed back from the playbook
        # (e.g. "[vc-00002] Some insight" → "Some insight").
        content = re.sub(r"^\[[a-z]+-\d{5}\]\s*", "", content)
        slug_to_bullets.setdefault(slug, []).append(f"[{bullet_id}] {content}")
        print(f"  [Curator] ADD [{bullet_id}] to '{slug}'")

    if not slug_to_bullets:
        return playbook_text, next_id

    # Insert all bullets for each section in one pass; shift header indices in batch
    for slug, bullets in slug_to_bullets.items():
        if slug not in header_to_lineidx:
            lines.extend(["", _SLUG_TO_HEADER[slug]] + bullets)
            continue
        insert_at = header_to_lineidx[slug] + 1
        lines[insert_at:insert_at] = bullets
        n = len(bullets)
        for other_slug in header_to_lineidx:
            if header_to_lineidx[other_slug] >= insert_at:
                header_to_lineidx[other_slug] += n

    return "\n".join(lines), next_id


class PlaybookState(AgentState):
    """ACE's sectioned playbook with curator-driven ADD operations."""

    def __init__(
        self,
        llm: LLMClient,
        embedder: EmbeddingClient | None = None,
        max_tokens: int = 50_000,
    ):
        # Accepted for a uniform constructor; full-playbook retrieval needs no embeddings.
        self._llm = llm
        self._max_tokens = max_tokens
        self._playbook = _INITIAL_PLAYBOOK
        self._next_id = 1
        self._update_count = 0
        self._total_bullets = 0

    def retrieve(self, task_description: str, k: int = 5) -> str:
        """Return the full playbook; task_description and k do not filter it."""
        return self._playbook.strip()

    def load_serialized(self, payload: str) -> None:
        data = json.loads(payload)
        if data.get("type") != "playbook":
            raise ValueError(f"Expected playbook snapshot, got: {data.get('type')!r}")
        self._playbook = str(data.get("playbook", _INITIAL_PLAYBOOK))
        self._update_count = int(data.get("update_count", 0))
        self._total_bullets = int(data.get("total_bullets", 0))
        self._next_id = int(data.get("next_id", _next_global_id(self._playbook)))

    def update(self, reflection: dict) -> None:
        self._update_count += 1

        reflection_json = reflection.get("reflection_json", {})
        # Compose a guidebook string from the structured reflection fields
        guidebook_parts = []
        for field in ("error_identification", "root_cause_analysis", "correct_approach", "key_insight"):
            val = reflection_json.get(field, "")
            if val:
                guidebook_parts.append(f"{field.replace('_', ' ').title()}: {val}")
        guidebook = "\n".join(guidebook_parts) or reflection.get("reflection_text", "")

        prompt = CURATOR_PROMPT.format(
            question_context=reflection.get("task_description", ""),
            current_playbook=self._playbook,
            trajectory=reflection.get("trajectory", ""),
            guidebook=guidebook,
        )
        raw = self._llm.generate(prompt, max_tokens=2048)
        curator_json = extract_json(raw)

        if curator_json is None:
            print(f"  [Curator] Failed to parse JSON — skipping update")
            return

        if "operations" not in curator_json or not isinstance(curator_json["operations"], list):
            print(f"  [Curator] Missing 'operations' field — skipping update")
            return

        _allowed_sections = set(_SLUG_TO_HEADER.keys())
        filtered_ops = []
        for op in curator_json["operations"]:
            if not isinstance(op, dict) or op.get("type") != "ADD":
                continue
            slug = _normalize_section(op.get("section", ""))
            if slug not in _allowed_sections:
                op = dict(op, section="others")
            filtered_ops.append(op)

        if not filtered_ops:
            return

        self._playbook, self._next_id = _apply_operations(
            self._playbook, filtered_ops, self._next_id
        )
        self._total_bullets += len(filtered_ops)
        self._prune_to_budget()


    def serialize(self) -> str:
        return json.dumps(
            {
                "type": "playbook",
                "merge_strategy": "llm",
                "playbook": self._playbook,
                "update_count": self._update_count,
                "total_bullets": self._total_bullets,
                "next_id": self._next_id,
            },
            indent=2,
        )

    def stats(self) -> dict:
        return {
            "type": "playbook",
            "merge_strategy": "llm",
            "total_bullets": self._total_bullets,
            "update_count": self._update_count,
            "approx_tokens": approx_tokens(self._playbook),
        }

    def _prune_to_budget(self) -> None:
        """Remove oldest bullets (lowest IDs) when playbook exceeds token budget.

        A bullet spans its ID line plus all following lines up to (but not including)
        the next bullet ID or section header — multi-line bodies (code blocks,
        numbered lists) are part of the bullet and must be removed together with
        the ID line.
        """
        if approx_tokens(self._playbook) <= self._max_tokens:
            return

        lines = self._playbook.split("\n")
        # Identify bullet ranges: (id_num, start_idx, end_idx_exclusive).
        # A range ends at the next bullet ID, the next section header, or EOF.
        bullet_ranges: list[tuple[int, int, int]] = []
        cur_id: Optional[int] = None
        cur_start: Optional[int] = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            m = _BULLET_RE.match(stripped)
            is_section = stripped.startswith("## ")
            if m or is_section:
                if cur_id is not None and cur_start is not None:
                    bullet_ranges.append((cur_id, cur_start, i))
                    cur_id = None
                    cur_start = None
                if m:
                    cur_id = int(m.group(2))
                    cur_start = i
        if cur_id is not None and cur_start is not None:
            bullet_ranges.append((cur_id, cur_start, len(lines)))

        bullet_ranges.sort()  # oldest first by numeric id

        total = approx_tokens(self._playbook)
        removed: set[int] = set()
        for _id_num, start, end in bullet_ranges:
            removed_chars = sum(len(lines[k]) + 1 for k in range(start, end))
            for k in range(start, end):
                removed.add(k)
            total -= removed_chars // 4
            if total <= self._max_tokens:
                break

        cleaned: list[str] = []
        prev_blank = False
        for i, line in enumerate(lines):
            if i in removed:
                continue
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned.append(line)
            prev_blank = is_blank
        self._playbook = "\n".join(cleaned)

