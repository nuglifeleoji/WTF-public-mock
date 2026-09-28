"""HermesState - Hermes Agent style self-evolving skill memory.

Reference implementation:
  https://github.com/NousResearch/hermes-agent

What is preserved here:
- Skills are procedural memory, represented as agentskills-compatible
  ``SKILL.md`` documents with YAML frontmatter.
- Retrieval follows Hermes' progressive-disclosure idea: expose a compact
  skill index, then load the most relevant full skill bodies for the task.
- Updates are expressed as ``skill_manage``-style actions: create, edit,
  patch, delete, write_file, and remove_file.
- The update prompt pushes the curator toward class-level umbrella skills,
  patching existing skills first, and using references/templates/scripts as
  support files rather than creating one narrow skill per benchmark task.

Skills and supporting files are kept in memory and serialized as JSON. The
default category and neutral update-prompt wording are retained for parity.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Optional

import yaml

from ._utils import approx_tokens, extract_json
from .base import AgentState
from .interfaces import EmbeddingClient, LLMClient


HERMES_SKILL_UPDATE_SYSTEM = """\
You are Hermes' skill self-improvement reviewer. You maintain procedural memory
as reusable SKILL.md documents, not as one-off notes.

Review the completed benchmark task and decide whether the skill library should
change. Prefer durable, class-level knowledge:
1. Patch a currently relevant skill if it already covers the lesson.
2. Patch an existing umbrella skill if one fits.
3. Add a support file under references/, templates/, or scripts/ when the
   detail is too specific for the main SKILL.md but useful for future tasks.
4. Create a new class-level umbrella skill only when no existing skill covers
   the recurring task type.

Create when a complex task succeeded, a tricky error was overcome, a benchmark
failure reveals a durable pitfall, or a non-trivial workflow emerged. Patch when
a skill is stale, incomplete, or missing a pitfall. Do not preserve transient
environment failures, one-off task narratives, PR numbers, instance IDs, commit
hashes, exact benchmark answers, or facts that will be stale soon.

Use these skill_manage-style actions:
- create: requires name and full SKILL.md content with frontmatter.
- edit: replace an existing SKILL.md with full content.
- patch: old_string/new_string replacement; use replace_all only when safe.
- write_file: add a supporting file under references/, templates/, scripts/,
  or assets/.
- remove_file: remove a supporting file.
- delete: remove a skill only when it is clearly obsolete or absorbed.

Return a pure JSON object with:
{
  "actions": [
    {
      "action": "create|edit|patch|delete|write_file|remove_file",
      "name": "lowercase-skill-name",
      "content": "full SKILL.md for create/edit",
      "old_string": "for patch",
      "new_string": "for patch",
      "replace_all": false,
      "file_path": "references/example.md",
      "file_content": "for write_file",
      "absorbed_into": "optional target skill"
    }
  ]
}
Return {"actions": []} when there is nothing durable to save.
"""


HERMES_UPDATE_PROMPT_TEMPLATE = """\
{system}

## Existing Hermes Skill Index
{skill_index}

## Existing Loaded Skill Bodies
{loaded_skills}

## Patch Budget
(no per-skill patch budget)

## Completed Task
Task: {task_description}
Outcome: {outcome}

Reflection:
{reflection_text}

Structured reflection:
{reflection_json}

Trajectory:
{trajectory}

## Instructions
Apply the skill self-improvement rules above. Keep skills class-level and
future-facing. Return pure JSON only.
"""


_VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_INVALID_NAME_CHARS_RE = re.compile(r"[^a-z0-9._-]+")
_MULTI_DASH_RE = re.compile(r"-{2,}")
_WORD_RE = re.compile(r"\w+")
_FRONTMATTER_END_RE = re.compile(r"\n---\s*\n")
_ALLOWED_SUPPORT_DIRS = {"references", "templates", "scripts", "assets"}
_MAX_SKILL_CONTENT_CHARS = 100_000
_MAX_SUPPORT_FILE_CHARS = 100_000


def _slugify_skill_name(raw: str) -> str:
    name = (raw or "").strip().lower().replace(" ", "-").replace("_", "-")
    name = _INVALID_NAME_CHARS_RE.sub("-", name)
    name = _MULTI_DASH_RE.sub("-", name).strip("-._")
    if not name:
        return "skill"
    if not name[0].isalnum():
        name = "skill-" + name
    return name[:64]


def _validate_skill_name(name: str) -> Optional[str]:
    if not name:
        return "Skill name is required."
    if len(name) > 64:
        return "Skill name exceeds 64 characters."
    if not _VALID_NAME_RE.match(name):
        return (
            f"Invalid skill name {name!r}. Use lowercase letters, numbers, "
            "hyphens, dots, and underscores; start with a letter or digit."
        )
    return None


def _parse_skill_md(content: str) -> tuple[dict[str, Any], str]:
    """Parse and validate a Hermes/agentskills SKILL.md document."""
    if not content or not content.strip():
        raise ValueError("SKILL.md content cannot be empty.")
    if len(content) > _MAX_SKILL_CONTENT_CHARS:
        raise ValueError(
            f"SKILL.md content is too large ({len(content)} chars)."
        )
    if not content.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter.")

    match = _FRONTMATTER_END_RE.search(content[3:])
    if not match:
        raise ValueError("SKILL.md frontmatter is not closed.")

    yaml_text = content[3 : match.start() + 3]
    body = content[match.end() + 3 :].strip()
    try:
        data = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid SKILL.md frontmatter YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping.")
    if not str(data.get("name") or "").strip():
        raise ValueError("SKILL.md frontmatter must include 'name'.")
    if not str(data.get("description") or "").strip():
        raise ValueError("SKILL.md frontmatter must include 'description'.")
    if len(str(data.get("description"))) > 1024:
        raise ValueError("SKILL.md description exceeds 1024 characters.")
    if not body:
        raise ValueError("SKILL.md must include body instructions.")
    return data, body


def _build_skill_md(
    *,
    name: str,
    description: str,
    body: str,
    category: str = "why-fail",
    tags: Optional[list[str]] = None,
    version: str = "1.0.0",
) -> str:
    frontmatter = {
        "name": name,
        "description": description.strip(),
        "version": version,
        "metadata": {
            "hermes": {
                "tags": list(tags or []),
                "category": category or "why-fail",
            }
        },
    }
    yaml_text = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    ).strip()
    return f"---\n{yaml_text}\n---\n\n{body.strip()}\n"


def _validate_support_path(file_path: str) -> Optional[str]:
    if not file_path:
        return "file_path is required."
    path = PurePosixPath(str(file_path))
    if path.is_absolute() or ".." in path.parts:
        return "Support file paths must be relative and may not contain '..'."
    if len(path.parts) < 2:
        return "Support file path must include a subdirectory and filename."
    if path.parts[0] not in _ALLOWED_SUPPORT_DIRS:
        allowed = ", ".join(sorted(_ALLOWED_SUPPORT_DIRS))
        return f"Support file must live under one of: {allowed}."
    return None


@dataclass
class HermesSkill:
    """One Hermes-style procedural memory skill."""

    id: str
    name: str
    description: str
    content: str
    category: str = "why-fail"
    support_files: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source_task_ids: list[str] = field(default_factory=list)
    use_count: int = 0
    patch_count: int = 0
    version: int = 1
    archived: bool = False
    absorbed_into: Optional[str] = None

    def render_index_line(self) -> str:
        return f"[{self.id}] {self.name}: {self.description}"

    def render_full(self, *, include_support_files: bool = True) -> str:
        parts = [self.render_index_line(), "", self.content.strip()]
        if include_support_files and self.support_files:
            parts.append("")
            parts.append("[Supporting files]")
            for path, content in sorted(self.support_files.items()):
                preview = content.strip()
                if len(preview) > 3000:
                    preview = preview[:3000] + "\n...[support file truncated]..."
                parts.append(f"### {path}\n{preview}")
        return "\n".join(parts).rstrip()


class HermesState(AgentState):
    """Hermes Agent style procedural skill memory.

    Parameters mirror the other AgentState subclasses. ``embedder`` is used for
    semantic skill retrieval when available; otherwise retrieval falls back to
    deterministic keyword overlap.
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: EmbeddingClient | None = None,
        max_tokens: int = 50_000,
        max_skills: int = 64,
        max_actions_per_update: int = 6,
        curator_max_tokens: int = 4096,
        retrieve_full_skills: bool = True,
    ):
        self._llm = llm
        self._embedder = embedder
        self._max_tokens = int(max_tokens)
        self._max_skills = max(1, int(max_skills))
        self._max_actions_per_update = max(1, int(max_actions_per_update))
        self._curator_max_tokens = max(256, int(curator_max_tokens))
        self._retrieve_full_skills = bool(retrieve_full_skills)

        self._skills: dict[str, HermesSkill] = {}
        self._next_id = 1
        self._update_count = 0
        self._curator_calls = 0
        self._parse_failures = 0
        self._action_failures = 0
        self._create_count = 0
        self._edit_count = 0
        self._patch_count = 0
        self._delete_count = 0
        self._write_file_count = 0
        self._remove_file_count = 0

    # ------------------------------------------------------------------
    # AgentState interface
    # ------------------------------------------------------------------

    def retrieve(self, task_description: str, k: int = 5) -> str:
        active = [s for s in self._skills.values() if not s.archived]
        if not active:
            return ""

        k = max(1, min(int(k), len(active)))
        candidates = [
            f"{s.name}\n{s.description}\n{s.content[:2000]}"
            for s in active
        ]
        if self._embedder is not None:
            try:
                indices = list(self._embedder.top_k_indices(task_description, candidates, k))
            except Exception:
                indices = self._keyword_topk(task_description, candidates, k)
        else:
            indices = self._keyword_topk(task_description, candidates, k)

        selected: list[HermesSkill] = []
        for idx in indices:
            if 0 <= idx < len(active):
                skill = active[idx]
                skill.use_count += 1
                selected.append(skill)

        if not selected:
            return ""

        lines = [
            "## Hermes Skills (procedural memory)",
            "Scan the available skills first. Use the loaded skill bodies when",
            "they are relevant, and carry their pitfalls/checklists into the",
            "solution attempt.",
            "",
            "<available_skills>",
        ]
        for skill in active:
            lines.append(skill.render_index_line())
        lines.extend(["</available_skills>", ""])

        if self._retrieve_full_skills:
            lines.append("## Loaded Skill Bodies")
            for skill in selected:
                lines.append("")
                lines.append(skill.render_full())

        return "\n".join(lines).rstrip()

    def update(self, reflection: dict) -> None:
        self._update_count += 1

        outcome = (reflection.get("outcome") or "").lower()
        if outcome == "pending":
            return

        task_description = str(reflection.get("task_description") or "").strip()
        trajectory = str(reflection.get("trajectory") or "").strip()
        reflection_text = str(reflection.get("reflection_text") or "").strip()
        if not task_description or not (trajectory or reflection_text):
            return

        prompt = HERMES_UPDATE_PROMPT_TEMPLATE.format(
            system=HERMES_SKILL_UPDATE_SYSTEM,
            skill_index=self._render_skill_index() or "(no skills yet)",
            loaded_skills=self._render_loaded_skills_for_update(task_description)
            or "(no skills yet)",
            task_description=task_description,
            outcome=outcome or "unknown",
            reflection_text=reflection_text or "(none)",
            reflection_json=json.dumps(
                reflection.get("reflection_json") or {},
                ensure_ascii=False,
                indent=2,
            ),
            trajectory=self._truncate_middle(trajectory, 60_000),
        )

        self._curator_calls += 1
        try:
            raw = self._llm.generate(prompt, max_tokens=self._curator_max_tokens)
        except Exception as exc:  # pragma: no cover - external LLM failure path
            print(f"  [Hermes] skill review LLM call failed: {type(exc).__name__}: {exc}")
            self._parse_failures += 1
            return

        data = extract_json(raw)
        if not isinstance(data, dict) or not isinstance(data.get("actions"), list):
            self._parse_failures += 1
            print("  [Hermes] skill review: could not parse actions JSON")
            return

        metadata = reflection.get("metadata") or {}
        task_id = (
            metadata.get("task_id")
            or metadata.get("instance_id")
            or reflection.get("task_id")
            or f"update-{self._update_count}"
        )

        for action in data["actions"][: self._max_actions_per_update]:
            if not isinstance(action, dict):
                self._action_failures += 1
                continue
            ok, message = self._apply_action(
                action,
                source_task_id=str(task_id),
            )
            if not ok:
                self._action_failures += 1
                print(f"  [Hermes] action skipped: {message}")

        self._prune_to_budget()

    def serialize(self) -> str:
        return json.dumps(
            {
                "type": "hermes",
                "skills": {name: asdict(skill) for name, skill in self._skills.items()},
                "next_id": self._next_id,
                "update_count": self._update_count,
                "curator_calls": self._curator_calls,
                "parse_failures": self._parse_failures,
                "action_failures": self._action_failures,
                "create_count": self._create_count,
                "edit_count": self._edit_count,
                "patch_count": self._patch_count,
                "delete_count": self._delete_count,
                "write_file_count": self._write_file_count,
                "remove_file_count": self._remove_file_count,
            },
            indent=2,
        )

    def load_serialized(self, payload: str) -> None:
        data = json.loads(payload)
        if data.get("type") != "hermes":
            raise ValueError(f"Expected hermes snapshot, got: {data.get('type')!r}")

        self._skills = {}
        for name, raw in (data.get("skills") or {}).items():
            if not isinstance(raw, dict):
                continue
            skill_name = _slugify_skill_name(str(raw.get("name") or name))
            self._skills[skill_name] = HermesSkill(
                id=str(raw.get("id") or f"hm-{len(self._skills) + 1:05d}"),
                name=skill_name,
                description=str(raw.get("description") or ""),
                content=str(raw.get("content") or ""),
                category=str(raw.get("category") or "why-fail"),
                support_files={
                    str(k): str(v)
                    for k, v in (raw.get("support_files") or {}).items()
                },
                created_at=float(raw.get("created_at") or time.time()),
                updated_at=float(raw.get("updated_at") or time.time()),
                source_task_ids=list(raw.get("source_task_ids") or []),
                use_count=int(raw.get("use_count") or 0),
                patch_count=int(raw.get("patch_count") or 0),
                version=int(raw.get("version") or 1),
                archived=bool(raw.get("archived") or False),
                absorbed_into=raw.get("absorbed_into"),
            )

        self._next_id = int(data.get("next_id") or (len(self._skills) + 1))
        self._update_count = int(data.get("update_count") or 0)
        self._curator_calls = int(data.get("curator_calls") or 0)
        self._parse_failures = int(data.get("parse_failures") or 0)
        self._action_failures = int(data.get("action_failures") or 0)
        self._create_count = int(data.get("create_count") or 0)
        self._edit_count = int(data.get("edit_count") or 0)
        self._patch_count = int(data.get("patch_count") or 0)
        self._delete_count = int(data.get("delete_count") or 0)
        self._write_file_count = int(data.get("write_file_count") or 0)
        self._remove_file_count = int(data.get("remove_file_count") or 0)

    def stats(self) -> dict:
        active = [s for s in self._skills.values() if not s.archived]
        support_files = sum(len(s.support_files) for s in self._skills.values())
        rendered = "\n\n".join(s.render_full() for s in active)
        return {
            "type": "hermes",
            "num_skills": len(active),
            "num_archived_skills": len(self._skills) - len(active),
            "num_support_files": support_files,
            "update_count": self._update_count,
            "curator_calls": self._curator_calls,
            "parse_failures": self._parse_failures,
            "action_failures": self._action_failures,
            "create_count": self._create_count,
            "edit_count": self._edit_count,
            "patch_count": self._patch_count,
            "delete_count": self._delete_count,
            "write_file_count": self._write_file_count,
            "remove_file_count": self._remove_file_count,
            "approx_tokens": approx_tokens(rendered),
            "skill_names": [s.name for s in active],
        }

    # ------------------------------------------------------------------
    # skill_manage action handlers
    # ------------------------------------------------------------------

    def _apply_action(
        self,
        action: dict[str, Any],
        source_task_id: str,
    ) -> tuple[bool, str]:
        verb = str(action.get("action") or action.get("type") or "").strip().lower()
        name = _slugify_skill_name(str(action.get("name") or ""))
        if verb not in {"create", "edit", "patch", "delete", "write_file", "remove_file"}:
            return False, f"unknown action {verb!r}"
        if verb != "create" and name not in self._skills:
            return False, f"skill {name!r} not found"

        if verb == "create":
            return self._create_skill(action, name, source_task_id)
        if verb == "edit":
            return self._edit_skill(action, name, source_task_id)
        if verb == "patch":
            return self._patch_skill(action, name, source_task_id)
        if verb == "delete":
            return self._delete_skill(action, name)
        if verb == "write_file":
            return self._write_file(action, name, source_task_id)
        return self._remove_file(action, name, source_task_id)

    def _create_skill(
        self,
        action: dict[str, Any],
        name: str,
        source_task_id: str,
    ) -> tuple[bool, str]:
        err = _validate_skill_name(name)
        if err:
            return False, err
        if name in self._skills:
            return False, f"skill {name!r} already exists"

        content = str(action.get("content") or "").strip()
        if not content:
            description = str(action.get("description") or "").strip()
            body = str(action.get("body") or "").strip()
            if not description or not body:
                return False, "create requires full content, or description plus body"
            category = str(action.get("category") or "why-fail").strip() or "why-fail"
            tags = action.get("tags")
            if not isinstance(tags, list):
                tags = []
            content = _build_skill_md(
                name=name,
                description=description,
                body=body,
                category=category,
                tags=[str(t) for t in tags],
            )

        try:
            frontmatter, _body = _parse_skill_md(content)
        except ValueError as exc:
            return False, str(exc)

        description = str(frontmatter.get("description") or "").strip()
        category = "why-fail"
        metadata = frontmatter.get("metadata")
        if isinstance(metadata, dict):
            hermes = metadata.get("hermes")
            if isinstance(hermes, dict) and hermes.get("category"):
                category = str(hermes.get("category"))

        skill_id = f"hm-{self._next_id:05d}"
        self._next_id += 1
        self._skills[name] = HermesSkill(
            id=skill_id,
            name=name,
            description=description,
            content=content,
            category=category,
            source_task_ids=[source_task_id],
        )
        self._create_count += 1
        print(f"  [Hermes] create [{skill_id}] {name}")
        return True, "created"

    def _edit_skill(
        self,
        action: dict[str, Any],
        name: str,
        source_task_id: str,
    ) -> tuple[bool, str]:
        content = str(action.get("content") or "").strip()
        if not content:
            return False, "edit requires full SKILL.md content"
        try:
            frontmatter, _body = _parse_skill_md(content)
        except ValueError as exc:
            return False, str(exc)

        skill = self._skills[name]
        skill.content = content
        skill.description = str(frontmatter.get("description") or skill.description).strip()
        skill.updated_at = time.time()
        skill.version += 1
        skill.patch_count += 1
        skill.source_task_ids = list(dict.fromkeys(skill.source_task_ids + [source_task_id]))
        self._edit_count += 1
        print(f"  [Hermes] edit [{skill.id}] {name}")
        return True, "edited"

    def _patch_skill(
        self,
        action: dict[str, Any],
        name: str,
        source_task_id: str,
    ) -> tuple[bool, str]:
        old = action.get("old_string")
        new = action.get("new_string")
        if old is None or old == "":
            return False, "patch requires old_string"
        if new is None:
            return False, "patch requires new_string"
        old_s = str(old)
        new_s = str(new)
        replace_all = bool(action.get("replace_all", False))
        file_path = action.get("file_path")
        skill = self._skills[name]

        if file_path:
            path = str(file_path)
            err = _validate_support_path(path)
            if err:
                return False, err
            if path not in skill.support_files:
                return False, f"support file {path!r} not found"
            current = skill.support_files[path]
            patched, count = self._replace(current, old_s, new_s, replace_all)
            if count == 0:
                return False, "old_string not found"
            if count < 0:
                return False, "old_string matched multiple places; use replace_all or more context"
            if len(patched) > _MAX_SUPPORT_FILE_CHARS:
                return False, "patched support file is too large"
            skill.support_files[path] = patched
        else:
            patched, count = self._replace(skill.content, old_s, new_s, replace_all)
            if count == 0:
                return False, "old_string not found"
            if count < 0:
                return False, "old_string matched multiple places; use replace_all or more context"
            try:
                frontmatter, _body = _parse_skill_md(patched)
            except ValueError as exc:
                return False, f"patch would break SKILL.md: {exc}"
            skill.content = patched
            skill.description = str(frontmatter.get("description") or skill.description).strip()

        skill.updated_at = time.time()
        skill.version += 1
        skill.patch_count += 1
        skill.source_task_ids = list(dict.fromkeys(skill.source_task_ids + [source_task_id]))
        self._patch_count += 1
        print(f"  [Hermes] patch [{skill.id}] {name} ({count} replacement{'s' if count != 1 else ''})")
        return True, "patched"

    def _delete_skill(self, action: dict[str, Any], name: str) -> tuple[bool, str]:
        skill = self._skills[name]
        skill.archived = True
        skill.absorbed_into = (
            str(action.get("absorbed_into")).strip()
            if action.get("absorbed_into") is not None
            else None
        )
        skill.updated_at = time.time()
        self._delete_count += 1
        print(f"  [Hermes] delete/archive [{skill.id}] {name}")
        return True, "deleted"

    def _write_file(
        self,
        action: dict[str, Any],
        name: str,
        source_task_id: str,
    ) -> tuple[bool, str]:
        path = str(action.get("file_path") or "")
        content = action.get("file_content")
        err = _validate_support_path(path)
        if err:
            return False, err
        if content is None:
            return False, "write_file requires file_content"
        content_s = str(content)
        if len(content_s) > _MAX_SUPPORT_FILE_CHARS:
            return False, "support file content is too large"

        skill = self._skills[name]
        skill.support_files[path] = content_s
        skill.updated_at = time.time()
        skill.version += 1
        skill.patch_count += 1
        skill.source_task_ids = list(dict.fromkeys(skill.source_task_ids + [source_task_id]))
        self._write_file_count += 1
        print(f"  [Hermes] write_file [{skill.id}] {name}:{path}")
        return True, "file written"

    def _remove_file(
        self,
        action: dict[str, Any],
        name: str,
        source_task_id: str,
    ) -> tuple[bool, str]:
        path = str(action.get("file_path") or "")
        err = _validate_support_path(path)
        if err:
            return False, err

        skill = self._skills[name]
        if path not in skill.support_files:
            return False, f"support file {path!r} not found"
        skill.support_files.pop(path, None)
        skill.updated_at = time.time()
        skill.version += 1
        skill.patch_count += 1
        skill.source_task_ids = list(dict.fromkeys(skill.source_task_ids + [source_task_id]))
        self._remove_file_count += 1
        print(f"  [Hermes] remove_file [{skill.id}] {name}:{path}")
        return True, "file removed"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_skill_index(self) -> str:
        return "\n".join(
            skill.render_index_line()
            for skill in self._skills.values()
            if not skill.archived
        )

    def _render_loaded_skills_for_update(self, task_description: str) -> str:
        active = [s for s in self._skills.values() if not s.archived]
        if not active:
            return ""
        k = min(5, len(active))
        candidates = [f"{s.name}\n{s.description}\n{s.content[:2000]}" for s in active]
        if self._embedder is not None:
            try:
                indices = list(self._embedder.top_k_indices(task_description, candidates, k))
            except Exception:
                indices = self._keyword_topk(task_description, candidates, k)
        else:
            indices = self._keyword_topk(task_description, candidates, k)
        blocks = []
        for i in indices:
            if not 0 <= i < len(active):
                continue
            skill = active[i]
            blocks.append(skill.render_full(include_support_files=False))
        return "\n\n".join(blocks)

    def _keyword_topk(self, query: str, candidates: list[str], k: int) -> list[int]:
        tokens = set(_WORD_RE.findall((query or "").lower()))
        if not tokens:
            return list(range(min(k, len(candidates))))
        scored: list[tuple[int, int]] = []
        for i, cand in enumerate(candidates):
            cand_tokens = set(_WORD_RE.findall(cand.lower()))
            scored.append((len(tokens & cand_tokens), i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [i for _score, i in scored[:k]]

    @staticmethod
    def _replace(text: str, old: str, new: str, replace_all: bool) -> tuple[str, int]:
        count = text.count(old)
        if count == 0:
            return text, 0
        if not replace_all and count != 1:
            return text, -count
        return text.replace(old, new, -1 if replace_all else 1), count

    @staticmethod
    def _truncate_middle(text: str, max_chars: int) -> str:
        if not text or len(text) <= max_chars:
            return text
        marker = f"\n...[trajectory truncated: omitted {len(text) - max_chars} chars]...\n"
        keep = max_chars - len(marker)
        if keep <= 0:
            return text[:max_chars]
        head = int(keep * 0.6)
        tail = keep - head
        return text[:head] + marker + text[-tail:]

    def _prune_to_budget(self) -> None:
        active = [s for s in self._skills.values() if not s.archived]
        if len(active) > self._max_skills:
            by_age = sorted(active, key=lambda s: (s.created_at, s.id))
            for skill in by_age[: len(active) - self._max_skills]:
                skill.archived = True

        while True:
            active = [s for s in self._skills.values() if not s.archived]
            rendered = "\n\n".join(s.render_full() for s in active)
            if approx_tokens(rendered) <= self._max_tokens or not active:
                return
            oldest = min(active, key=lambda s: (s.created_at, s.id))
            oldest.archived = True
