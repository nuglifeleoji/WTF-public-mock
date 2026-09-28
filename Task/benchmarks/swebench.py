"""SWE-bench benchmark adapter.

SWE-bench evaluates LLMs on real-world GitHub issues. Given a repository at a
specific commit and an issue description, the agent must produce a git patch
that makes the failing tests pass.

Install the SWE-bench optional dependencies and Java for BM25 retrieval;
see Task/README.md for setup and execution. Evaluation uses the official
harness with local Docker or an authenticated Modal account.

Dataset variants:
    princeton-nlp/SWE-bench_Lite     — 300 tasks (default)
    princeton-nlp/SWE-bench_Verified — 500 human-verified tasks
    princeton-nlp/SWE-bench          — 2,294 tasks (full set)

See: https://github.com/swe-bench/SWE-bench
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

from .base import Benchmark

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_REACT_STEPS = 15
_MAX_OUTPUT_CHARS = 10_000
_BASH_TIMEOUT = 60    # seconds per bash command
_CLONE_TIMEOUT = 300  # seconds for git clone
_HARNESS_TIMEOUT = 60 * 60 * 2  # 2 hours, to allow Modal image setup/eval
_OFFICIAL_DOCUMENT_ENCODING = "file_name_and_contents"
_OFFICIAL_MAX_CONTEXT_TOKENS = 16_000
_OFFICIAL_RETRIEVAL_TOP_K = 20

_OFFICIAL_PATCH_EXAMPLE = """--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
 def euclidean(a, b):
-    while b:
-        a, b = b, a % b
-    return a
+    if b == 0:
+        return a
+    return euclidean(b, a % b)
 
 
 def bresenham(x0, y0, x1, y1):
     points = []
     dx = abs(x1 - x0)
     dy = abs(y1 - y0)
-    sx = 1 if x0 < x1 else -1
-    sy = 1 if y0 < y1 else -1
-    err = dx - dy
+    x, y = x0, y0
+    sx = -1 if x0 > x1 else 1
+    sy = -1 if y0 > y1 else 1
 
-    while True:
-        points.append((x0, y0))
-        if x0 == x1 and y0 == y1:
-            break
-        e2 = 2 * err
-        if e2 > -dy:
+    if dx > dy:
+        err = dx / 2.0
+        while x != x1:
+            points.append((x, y))
             err -= dy
-            x0 += sx
-        if e2 < dx:
-            err += dx
-            y0 += sy
+            if err < 0:
+                y += sy
+                err += dx
+            x += sx
+    else:
+        err = dy / 2.0
+        while y != y1:
+            points.append((x, y))
+            err -= dx
+            if err < 0:
+                x += sx
+                err += dy
+            y += sy
 
+    points.append((x, y))
     return points"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(instance: dict, context: str) -> str:
    repo = instance["repo"]
    hints = (instance.get("hints_text") or "").strip()

    playbook_section = ""
    if _playbook_has_bullets(context):
        playbook_section = f"""
## Playbook (strategies from past experience)

The following strategies have been curated from past experience. Apply them where relevant:

{context}
"""
    hints_section = f"\n## Hints\n\n{hints}" if hints else ""

    return f"""\
You are an expert software engineer fixing a GitHub issue in `{repo}`.
Your ONLY job is to produce a minimal code patch — do NOT run tests or install packages.

{_SWEBENCH_TEST_GUARD}
{playbook_section}
## Workflow (follow exactly)

**Step 1 — Explore** (1-3 bash commands max):
- Read the error or failing test to understand what is broken.
- `grep`/`cat` the relevant source files. Keep it short.

**Step 2 — Fix** (edit the file with one python3 -c command):
```bash
python3 -c "
import pathlib
p = pathlib.Path('path/to/file.py')
src = p.read_text()
p.write_text(src.replace(
    r'''exact old lines''',
    r'''exact new lines''',
))
"
```

**Step 3 — Verify & finish**:
```bash
git diff
```
When the diff looks correct, output exactly:
```
DONE
```

Rules:
- Every command MUST be inside a ```bash ... ``` block.
- Do NOT run pytest, pip install, or create virtual environments.
- Do NOT output DONE until git diff shows your change.
{hints_section}"""


def _split_official_prompt(prompt: str) -> tuple[str, str]:
    """Match the official API calling convention: first line is the system prompt."""
    if "\n" not in prompt:
        return prompt.strip(), ""
    system, user = prompt.split("\n", 1)
    return system.strip(), user.lstrip("\n")


def _read_text_file(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _official_add_lines(content: str) -> str:
    return "\n".join(f"{idx} {line}" for idx, line in enumerate(content.split("\n"), start=1))


def _official_make_code_text(files_dict: dict[str, str], add_line_numbers: bool = True) -> str:
    all_text = ""
    for filename, contents in sorted(files_dict.items()):
        all_text += f"[start of {filename}]\n"
        all_text += _official_add_lines(contents) if add_line_numbers else contents
        all_text += f"\n[end of {filename}]\n"
    return all_text.strip("\n")


def _official_prompt_style_3(instance: dict) -> str:
    readmes_text = _official_make_code_text(instance["readmes"])
    code_text = _official_make_code_text(instance["file_contents"])
    final_text = [
        "You will be provided with a partial code base and an issue statement explaining a problem to resolve.",
        "<issue>",
        instance["problem_statement"],
        "</issue>",
        "",
        "<code>",
        readmes_text,
        code_text,
        "</code>",
        "",
        (
            "Here is an example of a patch file. It consists of changes to the code base. "
            "It specifies the file names, the line numbers of each change, and the removed and added lines. "
            "A single patch file can contain changes to multiple files."
        ),
        "<patch>",
        _OFFICIAL_PATCH_EXAMPLE,
        "</patch>",
        "",
        (
            "I need you to solve the provided issue by generating a single patch file that I can apply "
            "directly to this repository using git apply. Please respond with a single patch "
            "file in the format shown above."
        ),
        "Respond below:",
    ]
    return "\n".join(final_text)


def _build_official_prompt(instance: dict, repo_path: str, work_dir: str) -> tuple[str, list[str]]:
    """
    Build an official-style SWE-bench inference prompt.

    This mirrors the upstream inference pipeline:
      1. build a BM25 index over repository files
      2. retrieve top files for the problem statement
      3. format a style-3 patch-generation prompt
    """
    from swebench.inference.make_datasets.bm25_retrieval import (
        DOCUMENT_ENCODING_FUNCTIONS,
        make_index,
        search,
    )
    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenizer_func = lambda text, tok: tok.encode(text, disallowed_special=())
    encoding_func = DOCUMENT_ENCODING_FUNCTIONS[_OFFICIAL_DOCUMENT_ENCODING]
    query_instance = {
        "instance_id": instance["instance_id"],
        "problem_statement": instance["problem_statement"],
    }
    index_dir = make_index(
        repo_dir=repo_path,
        root_dir=work_dir,
        query=instance["problem_statement"],
        commit=instance["base_commit"],
        document_encoding_func=encoding_func,
        python=sys.executable,
        instance_id=instance["instance_id"],
    )
    retrieval = search(query_instance, index_dir) or {"hits": []}
    hits = retrieval.get("hits", [])[:_OFFICIAL_RETRIEVAL_TOP_K]

    prompt_instance = {
        "instance_id": instance["instance_id"],
        "problem_statement": instance["problem_statement"],
        "readmes": {},
        "file_contents": {},
    }
    base_prompt = _official_prompt_style_3(prompt_instance)
    current_tokens = len(tokenizer_func(base_prompt, tokenizer))
    included_files: list[str] = []
    file_contents: dict[str, str] = {}

    for hit in hits:
        relative_path = hit["docid"]
        abs_path = pathlib.Path(repo_path) / relative_path
        if not abs_path.exists() or not abs_path.is_file():
            continue
        content = _read_text_file(abs_path)
        candidate_fragment = f"[start of {relative_path}]\n{content}\n[end of {relative_path}]"
        candidate_tokens = len(tokenizer_func(candidate_fragment, tokenizer))
        if current_tokens + candidate_tokens > _OFFICIAL_MAX_CONTEXT_TOKENS:
            continue
        included_files.append(relative_path)
        file_contents[relative_path] = content
        current_tokens += candidate_tokens

    prompt_instance["file_contents"] = file_contents
    return _official_prompt_style_3(prompt_instance), included_files


# ---------------------------------------------------------------------------
# Bash helpers
# ---------------------------------------------------------------------------

def _extract_all_bash(text: str) -> list[str]:
    """Extract all ```bash or ```sh code blocks from an LLM response."""
    return re.findall(r"```(?:bash|sh)\n(.*?)```", text, re.DOTALL)


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$")

_BULLET_LINE_RE = re.compile(r"^\[[\w]+-\d{5}\]", re.MULTILINE)


def _playbook_has_bullets(context: str) -> bool:
    """True iff the rendered playbook contains at least one curated bullet.

    PlaybookState.retrieve() always returns the section scaffolding; without
    this check an empty playbook still injects headers into the prompt and
    primes the model to invent advice (e.g. insert regression tests).
    """
    return bool(context) and bool(_BULLET_LINE_RE.search(context))


# The SWE-bench harness applies the dataset's test_patch on top of the model
# patch before running FAIL_TO_PASS tests. When the model also edits test
# files, the two patches collide and the grader reports "unresolved" even if
# the source fix itself is correct.
_SWEBENCH_TEST_GUARD = (
    "IMPORTANT: Do not modify or add test files. The evaluation harness "
    "applies its own test patch; your patch must only change source files "
    "to fix the issue."
)

_RETRYABLE_MODAL_EVAL_ERROR_RE = re.compile(
    r"("
    r"Error creating sandbox|"
    r"FilesystemExecutionError|"
    r"internal server error|"
    r"Could not deserialize remote exception|"
    r"Sandbox.*deprecated"
    r")",
    re.IGNORECASE,
)

_NON_RETRYABLE_MODAL_EVAL_ERROR_RE = re.compile(
    r"("
    r">>>>> Patch Apply Failed|"
    r"patch does not apply|"
    r"failed to run builder command|"
    r"missing the 'build_editable' hook|"
    r"build backend that is missing.*build_editable"
    r")",
    re.IGNORECASE,
)


def _is_retryable_swebench_modal_error(text: str) -> bool:
    if not text:
        return False
    if _NON_RETRYABLE_MODAL_EVAL_ERROR_RE.search(text):
        return False
    return bool(_RETRYABLE_MODAL_EVAL_ERROR_RE.search(text))


def _is_unified_file_header_boundary(lines: list[str], idx: int) -> bool:
    line = lines[idx]
    if line.startswith("diff --git "):
        return True
    return (
        line.startswith("--- ")
        and idx + 1 < len(lines)
        and lines[idx + 1].startswith("+++ ")
    )


def _normalize_hunk_headers(patch: str) -> str:
    # Some models (e.g. gpt-5.4) emit @@ -L,N +L,M @@ headers whose N/M line counts don't
    # match the hunk body, which makes `git apply` reject the patch as corrupt. Recount
    # the body here and rewrite the header so the harness can apply it cleanly.
    if not patch:
        return patch
    lines = patch.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        match = _HUNK_HEADER_RE.match(lines[i])
        if not match:
            out.append(lines[i])
            i += 1
            continue
        old_start, new_start, trailing = match.group(1), match.group(2), match.group(3)
        body: list[str] = []
        j = i + 1
        while j < len(lines):
            ln = lines[j]
            if _is_unified_file_header_boundary(lines, j):
                break
            if not ln:
                body.append(ln)
                j += 1
                continue
            if ln[0] in (" ", "+", "-", "\\"):
                body.append(ln)
                j += 1
                continue
            break
        old_count = new_count = 0
        for ln in body:
            if not ln:
                old_count += 1
                new_count += 1
                continue
            c = ln[0]
            if c == " ":
                old_count += 1
                new_count += 1
            elif c == "-":
                old_count += 1
            elif c == "+":
                new_count += 1
        out.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{trailing}")
        out.extend(body)
        i = j
    return "\n".join(out)


def _run_bash(command: str, cwd: str) -> str:
    """Run a bash command in cwd, return combined stdout+stderr."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True,
            cwd=cwd, timeout=_BASH_TIMEOUT,
        )
        out = result.stdout + result.stderr
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"(timeout after {_BASH_TIMEOUT}s)"
    except Exception as exc:
        return f"(error: {exc})"


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _clone_repo(repo: str, base_commit: str, work_dir: str) -> str:
    """
    Clone *repo* (a 'owner/name' GitHub slug) into work_dir and check out
    base_commit. Returns the path to the cloned repo.

    Uses a multi-strategy fetch so shallow clones still reach old commits.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    auth = f"{token}@" if token else ""
    url = f"https://{auth}github.com/{repo}.git"
    repo_path = os.path.join(work_dir, repo.split("/")[-1])

    subprocess.run(
        ["git", "clone", "--depth", "1", url, repo_path],
        capture_output=True, timeout=_CLONE_TIMEOUT, check=False,
    )
    # Check for a valid git repo, not just the directory (partial clones leave a dir behind).
    if not os.path.isfile(os.path.join(repo_path, ".git", "HEAD")):
        shutil.rmtree(repo_path, ignore_errors=True)
        subprocess.run(
            ["git", "clone", url, repo_path],
            capture_output=True, timeout=_CLONE_TIMEOUT, check=True,
        )

    def _git(args: list[str]) -> int:
        return subprocess.run(
            ["git"] + args, capture_output=True, cwd=repo_path, timeout=120,
        ).returncode

    # Try direct fetch, then unshallow if needed.
    strategies = [
        ["fetch", "--depth=1", "origin", base_commit],
        ["fetch", "--unshallow"],
        ["fetch", "origin"],
    ]
    for strat in strategies:
        _git(strat)
        if _git(["checkout", base_commit]) == 0:
            return repo_path

    raise RuntimeError(f"Failed to checkout {base_commit} in {repo}")


def _get_diff(repo_path: str) -> str:
    """Return the current git diff (tracked modifications + new untracked files) in the repo."""
    # Stage everything so new files are included in the diff.
    subprocess.run(
        ["git", "add", "-A"],
        capture_output=True, cwd=repo_path, timeout=30,
    )
    result = subprocess.run(
        ["git", "diff", "--cached", "HEAD"],
        capture_output=True, text=True, cwd=repo_path, timeout=30,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Trajectory helper
# ---------------------------------------------------------------------------

def _messages_to_trajectory(messages: list[dict]) -> str:
    role_prefix = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}
    parts = []
    for m in messages:
        prefix = role_prefix.get(m["role"], m["role"].upper())
        parts.append(f"{prefix}:\n{m['content']}")
    return "\n\n".join(parts)


def _parse_harness_report(report_path: pathlib.Path, instance_id: str) -> tuple[bool, str]:
    """Return (resolved, status) for *instance_id* in a harness report file."""
    with report_path.open() as fh:
        eval_results = json.load(fh)

    if isinstance(eval_results, dict) and "resolved_ids" in eval_results:
        if instance_id in eval_results.get("resolved_ids", []):
            return True, "resolved"
        if instance_id in eval_results.get("unresolved_ids", []):
            return False, "unresolved"
        if instance_id in eval_results.get("error_ids", []):
            return False, "error"
        if instance_id in eval_results.get("empty_patch_ids", []):
            return False, "empty_patch"
        if instance_id in eval_results.get("incomplete_ids", []):
            return False, "incomplete"
        return False, "not_reported"
    if isinstance(eval_results, dict) and instance_id in eval_results:
        resolved = bool(eval_results[instance_id].get("resolved", False))
        return resolved, "resolved" if resolved else "unresolved"
    if isinstance(eval_results, dict):
        resolved = instance_id in eval_results.get("resolved", [])
        return resolved, "resolved" if resolved else "unresolved"
    return False, "unrecognized_report"


# ---------------------------------------------------------------------------
# Evaluation via official SWE-bench harness
# ---------------------------------------------------------------------------

def _evaluate_with_harness(
    instance: dict,
    patch: str,
    run_id: str = "why-fail-eval",
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    model_name: str = "why-fail-agent",
    use_modal: bool = False,
) -> dict:
    """
    Evaluate *patch* using the official swebench harness.

    - use_modal=True  → Modal cloud (no Docker needed; requires modal token new)
    - use_modal=False → local Docker (Docker daemon must be running)

    Returns a result dict with keys: outcome, score, details.
    Raises RuntimeError if the backend is unavailable or the harness fails.
    """
    if use_modal:
        try:
            from swebench.harness.modal_eval import validate_modal_credentials
            validate_modal_credentials()
        except Exception as exc:
            raise RuntimeError(
                f"Modal credentials not configured: {exc}. "
                "Run 'modal token new' to authenticate."
            ) from exc
    else:
        try:
            import docker  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Local SWE-bench evaluation requires the Docker Python package."
            ) from exc

    prediction = {
        "instance_id": instance["instance_id"],
        "model_patch": patch,
        "model_name_or_path": model_name,
    }

    # Use a unique run_id per instance so re-runs don't get skipped.
    safe_iid = instance["instance_id"].replace("/", "_").replace("-", "_")
    harness_run_id = f"{run_id}_{safe_iid}"
    attempts = 3 if use_modal else 1
    harness_summary = "(not run)"
    for attempt in range(attempts):
        with tempfile.TemporaryDirectory(prefix="swebench_eval_") as eval_dir:
            eval_path = pathlib.Path(eval_dir)
            pred_path = eval_path / "prediction.json"
            pred_path.write_text(json.dumps([prediction]))

            cmd = [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "--predictions_path", str(pred_path),
                "--dataset_name", dataset_name,
                "--split", split,
                "--max_workers", "1",
                "--instance_ids", instance["instance_id"],
                "--run_id", harness_run_id,
            ]
            if use_modal:
                cmd += ["--modal", "true"]

            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=_HARNESS_TIMEOUT,
                cwd=eval_dir,
            )
            harness_output = f"{proc.stdout}\n{proc.stderr}"
            harness_summary = (
                f"rc={proc.returncode} "
                f"stdout={proc.stdout[-400:]!r} "
                f"stderr={proc.stderr[-200:]!r}"
            )
            retryable_modal_error = (
                use_modal and _is_retryable_swebench_modal_error(harness_output)
            )
            if proc.returncode != 0:
                if retryable_modal_error and attempt < attempts - 1:
                    delay = min(60, 15 * (attempt + 1))
                    print(
                        "[SWEBench] Retrying Modal infrastructure failure "
                        f"for instance={instance['instance_id']} "
                        f"attempt={attempt + 1}/{attempts} after {delay}s"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Harness failed: {harness_summary}")

            summary_report = eval_path / f"{model_name.replace('/', '__')}.{harness_run_id}.json"
            instance_report = (
                eval_path
                / "logs"
                / "run_evaluation"
                / harness_run_id
                / model_name.replace("/", "__")
                / instance["instance_id"]
                / "report.json"
            )

            status = "unresolved"
            if summary_report.exists():
                resolved, status = _parse_harness_report(summary_report, instance["instance_id"])
            elif instance_report.exists():
                resolved, status = _parse_harness_report(instance_report, instance["instance_id"])
            else:
                existing_reports = sorted(str(p.relative_to(eval_path)) for p in eval_path.rglob("*.json"))
                raise RuntimeError(
                    f"Harness ran but no result file found for {instance['instance_id']!r}. "
                    f"harness_run_id={harness_run_id!r}, reports={existing_reports[:20]}. "
                    f"{harness_summary}"
                )

            if status == "error" and retryable_modal_error:
                if attempt < attempts - 1:
                    delay = min(60, 15 * (attempt + 1))
                    print(
                        "[SWEBench] Retrying Modal infrastructure error report "
                        f"for instance={instance['instance_id']} "
                        f"attempt={attempt + 1}/{attempts} after {delay}s"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    "SWE-bench Modal infrastructure error after retries: "
                    f"{harness_summary}"
                )

            return {
                "outcome": "success" if resolved else "failure",
                "score": 1.0 if resolved else 0.0,
                "details": (
                    f"swebench harness ({'modal' if use_modal else 'docker'}): "
                    f"{status}"
                ),
            }

    raise RuntimeError(f"Harness failed after retries: {harness_summary}")


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class SWEBenchBenchmark(Benchmark):
    """
    Adapter for SWE-bench.

    The agent is given the issue description and bash access to a fresh clone
    of the repository at the target commit. It explores and edits files freely,
    then the resulting git diff is evaluated via the swebench harness.

    Parameters
    ----------
    split:
        Dataset split — "test" (default), "dev", or "train".
    dataset_name:
        HuggingFace dataset name. Defaults to SWE-bench Lite.
    max_tasks:
        Cap the number of tasks. None = all tasks in the split.
    task_ids:
        Explicit list of instance_ids to run. Overrides max_tasks.
    max_react_steps:
        Maximum bash-command turns per task in the ReAct loop.
    predictions_path:
        Where to save generated patches (for auditing or later evaluation).
    solve_only:
        If True, skip evaluation — just generate and save patches.
    use_modal:
        If True, evaluate patches via Modal cloud (no Docker needed).
        Requires 'modal token new' to be configured.
    model_name:
        Label used in predictions JSON and harness run IDs.
    patch_retry_on_empty:
        If True, retry once with a patch-only prompt when the first official
        prompt response does not contain an extractable unified diff.
    dataset_revision:
        Optional Hugging Face dataset commit for reproducible task loading.
    """

    def __init__(
        self,
        split: str = "test",
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        max_tasks: Optional[int] = None,
        task_ids: Optional[list[str]] = None,
        max_react_steps: int = _MAX_REACT_STEPS,
        predictions_path: Optional[str] = None,
        solve_only: bool = False,
        use_modal: bool = False,
        model_name: str = "why-fail-agent",
        solver_backend: str = "official_prompt",
        patch_retry_on_empty: bool = False,
        dataset_revision: Optional[str] = None,
    ):
        if solver_backend not in {"official_prompt", "react"}:
            raise ValueError(
                f"Unsupported SWE-bench solver_backend: {solver_backend!r}. "
                "Expected 'official_prompt' or 'react'."
            )
        self._split = split
        self._dataset_name = dataset_name
        self._dataset_revision = dataset_revision
        self._max_tasks = max_tasks
        self._task_ids = task_ids
        self._max_react_steps = max_react_steps
        self._predictions_path = predictions_path
        self._solve_only = solve_only
        self._use_modal = use_modal
        self._model_name = model_name
        self._solver_backend = solver_backend
        self._patch_retry_on_empty = bool(patch_retry_on_empty)
        self._tasks: Optional[list[dict]] = None

    # ------------------------------------------------------------------
    # Benchmark interface — get_tasks
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[dict]:
        if self._tasks is not None:
            return self._tasks

        from datasets import load_dataset  # type: ignore[import]

        dataset_kwargs = {"split": self._split}
        if self._dataset_revision is not None:
            dataset_kwargs["revision"] = self._dataset_revision
        dataset = load_dataset(self._dataset_name, **dataset_kwargs)
        instances = list(dataset)

        if self._task_ids:
            by_id = {instance["instance_id"]: instance for instance in instances}
            missing = [task_id for task_id in self._task_ids if task_id not in by_id]
            if missing:
                raise ValueError("SWE-bench task_ids not found in the dataset: " + ", ".join(missing))
            instances = [by_id[task_id] for task_id in self._task_ids]
        elif self._max_tasks is not None:
            instances = instances[: self._max_tasks]

        self._tasks = [
            {
                "task_id": inst["instance_id"],
                "description": inst["problem_statement"],
                "split": self._split,
                "_swebench_instance": inst,
            }
            for inst in instances
        ]
        return self._tasks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_prediction(self, instance_id: str, patch: str) -> None:
        """Append or update one prediction in the JSON predictions file."""
        if not self._predictions_path:
            return
        prediction = {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": self._model_name,
        }
        path = self._predictions_path
        existing: list[dict] = []
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path) as fh:
                    loaded = json.load(fh)
                # Normalise: a single prediction dict, a list of dicts, or garbage.
                if isinstance(loaded, dict):
                    existing = [loaded]
                elif isinstance(loaded, list):
                    existing = [p for p in loaded if isinstance(p, dict)]
                # anything else (str, int, …) → start fresh
            except Exception:
                # Not valid JSON — try JSONL (one object per line).
                with open(path) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                obj = json.loads(line)
                                if isinstance(obj, dict):
                                    existing.append(obj)
                            except Exception:
                                pass
        existing = [p for p in existing if p.get("instance_id") != instance_id]
        existing.append(prediction)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(existing, fh, indent=2)

    # ------------------------------------------------------------------
    # Benchmark interface — solve (multi-turn bash ReAct loop)
    # ------------------------------------------------------------------

    def _solve_official_prompt(self, task: dict, context: str, llm) -> tuple[str, str]:
        instance = task["_swebench_instance"]
        work_dir = tempfile.mkdtemp(prefix="swebench_")
        try:
            repo_path = _clone_repo(
                instance["repo"], instance["base_commit"], work_dir
            )
            prompt, retrieved_files = _build_official_prompt(instance, repo_path, work_dir)
            system_prompt, user_prompt = _split_official_prompt(prompt)
            system_prompt = f"{system_prompt}\n\n{_SWEBENCH_TEST_GUARD}"
            if _playbook_has_bullets(context):
                system_prompt = (
                    f"{system_prompt}\n\n"
                    "## Playbook (strategies from past experience)\n\n"
                    "The following strategies have been curated from past tasks. "
                    "Apply them where relevant when producing your patch.\n\n"
                    f"{context}"
                )
            completion = llm.generate_messages(
                [{"role": "user", "content": user_prompt}],
                system=system_prompt,
                max_tokens=4096,
            )
            from swebench.inference.make_datasets.utils import extract_diff

            patch = (extract_diff(completion) or "").strip()
            retry_completion = ""
            if self._patch_retry_on_empty and not patch:
                retry_prompt = (
                    "No valid unified diff was extracted from your previous response. "
                    "Return only a valid unified diff patch now. Do not include analysis, "
                    "markdown fences, tests, or prose."
                )
                retry_completion = llm.generate_messages(
                    [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": completion},
                        {"role": "user", "content": retry_prompt},
                    ],
                    system=system_prompt,
                    max_tokens=4096,
                )
                patch = (extract_diff(retry_completion) or "").strip()
            patch = _normalize_hunk_headers(patch)
            if patch and not patch.endswith("\n"):
                patch += "\n"
            retrieved_summary = "\n".join(retrieved_files) if retrieved_files else "(none)"
            trajectory = (
                f"SYSTEM:\n{system_prompt}\n\n"
                f"USER:\n{user_prompt}\n\n"
                f"ASSISTANT:\n{completion}\n\n"
                + (
                    f"USER:\nNo valid unified diff was extracted. Return only a valid unified diff patch now.\n\n"
                    f"ASSISTANT_RETRY:\n{retry_completion}\n\n"
                    if retry_completion
                    else ""
                )
                + f"RETRIEVED_FILES:\n{retrieved_summary}"
            )
            return trajectory, patch
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _solve_react(self, task: dict, context: str, llm) -> tuple[str, str]:
        """
        Clone the repository, run a bash ReAct loop, capture the git diff,
        then return the resulting patch.

        Each turn:
          1. LLM reasons and emits ```bash``` command blocks.
          2. Commands are executed in the cloned repo.
          3. Output is fed back; repeat until DONE or max steps reached.

        Returns (trajectory_str, patch).
        """
        instance = task["_swebench_instance"]
        system_prompt = _build_system_prompt(instance, context)

        # Initialize before the try block so the except branch can always reference it,
        # even if _clone_repo() raises before messages is assigned.
        messages: list[dict] = []

        work_dir = tempfile.mkdtemp(prefix="swebench_")
        try:
            repo_path = _clone_repo(
                instance["repo"], instance["base_commit"], work_dir
            )

            # Seed the conversation: problem statement + repo listing in a single user message.
            init_listing = _run_bash(
                "find . -maxdepth 3 -type f -name '*.py' | head -60 && echo '---' && ls",
                repo_path,
            )
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Fix the following issue in the `{instance['repo']}` repository.\n\n"
                        f"{instance['problem_statement']}\n\n"
                        f"Repository cloned at `{instance['base_commit'][:12]}`. "
                        f"Python files (truncated):\n```\n{init_listing}\n```\n\n"
                        "Begin exploring and fixing the issue."
                    ),
                }
            ]

            no_bash_streak = 0
            for _step in range(self._max_react_steps):
                response = llm.generate_messages(
                    messages, system=system_prompt, max_tokens=4096
                )
                messages.append({"role": "assistant", "content": response})

                done_signaled = bool(re.search(r"\bDONE\b", response))
                codes = _extract_all_bash(response)

                if codes:
                    no_bash_streak = 0
                    outputs = []
                    for code in codes:
                        out = _run_bash(code, repo_path)
                        if len(out) > _MAX_OUTPUT_CHARS:
                            out = out[:_MAX_OUTPUT_CHARS] + "\n[OUTPUT TRUNCATED]"
                        outputs.append(out)
                    combined = "\n---\n".join(outputs)
                    messages.append({
                        "role": "user",
                        "content": f"Output:\n```\n{combined}\n```\n",
                    })
                    # Only honour DONE when there's an actual diff.
                    if done_signaled and _get_diff(repo_path):
                        break
                    continue

                # No bash block in this turn.
                if done_signaled and _get_diff(repo_path):
                    break
                no_bash_streak += 1
                if _get_diff(repo_path) or no_bash_streak >= 2:
                    break
                messages.append({
                    "role": "user",
                    "content": (
                        "No bash command was found. Please write your next step "
                        "inside a ```bash block. After editing, run `git diff` to "
                        "confirm changes, then output DONE."
                    ),
                })

            patch = _get_diff(repo_path)

        except Exception as exc:
            raise RuntimeError(f"Execution error: {type(exc).__name__}: {exc}") from exc

        shutil.rmtree(work_dir, ignore_errors=True)
        return _messages_to_trajectory(messages), patch

    def solve(self, task: dict, context: str, llm) -> tuple[str, dict]:
        try:
            if self._solver_backend == "official_prompt":
                trajectory, patch = self._solve_official_prompt(task, context, llm)
            else:
                trajectory, patch = self._solve_react(task, context, llm)
        except Exception as exc:
            return "", {
                "outcome": "failure",
                "score": 0.0,
                "details": str(exc),
            }

        if not patch:
            return trajectory, {
                "outcome": "failure",
                "score": 0.0,
                "details": "No patch extracted from the model response.",
            }

        instance = task["_swebench_instance"]
        self._save_prediction(instance["instance_id"], patch)

        if self._solve_only:
            saved = f" Saved to {self._predictions_path}." if self._predictions_path else ""
            return trajectory, {
                "outcome": "pending",
                "score": 0.0,
                "details": f"Patch generated ({len(patch)} chars). Evaluation is pending; use the official SWE-bench harness.{saved}",
            }

        result = self.evaluate(task, patch)
        return trajectory, result

    # ------------------------------------------------------------------
    # Benchmark interface — evaluate
    # ------------------------------------------------------------------

    def evaluate(self, task: dict, agent_output: str) -> dict:
        """
        Evaluate a patch string using the official swebench harness.

        With use_modal=True: evaluates on Modal cloud (no Docker needed).
        With use_modal=False: evaluates with local Docker.
        """
        instance = task["_swebench_instance"]
        try:
            return _evaluate_with_harness(
                instance, agent_output,
                run_id=self._model_name,
                dataset_name=self._dataset_name,
                split=self._split,
                model_name=self._model_name,
                use_modal=self._use_modal,
            )
        except Exception as exc:
            return {
                "outcome": "unknown",
                "score": 0.0,
                "details": f"Evaluation failed: {exc}",
            }
