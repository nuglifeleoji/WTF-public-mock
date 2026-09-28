"""AppWorld benchmark adapter.

Setup:
    pip install appworld==0.1.3.post1 jinja2
    appworld install
    appworld download data

AppWorld is a controlled environment of simulated apps (email, calendar, etc.)
where agents must complete multi-step tasks by executing Python code against
a live AppWorld server.

Usage notes:
- Each task is a self-contained scenario with a natural-language instruction.
- Evaluation executes the agent's generated code against the AppWorld environment
  and checks whether specified conditions hold.
- Tasks are run sequentially; the environment is reset between tasks.
- Set APPWORLD_DATA_DIR env var to point at your downloaded data, or pass
  data_dir explicitly.
- Available splits depend on the installed AppWorld data bundle. Common values
  include train/dev plus benchmark-specific test splits such as test_normal and
  test_challenge.

See: https://appworld.dev / the appworld Python package for full docs.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from .base import Benchmark

# ---------------------------------------------------------------------------
# ReAct constants (match reference implementation)
# ---------------------------------------------------------------------------

_MAX_REACT_STEPS = 40
_MAX_OUTPUT_CHARS = 20_000   # truncate AppWorld execution output at this length

# ---------------------------------------------------------------------------
# Regex for extracting Python code blocks (mirrors reference implementation)
# ---------------------------------------------------------------------------

_FULL_CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_PARTIAL_CODE_RE = re.compile(r".*```python\n(.*)", re.DOTALL)

# ---------------------------------------------------------------------------
# Official prompt helpers
# ---------------------------------------------------------------------------

_MAX_PROMPT_HISTORY_CHARS = 400_000


def _load_official_prompt_template() -> str:
    """Load the experiment's bundled prompt, or an explicit user override."""
    override = os.environ.get("APPWORLD_REACT_PROMPT_PATH")
    path = Path(override) if override else (
        Path(__file__).resolve().parents[1] / "assets" / "appworld_react_generator_prompt.txt"
    )
    return path.read_text(encoding="utf-8").lstrip()


def _text_to_messages(input_str: str) -> list[dict]:
    messages_json: list[dict] = []
    last_start = 0
    for m in re.finditer(r"(USER|ASSISTANT|SYSTEM):\n", input_str, flags=re.IGNORECASE):
        last_end = m.span()[0]
        if not messages_json:
            if last_end != 0:
                raise ValueError(
                    f"Start of the prompt has no assigned role: {input_str[:last_end]}"
                )
        else:
            messages_json[-1]["content"] = input_str[last_start:last_end]
        role = m.group(1).lower()
        messages_json.append({"role": role, "content": None})
        last_start = m.span()[1]
    if not messages_json:
        raise ValueError("AppWorld prompt contains no role messages.")
    messages_json[-1]["content"] = input_str[last_start:]
    return messages_json


def _messages_to_text(messages: list[dict]) -> str:
    output_str = ""
    for message in messages:
        role = message["role"]
        if role == "system":
            output_str += "SYSTEM:\n" + message["content"]
        elif role == "assistant":
            output_str += "ASSISTANT:\n" + message["content"]
        elif role == "user":
            output_str += "USER:\n" + message["content"]
        else:
            raise ValueError(f"Unknown message role {role!r}")
    return output_str


def _trimmed_messages(messages: list[dict], num_instruction_messages: int) -> list[dict]:
    import copy

    trimmed = copy.deepcopy(messages)
    pre_messages = trimmed[: num_instruction_messages - 1]
    post_messages = trimmed[num_instruction_messages - 1 :]
    output_str = _messages_to_text(post_messages)
    remove_prefix = output_str[: output_str.index("Task: ") + 6]
    output_str = output_str.removeprefix(remove_prefix)
    observation_index = 0

    while len(output_str) > _MAX_PROMPT_HISTORY_CHARS:
        found_block = False
        if observation_index < len(post_messages) - 5:
            for message_index, message in enumerate(post_messages[observation_index:]):
                if message["role"] == "user" and message["content"].startswith("Output:"):
                    message["content"] = "Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
                    found_block = True
                    observation_index += message_index + 1
                    break
            if not found_block:
                observation_index = len(post_messages)

        if not found_block and len(post_messages) >= 3:
            first_post_message = copy.deepcopy(post_messages[0])
            if not first_post_message["content"].endswith("[TRIMMED HISTORY]\n\n"):
                first_post_message["content"] += "[TRIMMED HISTORY]\n\n"
            post_messages = [first_post_message] + post_messages[2:]
            found_block = True

        if not found_block:
            raise ValueError("No blocks found to trim from AppWorld history.")

        output_str = _messages_to_text(post_messages).removeprefix(remove_prefix)

    return pre_messages + post_messages


def _build_react_messages(appworld_task, playbook: str) -> tuple[list[dict], int]:
    from jinja2 import Template

    template = Template(_load_official_prompt_template())
    app_descriptions = json.dumps(
        [{"name": k, "description": v} for (k, v) in appworld_task.app_descriptions.items()],
        indent=1,
    )
    output_str = template.render(
        {
            "input_str": appworld_task.instruction,
            "main_user": appworld_task.supervisor,
            "app_descriptions": app_descriptions,
            "playbook": playbook or "",
        }
    ).rstrip() + "\n\n"
    messages = _text_to_messages(output_str)
    return messages, len(messages)


# ---------------------------------------------------------------------------
# Code extraction (mirrors reference extract_code_and_fix_content logic)
# ---------------------------------------------------------------------------

def _extract_code_and_fix(text: str) -> tuple[str, str]:
    """
    Extract the first Python code block from the LLM response.
    Returns (code, fixed_text) where fixed_text has a closing ``` if the
    block was partial (model was cut off before the closing fence).
    """
    if not text:
        return "", ""

    # Full code block — take only the first one (ignore_multiple_calls=True in ref)
    m = _FULL_CODE_RE.search(text)
    if m:
        code = m.group(1).strip()
        fixed = text[: m.end()]
        return code, fixed

    # Partial block (no closing ```)
    m = _PARTIAL_CODE_RE.match(text)
    if m:
        code = m.group(1).strip()
        fixed = text if text.endswith("\n") else text + "\n"
        fixed = fixed + "```"
        return code, fixed

    return "", text


# ---------------------------------------------------------------------------
# Result helper (shared between solve() and evaluate())
# ---------------------------------------------------------------------------

def _messages_to_trajectory(messages: list[dict]) -> str:
    return _messages_to_text(messages)


def _tracker_to_result(tracker) -> dict:
    if isinstance(tracker, tuple):
        tracker = tracker[0]
    success = tracker.success
    num_tests = tracker.num_tests or 1
    score = tracker.pass_count / num_tests
    failures = tracker.to_dict().get("failures", [])
    details = "; ".join(f["requirement"] for f in failures) if failures else ""
    return {
        "outcome": "success" if success else "failure",
        "score": score,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Benchmark subclass
# ---------------------------------------------------------------------------

class AppWorldBenchmark(Benchmark):
    """
    Adapter for the AppWorld benchmark.

    Parameters
    ----------
    split:
        Which task split to use. Split names come from the installed
        ``data/datasets/*.txt`` files, for example ``train``, ``dev``,
        ``test_normal``, or ``test_challenge``.
    data_dir:
        Path to the AppWorld data directory. Defaults to APPWORLD_DATA_DIR env var,
        then to the appworld package default.
    max_tasks:
        Cap the number of tasks (useful for quick smoke tests). None = all tasks.
    task_ids:
        Explicit list of task IDs to run. Overrides max_tasks if provided.
        This is useful for targeted smoke runs via config or ``--set``.
    max_react_steps:
        Maximum number of code-execution turns per task in the ReAct loop.
    """

    def __init__(
        self,
        split: str = "dev",
        data_dir: Optional[str] = None,
        max_tasks: Optional[int] = None,
        task_ids: Optional[list[str]] = None,
        max_react_steps: int = _MAX_REACT_STEPS,
        experiment_name: Optional[str] = None,
    ):
        if max_tasks is not None and max_tasks < 0:
            raise ValueError("max_tasks must be non-negative.")
        if max_react_steps < 1:
            raise ValueError("max_react_steps must be positive.")
        self._split = split
        self._max_tasks = max_tasks
        self._task_ids = list(task_ids) if task_ids is not None else None
        self._tasks: Optional[list[dict]] = None
        self._max_react_steps = max_react_steps
        self._experiment_name = experiment_name

        self._data_dir = data_dir or os.environ.get("APPWORLD_DATA_DIR")

    def _configure_paths(self):
        from appworld.common.path_store import path_store

        if self._data_dir:
            data_path = Path(self._data_dir).expanduser().resolve()
            if data_path.name != "data":
                raise ValueError("AppWorld data_dir must point to the downloaded 'data' directory.")
            path_store.update_root(str(data_path.parent))
        return path_store

    @property
    def _appworld_kwargs(self) -> dict:
        kwargs: dict = {}
        if self._experiment_name:
            kwargs["experiment_name"] = self._experiment_name
        return kwargs

    # ------------------------------------------------------------------
    # Benchmark interface — get_tasks
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[dict]:
        if self._tasks is not None:
            return self._tasks

        path_store = self._configure_paths()
        from appworld.task import Task  # type: ignore[import]

        split_file = os.path.join(path_store.data, "datasets", f"{self._split}.txt")
        with open(split_file) as f:
            all_ids = [line.strip() for line in f if line.strip()]

        if self._task_ids is not None:
            missing = set(self._task_ids) - set(all_ids)
            if missing:
                raise ValueError(f"AppWorld task IDs absent from {self._split!r}: {sorted(missing)}")
            all_ids = self._task_ids
        elif self._max_tasks is not None:
            all_ids = all_ids[: self._max_tasks]

        tasks_raw: list[Task] = [Task.load(tid) for tid in all_ids]

        self._tasks = [
            {
                "task_id": t.id,
                "description": t.instruction,
                "split": self._split,
                # Preserve the raw Task object for evaluate() and solve()
                "_appworld_task": t,
            }
            for t in tasks_raw
        ]
        return self._tasks

    # ------------------------------------------------------------------
    # Benchmark interface — solve (multi-turn ReAct loop)
    # ------------------------------------------------------------------

    def solve(self, task: dict, context: str, llm) -> tuple[str, dict]:
        """
        Run the ReAct loop for one AppWorld task.

        Each turn:
          1. LLM generates a response (reasoning + ```python``` code block)
          2. Code is executed via app.execute()
          3. Output is fed back as the next user message
          4. Repeat until task_completed() or max steps reached

        Returns (trajectory_str, result_dict).
        """
        from appworld import AppWorld  # type: ignore[import]

        self._configure_paths()
        appworld_task = task["_appworld_task"]
        messages, num_instruction_messages = _build_react_messages(appworld_task, context)
        execution_output = None

        try:
            with AppWorld(task_id=appworld_task.id, **self._appworld_kwargs) as app:
                for _step in range(self._max_react_steps):
                    if execution_output is not None:
                        truncated = execution_output
                        if len(truncated) > _MAX_OUTPUT_CHARS:
                            truncated = (
                                truncated[:_MAX_OUTPUT_CHARS]
                                + "\n[REST NOT SHOWN FOR BREVITY]"
                            )
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Output:\n```\n{truncated}\n```\n\n",
                            }
                        )

                    response = llm.generate_messages(
                        _trimmed_messages(messages, num_instruction_messages),
                        max_tokens=4096,
                    )
                    code, fixed_response = _extract_code_and_fix(response)
                    messages.append({"role": "assistant", "content": fixed_response + "\n\n"})

                    execution_output = app.execute(code)

                    if app.task_completed():
                        break

                result = app.evaluate()
                tracker = result[0] if isinstance(result, tuple) else result

        except Exception as exc:
            trajectory = _messages_to_trajectory(messages)
            return trajectory, {
                "outcome": "failure",
                "score": 0.0,
                "details": f"Execution error: {type(exc).__name__}: {exc}",
            }

        return _messages_to_trajectory(messages), _tracker_to_result(tracker)

    # ------------------------------------------------------------------
    # Benchmark interface — evaluate (single-shot fallback)
    # ------------------------------------------------------------------

    def evaluate(self, task: dict, agent_output: str) -> dict:
        """
        Single-shot evaluation: execute agent_output once and score.
        Used by Benchmark.solve() default; AppWorldBenchmark.solve() bypasses this.
        """
        from appworld import AppWorld  # type: ignore[import]

        self._configure_paths()
        appworld_task = task["_appworld_task"]
        code, _ = _extract_code_and_fix(agent_output)
        if not code:
            code = agent_output.strip()

        try:
            with AppWorld(task_id=appworld_task.id, **self._appworld_kwargs) as app:
                app.execute(code)
                tracker = app.evaluate()
        except Exception as exc:
            return {
                "outcome": "failure",
                "score": 0.0,
                "details": f"Execution error: {type(exc).__name__}: {exc}",
            }

        return _tracker_to_result(tracker)
