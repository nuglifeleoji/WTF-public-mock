"""Terminal-Bench benchmark adapter backed by Harbor.

This adapter discovers Terminal-Bench task definitions from the official
task repository, then executes each task through `harbor trial start` so the
solve and evaluation loop happens inside Harbor's official harness.

To keep the memory update loop working, retrieved memory context is
prepended to a temporary copy of the task's `instruction.md` before the Harbor
trial starts. This mirrors how the other benchmark adapters inject retrieved
context into the current attempt without mutating the source task definition.

Setup:
    Install Harbor separately in your environment:
        pip install harbor
    or:
        uv tool install harbor

    Configure the benchmark to use a remote Harbor environment such as
    `modal`, `daytona`, or `e2b`.

See: https://harborframework.com/docs/tutorials/running-terminal-bench
"""

from __future__ import annotations

import json
import os
import re
import hashlib
import shutil
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any, Optional

from .base import Benchmark

_DEFAULT_REPO_URL = "https://github.com/harbor-framework/terminal-bench-2.git"
_DEFAULT_REPO_PATH = "data/terminalbench/terminal-bench-2"
_DEFAULT_RUN_OUTPUT_DIR = "data/terminalbench/runs"
_DEFAULT_ENVIRONMENT = "modal"
_DEFAULT_HARBOR_TIMEOUT_SEC = 60 * 60 * 4
_REMOTE_NON_MOUNTED_ENVIRONMENTS = {"modal", "daytona", "e2b", "runloop", "gke"}
_CODEX_LOG_FILTER_ENV_VAR = "WTF_FILTER_CODEX_AGENT_UPLOADS"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _truncate(text: str, max_chars: int = 400) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _sanitize_harbor_text(text: str) -> str:
    """Keep control bytes out of Harbor command arguments and ACE state."""
    return (text or "").replace("\x00", r"\x00")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return slug or "terminalbench-task"


def _is_task_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "task.toml").exists()
        and (path / "instruction.md").exists()
    )


def _discover_task_dirs(repo_path: Path) -> list[Path]:
    return sorted(
        [
            child
            for child in repo_path.iterdir()
            if child.is_dir() and not child.name.startswith(".") and _is_task_dir(child)
        ],
        key=lambda p: p.name,
    )


def _build_instruction_with_context(original_instruction: str, context: str) -> str:
    original_instruction = _sanitize_harbor_text(original_instruction)
    stripped_context = _sanitize_harbor_text(context).strip()
    if not stripped_context:
        return original_instruction
    return (
        "Additional guidance from prior terminal-bench attempts. Use this only when it "
        "helps and does not conflict with the task requirements.\n\n"
        f"{stripped_context}\n\n"
        "Original task instruction:\n\n"
        f"{original_instruction.strip()}\n"
    )


def _extract_primary_reward(result_payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    verifier_result = result_payload.get("verifier_result") or {}
    rewards = verifier_result.get("rewards")
    if isinstance(rewards, (int, float)):
        reward = float(rewards)
        return reward, {"reward": reward}
    if not isinstance(rewards, dict):
        return 0.0, {}

    numeric_rewards: dict[str, float] = {}
    for key, value in rewards.items():
        if isinstance(value, (int, float)):
            numeric_rewards[str(key)] = float(value)

    if "reward" in numeric_rewards:
        return numeric_rewards["reward"], numeric_rewards
    if len(numeric_rewards) == 1:
        only_key = next(iter(numeric_rewards))
        return numeric_rewards[only_key], numeric_rewards
    return 0.0, numeric_rewards


def _is_retryable_harbor_infra_failure(result_payload: dict[str, Any]) -> bool:
    exception_info = result_payload.get("exception_info")
    if not isinstance(exception_info, dict):
        return False
    return exception_info.get("exception_type") in {
        "AlreadyExistsError",
        "FilesystemExecutionError",
    }


def _trajectory_json_to_text(payload: dict[str, Any]) -> str:
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        return _sanitize_harbor_text(json.dumps(payload, indent=2, ensure_ascii=False))

    parts: list[str] = []
    for episode_idx, episode in enumerate(episodes):
        steps = episode.get("steps", []) if isinstance(episode, dict) else []
        if len(episodes) > 1:
            parts.append(f"EPISODE {episode_idx}")
        for step_idx, step in enumerate(steps):
            if not isinstance(step, dict):
                parts.append(f"STEP {step_idx}:\n{json.dumps(step, ensure_ascii=False)}")
                continue

            role = (
                step.get("source")
                or step.get("role")
                or step.get("actor")
                or step.get("type")
                or f"STEP_{step_idx}"
            )
            content_candidates = [
                step.get("content"),
                step.get("text"),
                step.get("message"),
                step.get("output"),
                step.get("observation"),
                step.get("summary"),
            ]
            content = next(
                (
                    str(candidate)
                    for candidate in content_candidates
                    if isinstance(candidate, (str, int, float)) and str(candidate).strip()
                ),
                "",
            )

            extra_lines: list[str] = []
            if isinstance(step.get("tool_name"), str) and step.get("tool_name"):
                extra_lines.append(f"tool_name={step['tool_name']}")
            if step.get("arguments") is not None:
                extra_lines.append(f"arguments={step['arguments']}")
            if step.get("result") is not None and not content:
                extra_lines.append(f"result={step['result']}")
            if not content and not extra_lines:
                compact = {
                    key: value
                    for key, value in step.items()
                    if key not in {"step_id", "timestamp"}
                }
                content = json.dumps(compact, ensure_ascii=False)

            block = "\n".join(line for line in [content, *extra_lines] if line)
            parts.append(f"{str(role).upper()}:\n{block}".rstrip())

    return _sanitize_harbor_text("\n\n".join(part for part in parts if part.strip()))


def _build_logs_trajectory(trial_dir: Path) -> str:
    log_paths = [
        trial_dir / "trial.log",
        trial_dir / "exception.txt",
        trial_dir / "agent" / "codex.txt",
        trial_dir / "verifier" / "test-stdout.txt",
        trial_dir / "verifier" / "test-stderr.txt",
    ]
    parts: list[str] = []
    for path in log_paths:
        if path.exists():
            parts.append(f"{path.parent.name.upper()}::{path.name}\n{_read_text(path)}")
    return _sanitize_harbor_text("\n\n".join(parts))


class TerminalBenchBenchmark(Benchmark):
    """Benchmark adapter for Terminal-Bench tasks executed through Harbor."""

    def __init__(
        self,
        repo_path: str = _DEFAULT_REPO_PATH,
        repo_url: Optional[str] = _DEFAULT_REPO_URL,
        refresh_repo: bool = False,
        max_tasks: Optional[int] = None,
        task_ids: Optional[list[str]] = None,
        environment: str = _DEFAULT_ENVIRONMENT,
        agent_name: str = "claude-code",
        model_name: Optional[str] = None,
        run_output_dir: Optional[str] = _DEFAULT_RUN_OUTPUT_DIR,
        run_id: str = "terminalbench_run",
        timeout_multiplier: float = 1.0,
        success_threshold: float = 1.0,
        harbor_bin: str = "harbor",
        harbor_timeout_sec: int = _DEFAULT_HARBOR_TIMEOUT_SEC,
        harbor_infra_retries: int = 2,
    ):
        self.repo_path = Path(repo_path)
        self.repo_url = repo_url
        self.refresh_repo = refresh_repo
        self.max_tasks = max_tasks
        self.task_ids = list(task_ids or [])
        self.environment = environment
        self.agent_name = agent_name
        self.model_name = model_name
        self.timeout_multiplier = timeout_multiplier
        self.success_threshold = success_threshold
        self.harbor_bin = harbor_bin
        self.harbor_timeout_sec = harbor_timeout_sec
        self.harbor_infra_retries = max(0, int(harbor_infra_retries))
        self.run_id = run_id
        self._trial_run_hash = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:8]
        self._tasks: Optional[list[dict[str, Any]]] = None

        self.run_output_dir: Optional[Path]
        if run_output_dir:
            self.run_output_dir = Path(run_output_dir) / run_id
            self.run_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_output_dir = None

    def _ensure_repo(self) -> Path:
        if self.repo_path.exists():
            if self.refresh_repo and (self.repo_path / ".git").exists():
                subprocess.run(
                    ["git", "-C", str(self.repo_path), "pull", "--ff-only"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
            return self.repo_path

        if not self.repo_url:
            raise FileNotFoundError(
                f"Terminal-Bench task repo not found at {self.repo_path} and no repo_url was provided."
            )

        self.repo_path.parent.mkdir(parents=True, exist_ok=True)
        clone = subprocess.run(
            ["git", "clone", "--depth", "1", self.repo_url, str(self.repo_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if clone.returncode != 0:
            raise RuntimeError(
                "Failed to clone Terminal-Bench task repo: "
                f"{_truncate(clone.stderr or clone.stdout, 600)}"
            )
        return self.repo_path

    def get_tasks(self) -> list[dict]:
        if self._tasks is not None:
            return self._tasks

        repo_path = self._ensure_repo()
        task_dirs = _discover_task_dirs(repo_path)
        selected_ids = set(self.task_ids)
        tasks: list[dict[str, Any]] = []

        for task_dir in task_dirs:
            task_id = str(task_dir.relative_to(repo_path))
            if selected_ids and task_id not in selected_ids:
                continue

            task_meta = tomllib.loads(_read_text(task_dir / "task.toml"))
            metadata = task_meta.get("metadata") or {}
            tasks.append(
                {
                    "task_id": task_id,
                    "description": _read_text(task_dir / "instruction.md").strip(),
                    "difficulty": metadata.get("difficulty"),
                    "category": metadata.get("category"),
                    "tags": list(metadata.get("tags") or []),
                    "_task_path": str(task_dir),
                }
            )
            if not selected_ids and self.max_tasks is not None and len(tasks) >= self.max_tasks:
                break

        if selected_ids:
            found_ids = {task["task_id"] for task in tasks}
            missing = sorted(selected_ids - found_ids)
            if missing:
                raise ValueError(
                    "Terminal-Bench task_ids not found in the task repo: "
                    + ", ".join(missing)
                )
            by_id = {task["task_id"]: task for task in tasks}
            tasks = [by_id[task_id] for task_id in self.task_ids]

        self._tasks = tasks
        return self._tasks

    def evaluate(self, task: dict, agent_output: str) -> dict:
        return {
            "outcome": "unknown",
            "score": 0.0,
            "details": "TerminalBenchBenchmark requires solve() for Harbor-backed evaluation.",
        }

    def _resolve_task_path(self, task: dict) -> Path:
        task_path = task.get("_task_path")
        if task_path:
            return Path(task_path)
        return self._ensure_repo() / task["task_id"]

    def _prepare_task_copy(self, task_path: Path, context: str) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
        if not (context or "").strip():
            return task_path, None

        temp_dir = tempfile.TemporaryDirectory(prefix="terminalbench_task_")
        staged_task = Path(temp_dir.name) / task_path.name
        shutil.copytree(task_path, staged_task)
        instruction_path = staged_task / "instruction.md"
        original_instruction = _read_text(instruction_path)
        instruction_path.write_text(
            _build_instruction_with_context(original_instruction, context),
            encoding="utf-8",
        )
        return staged_task, temp_dir

    def _locate_trial_dir(self, trials_dir: Path, trial_name: str) -> Path:
        direct = trials_dir / trial_name
        if direct.exists():
            return direct

        candidates = sorted(
            [
                child
                for child in trials_dir.iterdir()
                if child.is_dir() and child.name.startswith(trial_name)
            ],
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"Harbor trial output directory not found in {trials_dir} for {trial_name!r}."
            )
        return candidates[-1]

    def _resolved_harbor_executable(self) -> Path | None:
        harbor_path = shutil.which(self.harbor_bin)
        if harbor_path:
            return Path(harbor_path).resolve()

        candidate = Path(self.harbor_bin)
        if candidate.exists():
            return candidate.resolve()
        return None

    def _should_filter_remote_codex_agent_logs(self) -> bool:
        return (
            self.agent_name == "codex"
            and self.environment in _REMOTE_NON_MOUNTED_ENVIRONMENTS
        )

    def _wrap_harbor_command(self, cmd: list[str]) -> tuple[list[str], dict[str, str] | None]:
        if not self._should_filter_remote_codex_agent_logs():
            return cmd, None

        harbor_executable = self._resolved_harbor_executable()
        if harbor_executable is None:
            return cmd, None

        harbor_python = harbor_executable.with_name("python")
        wrapper_path = Path(__file__).resolve().parents[1] / "helpers" / "harbor_terminalbench_wrapper.py"
        if not harbor_python.exists() or not wrapper_path.exists():
            return cmd, None

        env = os.environ.copy()
        env[_CODEX_LOG_FILTER_ENV_VAR] = "1"
        return [str(harbor_python), str(wrapper_path), *cmd[1:]], env

    def _load_trajectory(self, trial_dir: Path) -> str:
        trajectory_candidates = [
            trial_dir / "agent" / "trajectory.json",
            trial_dir / "trajectory.json",
        ]
        for trajectory_path in trajectory_candidates:
            if trajectory_path.exists():
                return _trajectory_json_to_text(json.loads(_read_text(trajectory_path)))
        return _build_logs_trajectory(trial_dir)

    def _build_result(self, *, task_id: str, trial_dir: Path, result_payload: dict[str, Any]) -> dict:
        reward, reward_breakdown = _extract_primary_reward(result_payload)
        exception_info = result_payload.get("exception_info")
        verifier_result = result_payload.get("verifier_result") or {}
        verifier_stdout_path = trial_dir / "verifier" / "test-stdout.txt"
        verifier_stderr_path = trial_dir / "verifier" / "test-stderr.txt"
        verifier_stdout = _truncate(
            _read_text(verifier_stdout_path) if verifier_stdout_path.exists() else str(verifier_result.get("stdout") or ""),
            500,
        )
        verifier_stderr = _truncate(
            _read_text(verifier_stderr_path) if verifier_stderr_path.exists() else str(verifier_result.get("stderr") or ""),
            300,
        )

        detail_parts = [
            f"terminalbench harness ({self.environment})",
            f"reward={reward:.3f}",
        ]
        if reward_breakdown:
            detail_parts.append(f"rewards={json.dumps(reward_breakdown, ensure_ascii=False, sort_keys=True)}")
        if exception_info:
            detail_parts.append(
                "exception="
                + _truncate(json.dumps(exception_info, ensure_ascii=False, sort_keys=True), 300)
            )
        if verifier_stdout:
            detail_parts.append(f"verifier_stdout={verifier_stdout}")
        if verifier_stderr:
            detail_parts.append(f"verifier_stderr={verifier_stderr}")

        # Harbor reports the agent's token usage in agent_result. cost_usd in
        # the harbor payload is sometimes None (Codex doesn't fill it for all
        # providers), so we recompute from token counts using our pricing
        # table to keep the field comparable across runs/models.
        agent_result = result_payload.get("agent_result") or {}
        n_in    = int(agent_result.get("n_input_tokens", 0) or 0)
        n_out   = int(agent_result.get("n_output_tokens", 0) or 0)
        n_cache = int(agent_result.get("n_cache_tokens", 0) or 0)
        agent_cost_reported = agent_result.get("cost_usd")
        # Cache tokens are typically *cached input reads* in Harbor's accounting.
        # Build an Anthropic-style usage dict and let pricing.compute_cost
        # apply the cached-input rate.
        from runtime.pricing import compute_cost as _compute_cost
        agent_model = self.model_name or self.agent_name
        usage_for_cost = {
            "input_tokens":            max(0, n_in - n_cache),
            "output_tokens":           n_out,
            "cache_read_input_tokens": n_cache,
        }
        agent_cost_calc = _compute_cost(agent_model, usage_for_cost)
        agent_cost = (
            float(agent_cost_reported)
            if isinstance(agent_cost_reported, (int, float))
            else agent_cost_calc
        )

        return {
            "outcome": "success" if reward >= self.success_threshold and not exception_info else "failure",
            "score": reward,
            "details": " ".join(detail_parts),
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "trial_dir": str(trial_dir),
            "agent_name": self.agent_name,
            "environment": self.environment,
            "task_name": task_id,
            "cost_usd": agent_cost,
            "cost_breakdown": {
                "agent_cost_usd":          agent_cost,
                "agent_input_tokens":      n_in,
                "agent_output_tokens":     n_out,
                "agent_cache_tokens":      n_cache,
                "agent_cost_source":       "harbor" if isinstance(agent_cost_reported, (int, float)) else "computed_from_tokens",
                "pricing_model":           agent_model,
            },
        }

    def solve(self, task: dict, context: str, llm) -> tuple[str, dict]:
        task_path = self._resolve_task_path(task)
        if not task_path.exists():
            return "", {
                "outcome": "failure",
                "score": 0.0,
                "details": f"Terminal-Bench task path not found: {task_path}",
            }

        effective_model = self.model_name or getattr(llm, "model", None)
        if not effective_model:
            return "", {
                "outcome": "failure",
                "score": 0.0,
                "details": "Terminal-Bench requires a configured Harbor model_name or llm.model.",
            }

        staged_task_path, temp_dir = self._prepare_task_copy(task_path, context)
        base_trials_dir = self.run_output_dir or Path(tempfile.mkdtemp(prefix="terminalbench_trials_"))
        base_trials_dir.mkdir(parents=True, exist_ok=True)

        try:
            attempts = self.harbor_infra_retries + 1
            last_result: tuple[str, dict] | None = None
            for attempt in range(attempts):
                trial_name = f"{_slugify(task['task_id'])}-{self._trial_run_hash}-{time.time_ns()}"
                cmd = [
                    self.harbor_bin,
                    "trial",
                    "start",
                    "--path",
                    str(staged_task_path),
                    "--agent",
                    self.agent_name,
                    "--model",
                    str(effective_model),
                    "--environment-type",
                    self.environment,
                    "--trial-name",
                    trial_name,
                    "--trials-dir",
                    str(base_trials_dir),
                ]
                if self.timeout_multiplier and self.timeout_multiplier != 1.0:
                    cmd += ["--timeout-multiplier", str(self.timeout_multiplier)]

                try:
                    wrapped_cmd, wrapped_env = self._wrap_harbor_command(cmd)
                    proc = subprocess.run(
                        wrapped_cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.harbor_timeout_sec,
                        env=wrapped_env,
                    )
                except FileNotFoundError as exc:
                    return "", {
                        "outcome": "failure",
                        "score": 0.0,
                        "details": (
                            f"Terminal-Bench execution failed: {exc}. "
                            f"Install Harbor and ensure `{self.harbor_bin}` is on PATH."
                        ),
                    }

                trial_dir = self._locate_trial_dir(base_trials_dir, trial_name)
                result_path = trial_dir / "result.json"
                if not result_path.exists():
                    raise FileNotFoundError(f"Harbor result file not found: {result_path}")
                result_payload = json.loads(_read_text(result_path))
                trajectory = self._load_trajectory(trial_dir)

                result = self._build_result(
                    task_id=task["task_id"],
                    trial_dir=trial_dir,
                    result_payload=result_payload,
                )
                if proc.returncode != 0:
                    result["outcome"] = "failure"
                    result["details"] += (
                        " harbor_cli="
                        + _truncate((proc.stderr or proc.stdout or "").strip(), 400)
                    )

                last_result = (trajectory, result)
                if (
                    attempt < attempts - 1
                    and _is_retryable_harbor_infra_failure(result_payload)
                ):
                    delay = min(60, 15 * (attempt + 1))
                    print(
                        "[TerminalBench] Retrying Harbor infrastructure failure "
                        f"for task={task['task_id']} attempt={attempt + 1}/{attempts} "
                        f"after {delay}s"
                    )
                    time.sleep(delay)
                    continue
                return trajectory, result

            if last_result is not None:
                return last_result
            return "", {
                "outcome": "failure",
                "score": 0.0,
                "details": "Terminal-Bench execution produced no result.",
            }
        except Exception as exc:
            return "", {
                "outcome": "failure",
                "score": 0.0,
                "details": f"Execution error: {type(exc).__name__}: {exc}",
            }
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
