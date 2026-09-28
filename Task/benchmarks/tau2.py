"""Tau2-bench benchmark adapter.

tau2-bench evaluates customer service agents across multiple
domains (airline, retail, telecom, mock).  The agent interacts with a
simulated user and uses domain-specific tools.

Setup:
    Clone https://github.com/sierra-research/tau2-bench and check out
    0ed2fd8d830a20657d89ae9c2efcc94838aa7129, then pip install -e /path/to/tau2-bench.
    Set TAU2_DATA_DIR to that checkout's data directory.

Usage:
    Task descriptors select the domain and task IDs.
    Available domains: airline, retail, telecom, mock

See: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import os
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

from .base import Benchmark

_CONTEXT: ContextVar[str] = ContextVar("tau2_retrieved_context", default="")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_litellm_model(model: str) -> str:
    """Convert a plain model name to litellm provider/model format."""
    if "/" in model:
        return model
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai/{model}"
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    return model


def _messages_to_trajectory(sim_run) -> str:
    """Convert a tau2 SimulationRun to a trajectory string.

    Mirrors the USER:/ASSISTANT: format used by other benchmarks so the
    reflection loop sees a consistent format.  Tool calls are inlined under
    the ASSISTANT block; tool responses get their own TOOL block.
    """
    role_prefix = {
        "user": "USER",
        "assistant": "ASSISTANT",
        "system": "SYSTEM",
        "tool": "TOOL",
    }
    parts = []
    for msg in (sim_run.messages or []):
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        prefix = role_prefix.get(role, role.upper())

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            lines = []
            if msg.content:
                lines.append(str(msg.content))
            for tc in msg.tool_calls:
                args = getattr(tc, "arguments", "") or ""
                lines.append(f"TOOL_CALL: {tc.name}({args})")
            parts.append(f"{prefix}:\n" + "\n".join(lines))
        elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
            content = str(msg.content or "")[:500]
            parts.append(f"TOOL:\n{content}")
        else:
            content = str(msg.content or "")[:1000]
            parts.append(f"{prefix}:\n{content}")
    return "\n\n".join(parts)


def _result_details(sim_run, domain: str, task_id) -> str:
    """Build a detailed result string from tau2 RewardInfo.

    Field names in the pinned tau2 checkout:
      RewardInfo.db_check        → DBCheck(db_match, db_reward)
      RewardInfo.communicate_checks → list[CommunicateCheck(info, met, justification)]
      RewardInfo.env_assertions  → list[EnvAssertionCheck(env_assertion, met, reward)]
      RewardInfo.reward_breakdown → dict[RewardType, float]
    """
    ri = sim_run.reward_info
    if ri is None:
        return f"domain={domain} task_id={task_id} reward=unknown"

    parts = [f"domain={domain} task_id={task_id} reward={ri.reward:.3f}"]

    if ri.reward_breakdown:
        breakdown = " ".join(f"{k}={v:.3f}" for k, v in ri.reward_breakdown.items())
        parts.append(f"breakdown=({breakdown})")

    failures: list[str] = []

    # Database state check
    if ri.db_check is not None and not ri.db_check.db_match:
        failures.append(f"db_check failed (reward={ri.db_check.db_reward:.3f})")

    # Communication checks
    for chk in (ri.communicate_checks or []):
        if not chk.met:
            failures.append(f"communicate: {chk.info}")

    # Environment assertion checks
    for chk in (ri.env_assertions or []):
        if not chk.met:
            assertion_str = str(getattr(chk.env_assertion, "assertion", chk.env_assertion))
            failures.append(f"assertion: {assertion_str}")

    if failures:
        parts.append("failures=(" + "; ".join(failures) + ")")

    return " ".join(parts)


def _resolve_task_set_name(domain: str, task_split_name: Optional[str]) -> str:
    """Map the local split knob onto the official tau2 task-set registry."""
    from tau2.registry import registry

    if not task_split_name or task_split_name == "base":
        return domain

    available = set(registry.get_task_sets())
    candidates = [
        task_split_name,
        f"{domain}_{task_split_name}",
    ]
    if domain == "telecom" and task_split_name == "workflow":
        candidates.insert(0, "telecom-workflow")

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise ValueError(
        f"Unsupported tau2 task split {task_split_name!r} for domain {domain!r}. "
        f"Available task sets: {sorted(available)}"
    )


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class Tau2Benchmark(Benchmark):
    """Benchmark adapter for tau2-bench domains.

    Parameters
    ----------
    domain:
        Either a single domain name ("retail", "airline", "telecom", "mock"),
        or a dict mapping domain → per-domain task count for a single mixed run
        (e.g. ``{"airline": 33, "retail": 34, "telecom": 33}``).
    num_tasks:
        For single-domain mode: number of tasks to load (None = all tasks).
        Ignored when ``domain`` is a dict (the dict's values control counts).
    task_split_name:
        Task split to use ("base" is the default stable split matching the
        original τ-bench structure).
    max_steps:
        Maximum agent turns per simulation.
    user_model:
        LiteLLM model string for the user simulator.
    data_dir:
        Path to tau2 data directory.  Falls back to TAU2_DATA_DIR env var.
    """

    def __init__(
        self,
        domain="retail",
        num_tasks: Optional[int] = None,
        task_split_name: Optional[str] = "base",
        max_steps: int = 30,
        user_model: str = "openai/gpt-4o",
        data_dir: Optional[str] = None,
        task_ids: Optional[list[str]] = None,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if num_tasks is not None and num_tasks < 0:
            raise ValueError("num_tasks must be non-negative.")
        # task_ids overrides the (domain, num_tasks) selection. Each id may be
        # either bare ("0") and combined with `domain` to form "domain/0", or
        # already prefixed ("airline/0"). Multi-domain runs require the
        # prefixed form so we know which task set to load each from.
        self._task_ids: Optional[list[tuple[str, str]]] = None
        if task_ids is not None:
            self._task_ids = []
            for tid in task_ids:
                if not isinstance(tid, str) or not tid.strip():
                    raise ValueError("tau2 task IDs must be non-empty strings.")
                if "/" in tid:
                    dom, raw = tid.split("/", 1)
                else:
                    if isinstance(domain, dict):
                        raise ValueError(
                            f"tau2_task_ids entry {tid!r} has no domain prefix; "
                            "multi-domain runs need 'domain/id'"
                        )
                    dom, raw = (domain if isinstance(domain, str) else "retail"), tid
                if not dom or not raw:
                    raise ValueError(f"Invalid tau2 task ID: {tid!r}")
                self._task_ids.append((dom, raw))
            # Domain order preserved by first appearance
            seen: list[str] = []
            for dom, _ in self._task_ids:
                if dom not in seen:
                    seen.append(dom)
            self._domain_counts = [(d, None) for d in seen]
        elif isinstance(domain, dict):
            if not domain or any(n is not None and n < 0 for n in domain.values()):
                raise ValueError("domain counts must be non-empty and non-negative.")
            self._domain_counts = [(d, n) for d, n in domain.items()]
        else:
            self._domain_counts = [(domain, num_tasks)]
        self.domain = self._domain_counts[0][0] if self._domain_counts else "retail"
        self.task_split_name = task_split_name
        self.max_steps = max_steps
        self.user_model = user_model

        # tau2 reads TAU2_DATA_DIR at import-time; set it before the first import.
        self._data_dir = data_dir or os.environ.get("TAU2_DATA_DIR")

        self._tasks: Optional[list[dict]] = None

    def _configure_paths(self) -> None:
        """tau2 fixes its data path at first import; reject later path changes."""
        if not self._data_dir:
            return
        resolved = Path(self._data_dir).expanduser().resolve()
        imported = sys.modules.get("tau2.utils.utils")
        if imported is not None and Path(imported.DATA_DIR).resolve() != resolved:
            raise ValueError("tau2 was imported with a different data_dir; start a new process.")
        os.environ["TAU2_DATA_DIR"] = str(resolved)

    # ------------------------------------------------------------------
    # Benchmark interface — get_tasks
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[dict]:
        if self._tasks is not None:
            return self._tasks

        self._configure_paths()
        from tau2.run import get_tasks as _get_tasks

        tasks: list[dict] = []
        if self._task_ids is not None:
            # Pinned task ids — fetch each domain's requested ids and emit in
            # the order the user provided.
            from collections import defaultdict
            by_dom: dict[str, list[str]] = defaultdict(list)
            for dom, raw in self._task_ids:
                by_dom[dom].append(raw)
            tau2_by_id: dict[tuple[str, str], object] = {}
            for dom, ids in by_dom.items():
                fetched = _get_tasks(
                    _resolve_task_set_name(dom, self.task_split_name),
                    task_ids=list(dict.fromkeys(ids)),
                )
                for t in fetched:
                    tau2_by_id[(dom, str(t.id))] = t
            for dom, raw in self._task_ids:
                t = tau2_by_id.get((dom, raw))
                if t is None:
                    raise ValueError(f"tau2 task not found: {dom}/{raw}")
                tasks.append({
                    "task_id": f"{dom}/{t.id}",
                    "description": str(getattr(t.user_scenario, "instructions", "")),
                    "split": self.task_split_name,
                    "_tau2_task": t,
                    "_domain": dom,
                })
        else:
            for dom, n in self._domain_counts:
                tau2_tasks = _get_tasks(
                    _resolve_task_set_name(dom, self.task_split_name),
                    num_tasks=n,
                )
                for t in tau2_tasks:
                    tasks.append({
                        "task_id": f"{dom}/{t.id}",
                        "description": str(getattr(t.user_scenario, "instructions", "")),
                        "split": self.task_split_name,
                        "_tau2_task": t,
                        "_domain": dom,
                    })
        self._tasks = tasks
        return self._tasks

    # ------------------------------------------------------------------
    # Benchmark interface — evaluate (stub; real evaluation is in solve)
    # ------------------------------------------------------------------

    def evaluate(self, task: dict, agent_output: str) -> dict:
        return {
            "outcome": "unknown",
            "score": 0.0,
            "details": "Tau2Benchmark requires solve() for proper evaluation.",
        }

    # ------------------------------------------------------------------
    # Benchmark interface — solve
    # ------------------------------------------------------------------

    def solve(self, task: dict, context: str, llm) -> tuple[str, dict]:
        """Run a full tau2 simulation and return (trajectory, result).

        Args:
            task:    Task dict from get_tasks() — must contain _tau2_task or task_id.
            context: Retrieved knowledge from AgentState.retrieve().
            llm:     LLMClient instance (llm.model is used to pick the litellm model).

        Returns:
            (trajectory_str, result_dict) where result_dict has keys:
                outcome : "success" | "failure"
                score   : float 0–1
                details : str
        """
        self._configure_paths()
        from tau2.agent.llm_agent import LLMAgent
        from tau2.evaluator.evaluator import EvaluationType
        from tau2.registry import registry
        from tau2.run import get_tasks as _get_tasks
        from tau2.run import run_task

        try:
            prefixed_domain = task.get("task_id", "").partition("/")
            domain = task.get("_domain") or (
                prefixed_domain[0] if prefixed_domain[1] else self.domain
            )
            # Recover the tau2 Task object — it might have been stripped during
            # serialization (e.g. when tasks are re-loaded from a log file).
            tau2_task = task.get("_tau2_task")
            if tau2_task is None:
                # task_id may be prefixed with "<domain>/" for multi-domain runs.
                raw_id = task["task_id"].split("/", 1)[-1]
                tasks = _get_tasks(
                    _resolve_task_set_name(
                        domain,
                        task.get("split", self.task_split_name),
                    ),
                    task_ids=[raw_id],
                )
                if not tasks:
                    raise ValueError(
                        f"Task not found: domain={domain} id={raw_id}"
                    )
                tau2_task = tasks[0]

            model = _to_litellm_model(llm.model)

            agent_name = "wtf_context_agent"
            if agent_name not in registry.get_agents():
                class _ContextAgent(LLMAgent):
                    @property
                    def system_prompt(self_a) -> str:
                        base = LLMAgent.system_prompt.fget(self_a)
                        ctx = (_CONTEXT.get() or "").strip()
                        if not ctx:
                            return base
                        return (
                            f"{base}\n<retrieved_knowledge>\n{ctx}\n"
                            "</retrieved_knowledge>"
                        )

                registry.register_agent(_ContextAgent, agent_name)

            llm_args_agent = {"parallel_tool_calls": False}
            if getattr(llm, "base_url", None):
                llm_args_agent["api_base"] = llm.base_url
            extra_body = getattr(llm, "extra_body", None) or {}
            reasoning = extra_body.get("reasoning") or {}
            if reasoning.get("effort"):
                llm_args_agent["reasoning_effort"] = reasoning["effort"]
            if extra_body.get("verbosity"):
                llm_args_agent["verbosity"] = extra_body["verbosity"]

            token = _CONTEXT.set(context)
            try:
                sim_run = run_task(
                    domain=domain,
                    task=tau2_task,
                    agent=agent_name,
                    user="user_simulator",
                    llm_agent=model,
                    # The orchestrator consumes tool responses one at a time.
                    llm_args_agent=llm_args_agent,
                    llm_user=self.user_model,
                    llm_args_user={},
                    max_steps=self.max_steps,
                    max_errors=10,
                    evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
                )
            finally:
                _CONTEXT.reset(token)
            reward = sim_run.reward_info.reward if sim_run.reward_info else 0.0
            # tau2 attaches per-role cost to the SimulationRun. The agent here
            # is the model we're benchmarking (its cost is not in our LLMClient
            # because tau2 owns the call); the user simulator is gpt-4o by
            # default (also not in our LLMClient). Surface both so Runner can
            # roll them into the run-level cost summary.
            agent_cost = float(getattr(sim_run, "agent_cost", 0.0) or 0.0)
            user_cost  = float(getattr(sim_run, "user_cost",  0.0) or 0.0)
            return _messages_to_trajectory(sim_run), {
                "outcome": "success" if reward >= 1.0 else "failure",
                "score": float(reward),
                "details": _result_details(sim_run, domain, tau2_task.id),
                "cost_usd": agent_cost + user_cost,
                "cost_breakdown": {
                    "agent_cost_usd":          agent_cost,
                    "user_simulator_cost_usd": user_cost,
                },
            }

        except Exception as exc:
            return "", {
                "outcome": "failure",
                "score": 0.0,
                "details": f"Execution error: {type(exc).__name__}: {exc}",
            }
