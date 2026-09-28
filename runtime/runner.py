"""Runner — the main cross-session self-improvement loop.

The Runner is completely state-agnostic: it always runs the same
  retrieve → generate → evaluate → reflect → update → log
loop regardless of which AgentState subclass is used.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from Methods.base import AgentState
from Methods._utils import extract_json
from Task.benchmarks.base import Benchmark
from .llm import LLMClient
from .prompts import build_reflection_prompt, prepare_reflection_inputs


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    # Experiment identity
    run_id: str
    state_type: str          # playbook, awm, reasoning_bank, hermes, or baseline
    benchmark: str

    # LLM
    model: str               # e.g. "claude-sonnet-4-6"
    temperature: float = 0.0
    llm_provider: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_api_key_env: Optional[str] = None
    llm_extra_body: Optional[dict] = None
    llm_reasoning_enabled: Optional[bool] = None
    llm_reasoning_effort: Optional[str] = None
    llm_verbosity: Optional[str] = None

    # State-specific
    max_state_tokens: int = 50_000
    retrieval_k: int = 5

    # Logging
    snapshot_interval: int = 10            # save full state every N steps
    log_dir: str = "experiments"
    eval_only: bool = False
    load_state: Optional[str] = None
    task_file: Optional[str] = None
    settings: Optional[dict] = None


# ---------------------------------------------------------------------------
# Experiment logger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Writes one JSON object per step to log.jsonl and periodic state snapshots.

    Directory layout::

        {log_dir}/{run_id}/
            config.json
            log.jsonl
            state_snapshots/
                step_{i:06d}.json
            summary.json          ← written by Runner.finalize()
    """

    def __init__(self, config: RunConfig):
        self._config = config
        self._run_dir = Path(config.log_dir) / config.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "state_snapshots").mkdir(exist_ok=True)

        # Write config for reproducibility
        with open(self._run_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        self._log_path = self._run_dir / "log.jsonl"
        self._log_fh = open(self._log_path, "a")

    def log_step(
        self,
        step: int,
        task: dict,
        context_retrieved: str,
        trajectory: str,
        result: dict,
        reflection: dict,
        state_snapshot: str,
        state_stats: dict,
        timing: dict,
        cost: dict | None = None,
    ) -> None:
        record = {
            "step": step,
            "task_id": task.get("task_id"),
            "task_description": task.get("description"),
            # Omit raw _appworld_task object — not serialisable
            "task_meta": {k: v for k, v in task.items() if not k.startswith("_")},
            "context_retrieved": context_retrieved,
            "trajectory": trajectory,
            "result": result,
            "reflection_text": reflection.get("reflection_text", ""),
            "state_stats": state_stats,
            "timing": timing,
        }
        if cost is not None:
            record["cost"] = cost
        self._log_fh.write(json.dumps(record) + "\n")
        self._log_fh.flush()

        if step % self._config.snapshot_interval == 0:
            snapshot_path = (
                self._run_dir / "state_snapshots" / f"step_{step:06d}.json"
            )
            with open(snapshot_path, "w") as f:
                f.write(state_snapshot)

    def write_summary(self, summary: dict) -> None:
        with open(self._run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def close(self) -> None:
        self._log_fh.close()

    @property
    def run_dir(self) -> Path:
        return self._run_dir


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    """
    Orchestrates the cross-session self-improvement loop.

    The loop for each task:
      1. retrieve: pull relevant state context
      2. generate: LLM solves the task using that context
      3. evaluate: benchmark scores the output
      4. reflect: LLM analyses the execution
      5. update: agent state is updated with the reflection
      6. log: everything is written to disk
    """

    def __init__(
        self,
        agent_state: AgentState,
        benchmark: Benchmark,
        llm: LLMClient,
        config: RunConfig,
    ):
        self.state = agent_state
        self.benchmark = benchmark
        self.llm = llm
        self.config = config
        self.logger = ExperimentLogger(config)

    def run(self) -> dict:
        """Run the full experiment. Returns a summary dict."""
        tasks = self.benchmark.get_tasks()
        successes: list[int] = []
        scores: list[float] = []
        outcome_counts: dict[str, int] = {}

        print(f"[Runner] Starting run '{self.config.run_id}' — {len(tasks)} tasks")

        for i, task in enumerate(tasks):
            timing: dict[str, float] = {}
            # Snapshot LLM client cost before this step so we can compute deltas.
            cost_pre = self.llm.get_cost_snapshot() if hasattr(self.llm, "get_cost_snapshot") else None

            # --- 1. Retrieve ---
            t0 = time.perf_counter()
            context = self.state.retrieve(task["description"], k=self.config.retrieval_k)
            timing["retrieve_s"] = time.perf_counter() - t0

            cost_phases: dict[str, float] = {}
            cost_after_solve = cost_pre

            # --- 2. Generate + Evaluate ---
            # benchmark.solve() handles the full interaction (e.g. ReAct loop for
            # AppWorld) and returns the trajectory string alongside the result dict.
            t0 = time.perf_counter()
            trajectory, result = self.benchmark.solve(task, context, self.llm)
            timing["generate_s"] = time.perf_counter() - t0
            if cost_pre is not None:
                cost_after_solve = self.llm.get_cost_snapshot()
                cost_phases["solve"] = cost_after_solve["total_cost_usd"] - cost_pre["total_cost_usd"]

            # --- 3. Reflect / 4. Update ---
            if self.config.eval_only or result.get("outcome") not in {"success", "failure"}:
                reflection = {}
                timing["reflect_s"] = 0.0
                timing["update_s"] = 0.0
            else:
                t0 = time.perf_counter()
                reflection = self._reflect(task, trajectory, result, state_context=context)
                timing["reflect_s"] = time.perf_counter() - t0
                if cost_pre is not None:
                    snap = self.llm.get_cost_snapshot()
                    cost_phases["reflect"] = snap["total_cost_usd"] - cost_after_solve["total_cost_usd"]
                    cost_after_reflect = snap
                else:
                    cost_after_reflect = None

                t0 = time.perf_counter()
                self.state.update(reflection)
                timing["update_s"] = time.perf_counter() - t0
                if cost_after_reflect is not None:
                    snap = self.llm.get_cost_snapshot()
                    cost_phases["update"] = snap["total_cost_usd"] - cost_after_reflect["total_cost_usd"]

            # Build cost record for this step (combines our LLMClient deltas
            # with any benchmark-internal cost the adapter put on the result dict).
            cost_record: dict | None = None
            if cost_pre is not None:
                cost_post = self.llm.get_cost_snapshot()
                step_total = cost_post["total_cost_usd"] - cost_pre["total_cost_usd"]
                step_calls = cost_post["call_count"] - cost_pre["call_count"]
                step_usage = {
                    k: cost_post["total_usage"].get(k, 0) - cost_pre["total_usage"].get(k, 0)
                    for k in cost_post["total_usage"]
                }
                # Benchmark-internal cost (tau2 user_simulator + agent inside
                # tau2 framework, terminalbench harbor agent, etc.) — adapters
                # may stash a "cost_usd" / "cost_breakdown" on the result dict.
                bench_cost = float(result.get("cost_usd", 0.0) or 0.0) if isinstance(result, dict) else 0.0
                bench_breakdown = (
                    result.get("cost_breakdown") if isinstance(result, dict) else None
                ) or {}
                cost_record = {
                    "model": cost_post["model"],
                    "step_cost_usd": float(step_total),
                    "benchmark_cost_usd": bench_cost,
                    "total_step_cost_usd": float(step_total) + bench_cost,
                    "step_calls": step_calls,
                    "step_usage": step_usage,
                    "phases": {k: float(v) for k, v in cost_phases.items()},
                    "benchmark_breakdown": bench_breakdown,
                    "cumulative_cost_usd": float(cost_post["total_cost_usd"]),
                    "cumulative_calls": cost_post["call_count"],
                }

            # --- 5. Log ---
            self.logger.log_step(
                step=i,
                task=task,
                context_retrieved=context,
                trajectory=trajectory,
                result=result,
                reflection=reflection,
                state_snapshot=self.state.serialize(),
                state_stats=self.state.stats(),
                timing=timing,
                cost=cost_record,
            )

            outcome = result["outcome"]
            score = result.get("score", 1.0 if outcome == "success" else 0.0)
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            if outcome in {"success", "failure"}:
                successes.append(1 if outcome == "success" else 0)
                scores.append(score)

            print(
                f"  step {i:4d}/{len(tasks)}  "
                f"task={task['task_id']}  "
                f"outcome={outcome}  "
                f"score={score:.2f}  "
                f"state={self.state.stats().get('approx_tokens', '?')} tok"
            )

        summary = self._compute_summary(
            successes,
            scores,
            total_tasks=len(tasks),
            outcome_counts=outcome_counts,
        )
        # Attach cost totals to the summary so downstream analysis doesn't have
        # to re-grep log.jsonl. Includes both LLMClient (our own calls) and
        # benchmark-internal cost (tau2 user_simulator, terminalbench harbor
        # agent, etc.) summed from per-step records.
        summary["cost"] = self._compute_cost_summary()
        self.logger.write_summary(summary)

        # Always save a final snapshot so the end state is captured regardless
        # of whether the last step happened to fall on snapshot_interval.
        final_step = len(tasks) - 1
        if final_step >= 0 and final_step % self.config.snapshot_interval != 0:
            snapshot_path = (
                self.logger.run_dir / "state_snapshots" / f"step_{final_step:06d}.json"
            )
            with open(snapshot_path, "w") as f:
                f.write(self.state.serialize())

        self.logger.close()

        cost_total = summary.get("cost", {}).get("total_cost_usd", 0.0)
        print(
            f"[Runner] Done. "
            f"accuracy={summary['accuracy']:.3f}  "
            f"avg_score={summary['avg_score']:.3f}  "
            f"cost=${cost_total:.4f}  "
            f"logs → {self.logger.run_dir}"
        )
        return summary

    def _compute_cost_summary(self) -> dict:
        """Aggregate cost across all logged steps + LLMClient running totals.

        Reads log.jsonl back so we capture both LLMClient calls (already in
        our running total) and any benchmark-internal cost the adapter wrote
        onto the result dict. Resilient to missing fields on older logs.
        """
        client_snap = (
            self.llm.get_cost_snapshot()
            if hasattr(self.llm, "get_cost_snapshot")
            else None
        )

        total_bench_cost = 0.0
        bench_breakdown_total: dict[str, float] = {}
        per_phase_total: dict[str, float] = {"solve": 0.0, "reflect": 0.0, "update": 0.0}
        log_path = self.logger.run_dir / "log.jsonl"
        try:
            for line in open(log_path):
                rec = json.loads(line)
                cost = rec.get("cost") or {}
                total_bench_cost += float(cost.get("benchmark_cost_usd") or 0.0)
                for k, v in (cost.get("phases") or {}).items():
                    per_phase_total[k] = per_phase_total.get(k, 0.0) + float(v or 0.0)
                for k, v in (cost.get("benchmark_breakdown") or {}).items():
                    if isinstance(v, (int, float)):
                        bench_breakdown_total[k] = bench_breakdown_total.get(k, 0.0) + float(v)
        except FileNotFoundError:
            pass

        client_cost = float(client_snap["total_cost_usd"]) if client_snap else 0.0
        return {
            "total_cost_usd":          client_cost + total_bench_cost,
            "client_cost_usd":         client_cost,
            "benchmark_cost_usd":      total_bench_cost,
            "client_calls":            int(client_snap["call_count"]) if client_snap else 0,
            "client_total_usage":      dict(client_snap["total_usage"]) if client_snap else {},
            "phase_breakdown":         per_phase_total,
            "benchmark_breakdown":     bench_breakdown_total,
            "model":                   client_snap["model"] if client_snap else self.config.model,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reflect(self, task: dict, trajectory: str, result: dict, state_context: str = "") -> dict:
        prepared = prepare_reflection_inputs(
            task=task["description"],
            trajectory=trajectory,
            outcome=result["outcome"],
            details=result.get("details", ""),
            playbook=state_context,
        )
        prompt = build_reflection_prompt(
            task=prepared["task"],
            trajectory=prepared["trajectory"],
            outcome=prepared["outcome"],
            details=prepared["details"],
            playbook=prepared["playbook"],
            benchmark=self.config.benchmark,
        )
        reflection_json: dict = {}
        reflection_error = ""
        try:
            raw = self.llm.generate(prompt, max_tokens=2048)
            reflection_json = extract_json(raw) or {}
            key_insight = reflection_json.get("key_insight", raw)
        except Exception as exc:
            reflection_error = f"{type(exc).__name__}: {exc}"
            key_insight = (
                "Reflection skipped because the reflector call failed; rely on the "
                f"benchmark result details instead. ({reflection_error})"
            )

        return {
            "task_description": task["description"],
            "trajectory": prepared["trajectory"],
            "outcome": result["outcome"],
            # reflection_text: the most actionable single sentence, used by all AgentState.update()
            "reflection_text": key_insight,
            # reflection_json: full structured output, used by PlaybookState curator
            "reflection_json": reflection_json,
            "metadata": {
                "task_id": task.get("task_id"),
                **result,
                "reflection_error": reflection_error,
            },
        }

    def _compute_summary(
        self,
        successes: list[int],
        scores: list[float],
        total_tasks: int,
        outcome_counts: dict[str, int],
    ) -> dict:
        n = len(successes)
        if n == 0:
            # Cost is filled in by run() itself after this returns; leave a
            # zeroed placeholder so JSON consumers always see the same shape.
            return {
                "n_tasks": total_tasks,
                "n_evaluated_tasks": 0,
                "accuracy": 0.0,
                "avg_score": 0.0,
                "successes": successes,
                "scores": scores,
                "windowed_accuracy": [],
                "outcome_counts": outcome_counts,
                "state_type": self.config.state_type,
                "model": self.config.model,
                "run_id": self.config.run_id,
            }

        # Sliding window accuracy (window = 10% of tasks, min 10) — O(n) incremental
        window = max(10, n // 10)
        windowed: list[float] = []
        running = 0
        for i, s in enumerate(successes):
            running += s
            if i >= window:
                running -= successes[i - window]
            windowed.append(running / min(i + 1, window))

        return {
            "n_tasks": total_tasks,
            "n_evaluated_tasks": n,
            "accuracy": sum(successes) / n,
            "avg_score": sum(scores) / n,
            "successes": successes,
            "scores": scores,
            "windowed_accuracy": windowed,
            "outcome_counts": outcome_counts,
            "state_type": self.config.state_type,
            "model": self.config.model,
            "run_id": self.config.run_id,
        }
