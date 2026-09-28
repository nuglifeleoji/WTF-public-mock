from abc import ABC, abstractmethod


class Benchmark(ABC):
    """
    Abstract base class for benchmarks.
    Each benchmark provides a stream of tasks and evaluation.
    """

    @abstractmethod
    def get_tasks(self) -> list[dict]:
        """
        Return ordered list of tasks. Each task is a dict with at least:
        - task_id: str
        - description: str
        - any benchmark-specific fields
        """
        pass

    @abstractmethod
    def evaluate(self, task: dict, agent_output: str) -> dict:
        """
        Evaluate agent output on a task. Returns:
        - outcome: str ("success" | "failure")
        - score: float (0-1, for partial credit if applicable)
        - details: str (error messages, test results, etc.)
        """
        pass

    def solve(self, task: dict, context: str, llm) -> tuple[str, dict]:
        """
        Solve a task and return (trajectory, result).

        Default: single-shot — build a prompt, call the LLM once, then evaluate.
        Benchmarks that require interactive execution (e.g. ReAct loops) should
        override this method.
        """
        from runtime.prompts import build_generation_prompt
        prompt = build_generation_prompt(
            task["description"], context, task.get("env_context", "")
        )
        trajectory = llm.generate(prompt)
        result = self.evaluate(task, trajectory)
        return trajectory, result
