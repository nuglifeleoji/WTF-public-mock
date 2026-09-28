"""Common interface for memory that evolves across tasks."""

from abc import ABC, abstractmethod


class AgentState(ABC):
    @abstractmethod
    def retrieve(self, task_description: str, k: int = 5) -> str:
        """Return memory context to include when solving the next task."""

    @abstractmethod
    def update(self, reflection: dict) -> None:
        """Learn from a completed task.

        Expected fields: task_description, trajectory, outcome (success/failure),
        reflection_text, optional reflection_json, and optional metadata.
        Reflection and evaluation are supplied by the caller.
        """

    @abstractmethod
    def serialize(self) -> str:
        """Serialize memory and bookkeeping as JSON."""

    @abstractmethod
    def load_serialized(self, payload: str) -> None:
        """Restore memory from a serialized snapshot."""

    @abstractmethod
    def stats(self) -> dict:
        """Return memory size and method-specific counters."""
