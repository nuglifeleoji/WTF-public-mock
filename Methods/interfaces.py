"""Client contracts; callers supply their own model and embedding adapters."""

from typing import Protocol

import numpy as np


class LLMClient(Protocol):
    def generate(
        self, prompt: str, max_tokens: int = 2048, system: str | None = None
    ) -> str:
        """Generate text, optionally with a separate system message."""
        ...


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) array of L2-normalized embeddings."""
        ...

    def top_k_indices(
        self, query: str, candidates: list[str], k: int
    ) -> list[int]:
        """Return candidate indices ordered by decreasing query similarity."""
        ...
