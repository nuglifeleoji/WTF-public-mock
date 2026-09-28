"""Thin wrappers around LLM APIs (Anthropic / OpenAI) and sentence-transformers (embeddings)."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

from .pricing import compute_cost, normalize_usage


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))


def _is_openrouter_model(model: str) -> bool:
    return model.startswith("openrouter/")


def _normalise_provider(provider: Optional[str]) -> Optional[str]:
    if provider is None:
        return None
    provider = provider.lower().strip()
    if provider not in {"openai", "anthropic", "openrouter"}:
        raise ValueError(
            "Unsupported LLM provider "
            f"{provider!r}; expected openai, anthropic, or openrouter"
        )
    return provider


def _infer_backend(model: str, provider: Optional[str] = None) -> str:
    provider = _normalise_provider(provider)
    if provider:
        return provider
    if _is_openrouter_model(model):
        return "openrouter"
    return "openai" if model.startswith("openai/") or _is_openai_model(model) else "anthropic"


def _api_model_name(model: str, backend: str) -> str:
    if backend == "openrouter" and model.startswith("openrouter/"):
        return model.split("/", 1)[1]
    if backend == "openai" and model.startswith("openai/"):
        return model.split("/", 1)[1]
    if backend == "anthropic" and model.startswith("anthropic/"):
        return model.split("/", 1)[1]
    return model


def _uses_openai_responses_api(model: str) -> bool:
    return model.startswith(("o1", "o3", "o4"))


def _uses_completion_tokens_param(model: str) -> bool:
    """Newer OpenAI models (gpt-5.x) reject 'max_tokens'; they need 'max_completion_tokens'."""
    return model.startswith("gpt-5")


def _fixed_temperature_model(model: str) -> bool:
    """Reasoning-style models only support the default temperature; don't send a custom value."""
    return model.startswith("gpt-5")


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around Anthropic, OpenAI, or OpenRouter's OpenAI-compatible API.
    Backend is selected automatically based on the model name:
      - claude-*  → Anthropic
      - gpt-* / o1* / o3* / o4* → OpenAI
      - openrouter/* → OpenRouter
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.provider = _normalise_provider(provider)
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.extra_body = extra_body or {}
        self._backend = _infer_backend(model, self.provider)
        self._api_model = _api_model_name(model, self._backend)

        if self._backend in {"openai", "openrouter"}:
            import openai

            api_key = os.environ.get(api_key_env) if api_key_env else None
            client_kwargs: dict = {"api_key": api_key}
            if self._backend == "openrouter":
                client_kwargs["api_key"] = (
                    api_key
                    or os.environ.get("OPENROUTER_API_KEY")
                    or os.environ.get("openrouter_api_key")
                )
                client_kwargs["base_url"] = base_url or OPENROUTER_BASE_URL
                default_headers = {}
                referer = (
                    os.environ.get("OPENROUTER_HTTP_REFERER")
                    or os.environ.get("OPENROUTER_SITE_URL")
                )
                title = os.environ.get("OPENROUTER_APP_TITLE")
                if referer:
                    default_headers["HTTP-Referer"] = referer
                if title:
                    default_headers["X-OpenRouter-Title"] = title
                if default_headers:
                    client_kwargs["default_headers"] = default_headers
            else:
                client_kwargs["api_key"] = (
                    api_key
                    or os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("openai_api_key")
                )
                if base_url:
                    client_kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**client_kwargs)
        else:
            import anthropic
            api_key = os.environ.get(api_key_env) if api_key_env else None
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = anthropic.Anthropic(**client_kwargs)

        # Cost / usage accounting. Every successful call updates these.
        # Callers (e.g. Runner) read snapshots before+after each step to get
        # per-step deltas without having to thread cost through every method.
        self.total_cost: float = 0.0
        self.call_count: int = 0
        self.total_usage: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        self.last_call: Optional[dict] = None  # {cost, usage} for the most recent successful call

    def _record_usage(self, raw_usage: dict) -> dict:
        """Update running totals from a raw API usage dict; return a normalised
        per-call snapshot {cost, usage}."""
        if not raw_usage:
            return {"cost": 0.0, "usage": normalize_usage({})}
        usage_norm = normalize_usage(raw_usage)
        cost = compute_cost(self.model, raw_usage)
        self.total_cost += cost
        self.call_count += 1
        for k, v in usage_norm.items():
            self.total_usage[k] = self.total_usage.get(k, 0) + v
        self.last_call = {"cost": cost, "usage": usage_norm, "model": self.model}
        return self.last_call

    def get_cost_snapshot(self) -> dict:
        """Stable read-only view of running totals."""
        return {
            "total_cost_usd": self.total_cost,
            "call_count":     self.call_count,
            "total_usage":    dict(self.total_usage),
            "model":          self.model,
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> str:
        """Single-turn generation. Returns the assistant's text response."""
        messages = [{"role": "user", "content": prompt}]
        return self._call(messages, max_tokens=max_tokens, system=system)

    def generate_messages(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> str:
        """Multi-turn generation. Messages is a list of {role, content} dicts."""
        return self._call(messages, max_tokens=max_tokens, system=system)

    def _call(
        self,
        messages: list[dict],
        max_tokens: int,
        system: Optional[str],
    ) -> str:
        if self._backend in {"openai", "openrouter"}:
            return self._call_openai(messages, max_tokens, system)
        return self._call_anthropic(messages, max_tokens, system)

    def _call_openai(
        self,
        messages: list[dict],
        max_tokens: int,
        system: Optional[str],
    ) -> str:
        import openai
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        for attempt in range(self.max_retries):
            try:
                if self._backend == "openai" and _uses_openai_responses_api(self._api_model):
                    request: dict = {
                        "model": self._api_model,
                        "input": all_messages,
                        "max_output_tokens": max_tokens,
                    }
                    response = self._client.responses.create(**request)
                    self._record_usage(_extract_openai_usage(response))
                    text = getattr(response, "output_text", "") or ""
                    if text:
                        return text

                    text_parts: list[str] = []
                    for item in getattr(response, "output", []) or []:
                        if getattr(item, "type", None) != "message":
                            continue
                        for part in getattr(item, "content", []) or []:
                            if getattr(part, "type", None) == "output_text":
                                text_parts.append(getattr(part, "text", ""))
                    return "\n".join(part for part in text_parts if part).strip()

                create_kwargs: dict = {
                    "model": self._api_model,
                    "messages": all_messages,
                }
                if self._backend == "openai" and _uses_completion_tokens_param(self._api_model):
                    create_kwargs["max_completion_tokens"] = max_tokens
                else:
                    create_kwargs["max_tokens"] = max_tokens
                if self._backend != "openai" or not _fixed_temperature_model(self._api_model):
                    create_kwargs["temperature"] = self.temperature
                if self.extra_body:
                    create_kwargs["extra_body"] = self.extra_body
                response = self._client.chat.completions.create(**create_kwargs)
                self._record_usage(_extract_openai_usage(response))
                return response.choices[0].message.content or ""
            except openai.RateLimitError:
                time.sleep(2 ** attempt * 5)
            except (openai.APIConnectionError, openai.APITimeoutError, json.JSONDecodeError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
            except openai.APIStatusError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries")

    def _call_anthropic(
        self,
        messages: list[dict],
        max_tokens: int,
        system: Optional[str],
    ) -> str:
        import anthropic
        kwargs: dict = dict(
            model=self._api_model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(**kwargs)
                self._record_usage(_extract_anthropic_usage(response))
                return response.content[0].text
            except anthropic.RateLimitError:
                time.sleep(2 ** attempt * 5)
            except anthropic.APIStatusError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries")


def _extract_openai_usage(response) -> dict:
    """Pull usage fields from an OpenAI chat / responses response."""
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    out = {
        "prompt_tokens":     getattr(u, "prompt_tokens", 0) or getattr(u, "input_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or getattr(u, "output_tokens", 0) or 0,
    }
    pdetails = getattr(u, "prompt_tokens_details", None) or getattr(u, "input_tokens_details", None)
    if pdetails is not None:
        out["prompt_tokens_details"] = {
            "cached_tokens": getattr(pdetails, "cached_tokens", 0) or 0,
        }
    cdetails = getattr(u, "completion_tokens_details", None) or getattr(u, "output_tokens_details", None)
    if cdetails is not None:
        out["completion_tokens_details"] = {
            "reasoning_tokens": getattr(cdetails, "reasoning_tokens", 0) or 0,
        }
    return out


def _extract_anthropic_usage(response) -> dict:
    """Pull usage fields from an Anthropic messages.create response."""
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    return {
        "input_tokens":  getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_input_tokens":     getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


# ---------------------------------------------------------------------------
# Embedding Client
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """
    Embedding client backed by sentence-transformers.
    Embeddings are L2-normalised so cosine similarity = dot product.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, D) float32 array of normalised embeddings."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def top_k_indices(self, query: str, candidates: list[str], k: int) -> list[int]:
        """Return indices of the top-k most similar candidates to query."""
        if not candidates:
            return []
        k = min(k, len(candidates))
        vecs = self.embed([query] + candidates)
        query_vec = vecs[0]
        cand_vecs = vecs[1:]
        scores = cand_vecs @ query_vec
        return list(np.argsort(scores)[::-1][:k])
