"""Historical experiment pricing estimates for LLM cost accounting.

These rates are retained from the original experiment implementation. They are
not current billing rates; use provider billing records for actual costs.

Values are USD per million tokens. Where the API distinguishes cached input
from regular input we apply the cached rate to that fraction.

Coverage covers the models we actually invoke in this rig:
  - OpenAI: gpt-4o(-mini), gpt-5.x (reasoning-style, used as gpt-5.4 internally),
            o1/o3/o4
  - Anthropic: claude-sonnet-4-6/4-5, claude-haiku-4-5, claude-opus-4-7
  - OpenRouter: DeepSeek V4 Pro

Unknown models silently price at zero so tests / smokes keep running. Callers
that want hard failures can check `is_known(model)` first.
"""

from __future__ import annotations

from typing import Optional


# (input, output, cached_input, cache_write) — all per 1M tokens (USD).
# cached_input applies to the *cache hit* portion of the input tokens.
# cache_write applies to tokens stored into cache (Anthropic only).
MODEL_PRICING: dict[str, dict[str, float]] = {
    # ---- OpenAI ----
    "gpt-4o":      {"input": 2.50,  "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15,  "output":  0.60, "cached_input": 0.075},
    # gpt-5.x is treated as a reasoning model in our LLMClient. Output tokens
    # include reasoning tokens. Pricing here is a best-effort placeholder
    # consistent with public guidance for the gpt-5 family at the time of
    # writing; adjust if Anthropic/OpenAI publishes other numbers later.
    "gpt-5":       {"input": 1.25,  "output": 10.00, "cached_input": 0.125},
    "gpt-5.4":     {"input": 1.25,  "output": 10.00, "cached_input": 0.125},
    "o1":          {"input": 15.00, "output": 60.00, "cached_input": 7.50},
    "o3":          {"input":  2.00, "output":  8.00, "cached_input": 0.50},
    "o4":          {"input":  3.00, "output": 12.00, "cached_input": 0.75},

    # ---- Anthropic ----
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cached_input": 0.30, "cache_write": 3.75},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00, "cached_input": 0.30, "cache_write": 3.75},
    "claude-sonnet":     {"input": 3.00, "output": 15.00, "cached_input": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5":  {"input": 0.80, "output":  4.00, "cached_input": 0.08, "cache_write": 1.00},
    "claude-haiku":      {"input": 0.80, "output":  4.00, "cached_input": 0.08, "cache_write": 1.00},
    "claude-opus-4-7":   {"input": 15.00, "output": 75.00, "cached_input": 1.50, "cache_write": 18.75},
    "claude-opus":       {"input": 15.00, "output": 75.00, "cached_input": 1.50, "cache_write": 18.75},

    # ---- OpenRouter / DeepSeek ----
    "deepseek/deepseek-v4-pro": {"input": 0.435, "output": 0.87},
}


def _resolve(model: str) -> Optional[dict[str, float]]:
    """Look up the pricing entry for *model*, with longest-prefix fallback."""
    if not model:
        return None
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # litellm/OpenRouter provider/model strings can be nested:
    # openrouter/openai/gpt-5.4, openrouter/deepseek/deepseek-v4-pro.
    for prefix in ("openrouter/", "openai/", "anthropic/"):
        while model.startswith(prefix):
            model = model[len(prefix):]
            if model in MODEL_PRICING:
                return MODEL_PRICING[model]
    # longest-prefix match (handles dated revisions like "claude-sonnet-4-5-20240620")
    candidates = [k for k in MODEL_PRICING if model.startswith(k)]
    if not candidates:
        return None
    return MODEL_PRICING[max(candidates, key=len)]


def is_known(model: str) -> bool:
    return _resolve(model) is not None


def compute_cost(model: str, usage: dict) -> float:
    """Compute USD cost from a usage dict.

    Accepts both Anthropic and OpenAI usage shapes. Tolerates missing fields.

    Anthropic:
      input_tokens, output_tokens
      cache_read_input_tokens, cache_creation_input_tokens   (optional)

    OpenAI Chat / Responses:
      prompt_tokens, completion_tokens
      prompt_tokens_details: {cached_tokens}                 (optional)
      completion_tokens_details: {reasoning_tokens}          (optional, no extra charge)

    Returns 0.0 if the model isn't in the pricing table.
    """
    pricing = _resolve(model)
    if pricing is None:
        return 0.0

    inp_rate         = pricing["input"] / 1_000_000
    out_rate         = pricing["output"] / 1_000_000
    cached_inp_rate  = pricing.get("cached_input", pricing["input"]) / 1_000_000
    cache_write_rate = pricing.get("cache_write", pricing["input"]) / 1_000_000

    in_tok = (
        usage.get("input_tokens")
        if "input_tokens" in usage
        else usage.get("prompt_tokens", 0)
    ) or 0
    out_tok = (
        usage.get("output_tokens")
        if "output_tokens" in usage
        else usage.get("completion_tokens", 0)
    ) or 0

    cached_in_tok = (
        usage.get("cache_read_input_tokens", 0)
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        or 0
    )
    cache_write_tok = usage.get("cache_creation_input_tokens", 0) or 0

    # OpenAI counts cached tokens *inside* prompt_tokens; Anthropic counts them
    # outside `input_tokens`. Normalise so non-cached input is the uncached
    # remainder either way.
    if "prompt_tokens" in usage:  # OpenAI shape
        non_cached_in = max(0, in_tok - cached_in_tok)
    else:                          # Anthropic shape
        non_cached_in = in_tok

    cost = (
        non_cached_in   * inp_rate
        + cached_in_tok * cached_inp_rate
        + out_tok       * out_rate
        + cache_write_tok * cache_write_rate
    )
    return float(cost)


def normalize_usage(usage: dict) -> dict:
    """Return a stable dict with a common schema for logging.

    Always present:
      input_tokens, output_tokens, cached_input_tokens, cache_creation_tokens,
      reasoning_tokens (best-effort), total_tokens
    """
    in_tok = (
        usage.get("input_tokens")
        if "input_tokens" in usage
        else usage.get("prompt_tokens", 0)
    ) or 0
    out_tok = (
        usage.get("output_tokens")
        if "output_tokens" in usage
        else usage.get("completion_tokens", 0)
    ) or 0
    cached_in = (
        usage.get("cache_read_input_tokens", 0)
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        or 0
    )
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    return {
        "input_tokens":          int(in_tok),
        "output_tokens":         int(out_tok),
        "cached_input_tokens":   int(cached_in),
        "cache_creation_tokens": int(cache_write),
        "reasoning_tokens":      int(reasoning),
        "total_tokens":          int(in_tok) + int(out_tok),
    }
