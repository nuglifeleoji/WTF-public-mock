"""Shared utilities for AgentState subclasses."""

from __future__ import annotations

import json
import re
from typing import Optional

# 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def extract_json(text: Optional[str]) -> Optional[dict]:
    """Best-effort JSON extraction from LLM output.

    Tries, in order:
    1. Direct parse
    2. ```json ... ``` fenced block
    3. First balanced { ... } in the text
    """
    if text is None:
        return None

    text = str(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # First balanced brace group
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, start = 1, i
        i += 1
        while i < len(text) and depth:
            c = text[i]
            if c == '"':
                i += 1
                while i < len(text) and text[i] != '"':
                    if text[i] == "\\":
                        i += 1
                    i += 1
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth == 0:
            try:
                return json.loads(text[start:i])
            except json.JSONDecodeError:
                pass

    return None
