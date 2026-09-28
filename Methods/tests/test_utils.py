"""Model responses may wrap JSON in prose or Markdown fences."""

from Methods._utils import extract_json


def test_extract_json_handles_wrappers_and_quoted_braces():
    expected = {"text": "a } brace", "nested": {"value": 1}}
    payload = '{"text": "a } brace", "nested": {"value": 1}}'
    assert extract_json(payload) == expected
    assert extract_json(f"```json\n{payload}\n```") == expected
    assert extract_json(f"Result follows: {payload} Done.") == expected
    assert extract_json(None) is None
    assert extract_json("No structured result") is None
