from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from runtime.llm import LLMClient


class _FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": "response text", "output": []})()


class _FakeChatCompletionsAPI:
    def create(self, **kwargs):
        raise AssertionError("chat.completions should not be used for o-series models")


class _RecordingChatCompletionsAPI:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": "chat text"})()
        choice = type("Choice", (), {"message": message})()
        usage = type("Usage", (), {"prompt_tokens": 3, "completion_tokens": 4})()
        return type("Response", (), {"choices": [choice], "usage": usage})()


class _NullContentChatCompletionsAPI:
    def create(self, **kwargs):
        message = type("Message", (), {"content": None})()
        choice = type("Choice", (), {"message": message})()
        usage = type("Usage", (), {"prompt_tokens": 3, "completion_tokens": 4})()
        return type("Response", (), {"choices": [choice], "usage": usage})()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponsesAPI()
        self.chat = type("Chat", (), {"completions": _FakeChatCompletionsAPI()})()


class _RecordingOpenAIClient:
    instances: list["_RecordingOpenAIClient"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.responses = _FakeResponsesAPI()
        self.chat_completions = _RecordingChatCompletionsAPI()
        self.chat = type("Chat", (), {"completions": self.chat_completions})()
        self.instances.append(self)


class _NullContentOpenAIClient:
    def __init__(self, **kwargs) -> None:
        self.responses = _FakeResponsesAPI()
        self.chat = type("Chat", (), {"completions": _NullContentChatCompletionsAPI()})()


class LLMClientTests(unittest.TestCase):
    def test_anthropic_prefix_and_endpoint_are_normalized(self) -> None:
        fake = mock.Mock()
        fake.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="answer")], usage=SimpleNamespace()
        )
        with mock.patch("anthropic.Anthropic", return_value=fake) as constructor:
            client = LLMClient(model="anthropic/claude-sonnet-4-6", base_url="https://example.invalid")
            self.assertEqual(client.generate("hello"), "answer")
        self.assertEqual(constructor.call_args.kwargs["base_url"], "https://example.invalid")
        self.assertEqual(fake.messages.create.call_args.kwargs["model"], "claude-sonnet-4-6")

    def test_o_series_models_use_responses_api(self) -> None:
        fake_client = _FakeOpenAIClient()

        with mock.patch("openai.OpenAI", return_value=fake_client):
            client = LLMClient(model="o4", temperature=0.0)
            text = client.generate("hello", max_tokens=128, system="be concise")

        self.assertEqual(text, "response text")
        self.assertEqual(len(fake_client.responses.calls), 1)
        request = fake_client.responses.calls[0]
        self.assertEqual(request["model"], "o4")
        self.assertEqual(request["max_output_tokens"], 128)
        self.assertEqual(request["input"][0]["role"], "system")
        self.assertEqual(request["input"][1]["role"], "user")

    def test_openrouter_uses_openai_compatible_client(self) -> None:
        _RecordingOpenAIClient.instances = []

        with (
            mock.patch.dict("os.environ", {"ACE_OPENROUTER_API_KEY": "test-key"}, clear=True),
            mock.patch("openai.OpenAI", side_effect=lambda **kw: _RecordingOpenAIClient(**kw)),
        ):
            client = LLMClient(
                model="openrouter/deepseek/deepseek-v4-pro",
                temperature=0.0,
                provider="openrouter",
                api_key_env="ACE_OPENROUTER_API_KEY",
            )
            text = client.generate("hello", max_tokens=128)

        self.assertEqual(text, "chat text")
        fake_client = _RecordingOpenAIClient.instances[0]
        self.assertEqual(fake_client.kwargs["api_key"], "test-key")
        self.assertEqual(fake_client.kwargs["base_url"], "https://openrouter.ai/api/v1")
        request = fake_client.chat_completions.calls[0]
        self.assertEqual(request["model"], "deepseek/deepseek-v4-pro")
        self.assertEqual(request["max_tokens"], 128)
        self.assertNotIn("max_completion_tokens", request)

    def test_chat_completion_null_content_returns_empty_string(self) -> None:
        with mock.patch("openai.OpenAI", side_effect=lambda **kw: _NullContentOpenAIClient(**kw)):
            client = LLMClient(model="openrouter/deepseek/deepseek-v4-pro", provider="openrouter")
            text = client.generate("hello", max_tokens=128)

        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
