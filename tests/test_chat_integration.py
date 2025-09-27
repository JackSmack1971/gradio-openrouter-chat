from __future__ import annotations

from types import SimpleNamespace

import main
import utils
from utils import sanitize_text


def _chunk(content: str | None = None, usage: object | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta)
    chunk = SimpleNamespace(choices=[choice])
    if usage is not None:
        chunk.usage = usage
    return chunk


def _stub_logger():
    return SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )


def test_chat_streaming_success(mock_client, conversation_state, monkeypatch):
    monkeypatch.setattr(main, "logger", _stub_logger())
    monkeypatch.setattr(main.limiter, "check", lambda _ip: True)

    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=7)
    mock_client.queue_chunks([_chunk("Hello"), _chunk(" world", usage)])

    noisy_message = "Streaming sanitized message\x07"
    expected_message = sanitize_text(noisy_message, main.settings.max_input_chars)
    request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.1"))

    outputs = list(
        main.chat_fn(noisy_message, conversation_state, "openai/gpt-integration", 0.3, "", request)
    )

    assert outputs == ["Hello", "Hello world"]
    assert len(mock_client.calls) == 1

    _args, kwargs = mock_client.calls[0]
    assert kwargs["model"] == "openai/gpt-integration"
    assert kwargs["stream"] is True

    messages = kwargs["messages"]
    assert messages[-1]["content"] == expected_message
    assert messages[0]["role"] == "system"


def test_chat_stream_error_path(mock_client, conversation_state, monkeypatch):
    monkeypatch.setattr(main, "logger", _stub_logger())
    monkeypatch.setattr(main.limiter, "check", lambda _ip: True)

    mock_client.queue_exception(RuntimeError("boom"))

    request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.2"))
    outputs = list(
        main.chat_fn("trigger", conversation_state, "openai/gpt-integration", 0.1, "", request)
    )

    assert outputs == ["[API Error] RuntimeError: boom"]
    assert len(mock_client.calls) == 1

    _args, kwargs = mock_client.calls[0]
    assert kwargs["stream"] is True


def test_chat_rate_limit(monkeypatch, mock_client, conversation_state):
    monkeypatch.setattr(main, "logger", _stub_logger())
    monkeypatch.setattr(utils.RateLimiter, "check", lambda self, _ip: False)
    monkeypatch.setattr(main, "limiter", utils.RateLimiter(capacity_per_min=1))
    assert main.limiter.check("test-client") is False

    request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.3"))
    outputs = list(
        main.chat_fn("hello", conversation_state, "openai/gpt-integration", 0.1, "", request)
    )

    assert outputs == ["Rate limit exceeded. Please slow down and try again."]
    assert mock_client.calls == []
