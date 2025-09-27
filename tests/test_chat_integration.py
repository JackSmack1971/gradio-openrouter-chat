from __future__ import annotations

from types import SimpleNamespace

import main
import utils
from tests.fixtures.chat import request_stub
from utils import sanitize_text


def _stub_logger():
    return SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )


def test_chat_streaming_success(
    mock_openai_client, fake_openai_stream, conversation_state, monkeypatch
):
    monkeypatch.setattr(main, "logger", _stub_logger())
    monkeypatch.setattr(main.limiter, "check", lambda _ip: True)

    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=7)
    mock_openai_client.queue_stream(
        fake_openai_stream(["Hello", " world"], usage=usage)
    )

    noisy_message = "Streaming sanitized message\x07"
    expected_message = sanitize_text(noisy_message, main.settings.max_input_chars)
    request = request_stub("10.0.0.1")

    outputs = list(
        main.chat_fn(
            noisy_message,
            conversation_state,
            "openai/gpt-integration",
            0.3,
            "",
            request,
        )
    )

    assert outputs == ["Hello", "Hello world"]
    assert len(mock_openai_client.calls) == 1

    _args, kwargs = mock_openai_client.calls[0]
    assert kwargs["model"] == "openai/gpt-integration"
    assert kwargs["stream"] is True

    messages = kwargs["messages"]
    assert messages[-1]["content"] == expected_message
    assert messages[0]["role"] == "system"


def test_chat_stream_error_path(mock_openai_client, conversation_state, monkeypatch):
    monkeypatch.setattr(main, "logger", _stub_logger())
    monkeypatch.setattr(main.limiter, "check", lambda _ip: True)

    mock_openai_client.queue_exception(RuntimeError("boom"))

    request = request_stub("10.0.0.2")
    outputs = list(
        main.chat_fn(
            "trigger", conversation_state, "openai/gpt-integration", 0.1, "", request
        )
    )

    assert outputs == ["[API Error] RuntimeError: boom"]
    assert len(mock_openai_client.calls) == 1

    _args, kwargs = mock_openai_client.calls[0]
    assert kwargs["stream"] is True


def test_chat_rate_limit(monkeypatch, mock_openai_client, conversation_state):
    monkeypatch.setattr(main, "logger", _stub_logger())
    monkeypatch.setattr(utils.RateLimiter, "check", lambda self, _ip: False)
    monkeypatch.setattr(main, "limiter", utils.RateLimiter(capacity_per_min=1))
    assert main.limiter.check("test-client") is False

    request = request_stub("10.0.0.3")
    outputs = list(
        main.chat_fn(
            "hello", conversation_state, "openai/gpt-integration", 0.1, "", request
        )
    )

    assert outputs == ["Rate limit exceeded. Please slow down and try again."]
    assert mock_openai_client.calls == []
