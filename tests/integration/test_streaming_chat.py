from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest

import main

pytestmark = pytest.mark.integration


def test_chat_fn_streams_incremental_tokens(
    fake_openai_stream,
    mock_openai_client,
    app_storage,
):
    """Ensure ``chat_fn`` yields incremental completions and logs usage."""

    stream = fake_openai_stream(["Hello", " world", None])
    mock_openai_client.queue_stream(stream)

    request = SimpleNamespace(client=SimpleNamespace(host="203.0.113.5"))

    generator = main.chat_fn(
        message="Hi there",
        history=[],
        model=main.settings.default_model,
        temperature=0.1,
        system_prompt="System safety",
        request=request,
    )

    outputs = list(generator)

    # Streaming should provide partial updates culminating in the final message.
    assert outputs == ["Hello", "Hello world"]
    assert mock_openai_client.calls, "Expected OpenRouter client to be invoked"

    # Usage logging should capture the request in the redirected CSV file.
    with app_storage.usage_log.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert rows[0] == [
        "ts",
        "ip",
        "model",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "cost_estimate",
    ]
    assert len(rows) == 2
    assert rows[1][1] == "203.0.113.5"
    assert rows[1][2] == main.settings.default_model
