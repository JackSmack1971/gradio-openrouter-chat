from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import main

pytestmark = pytest.mark.e2e


def test_full_user_workflow(
    fake_openai_stream,
    mock_openai_client,
    app_storage,
):
    """Simulate conversation creation, messaging, export, and re-import."""

    conversations: list[dict] = []
    new_convo = main.create_new_conversation(conversations)
    conversations.append(new_convo)

    history = new_convo["history"]
    user_message = "How are you?"

    mock_openai_client.queue_stream(fake_openai_stream(["I'm", " great today!"]))
    request = SimpleNamespace(client=SimpleNamespace(host="192.0.2.55"))

    responses = list(
        main.chat_fn(
            message=user_message,
            history=history,
            model=main.settings.default_model,
            temperature=0.5,
            system_prompt="Stay positive",
            request=request,
        )
    )

    assistant_reply = responses[-1]
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_reply})

    main.save_conversations(conversations)

    # Verify persistence writes the expected structure
    persisted = json.loads(app_storage.conversations_file.read_text(encoding="utf-8"))
    assert persisted[0]["history"][-1]["content"] == assistant_reply

    export_path = Path(main.export_handler(history))
    assert export_path.exists()

    imported_history = main.import_handler(str(export_path), history)
    assert imported_history == history

    # Loading back from disk should match what we saved previously
    loaded_state = main.load_conversations()
    assert loaded_state[0]["history"][-1]["content"] == assistant_reply
