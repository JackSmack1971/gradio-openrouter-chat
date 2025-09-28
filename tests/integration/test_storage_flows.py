from __future__ import annotations

import json
from pathlib import Path

import pytest

import main

pytestmark = pytest.mark.integration


@pytest.fixture
def sample_history() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]


def test_save_and_load_round_trip(app_storage, sample_history):
    conversations = [
        {"id": "c1", "title": "Test", "history": sample_history},
    ]

    main.save_conversations(conversations)

    on_disk = json.loads(app_storage.conversations_file.read_text(encoding="utf-8"))
    assert on_disk == conversations

    loaded = main.load_conversations()
    assert loaded == conversations


def test_export_and_reimport_history(app_storage, sample_history):
    export_path = Path(main.export_handler(sample_history))

    assert export_path.exists()
    assert export_path.parent == app_storage.data_dir

    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported == sample_history

    imported = main.import_handler(str(export_path), sample_history)
    assert isinstance(imported, list)
    assert imported == sample_history
