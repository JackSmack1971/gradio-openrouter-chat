from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

try:  # pragma: no cover - dependency shim for test environment
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _ProcessorFormatter:
        wrap_for_formatter = staticmethod(lambda *args, **kwargs: None)

    structlog = SimpleNamespace(  # type: ignore[assignment]
        processors=SimpleNamespace(
            TimeStamper=lambda fmt, key: (lambda logger, method, event_dict: event_dict),
            JSONRenderer=lambda: (lambda logger, method, event_dict: event_dict),
            add_log_level=lambda logger, method, event_dict: event_dict,
            StackInfoRenderer=lambda logger, method, event_dict: event_dict,
            format_exc_info=lambda logger, method, event_dict: event_dict,
        ),
        contextvars=SimpleNamespace(
            merge_contextvars=lambda logger=None, method=None, event_dict=None: event_dict or {},
            bind_contextvars=lambda **kwargs: None,
            clear_contextvars=lambda: None,
        ),
        dev=SimpleNamespace(ConsoleRenderer=lambda: (lambda logger, method, event_dict: event_dict)),
        stdlib=SimpleNamespace(
            ProcessorFormatter=_ProcessorFormatter,
            BoundLogger=SimpleNamespace,
            LoggerFactory=lambda: None,
        ),
        configure=lambda **kwargs: None,
        get_logger=lambda *args, **kwargs: SimpleNamespace(
            info=lambda *a, **k: None,
            error=lambda *a, **k: None,
            exception=lambda *a, **k: None,
        ),
    )
    sys.modules["structlog"] = structlog
    sys.modules["structlog.contextvars"] = structlog.contextvars

import main  # noqa: E402


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def info(self, message: str, **kwargs) -> None:
        self.records.append(("info", message, kwargs))

    def warning(self, message: str, **kwargs) -> None:
        self.records.append(("warning", message, kwargs))

    def exception(self, message: str, **kwargs) -> None:
        self.records.append(("exception", message, kwargs))


def _install_test_logger(monkeypatch, ensure_return: str = "123e4567-e89b-12d3-a456-426614174000"):
    logger = DummyLogger()

    def fake_ensure(request=None):
        fake_ensure.calls.append(request)
        return ensure_return

    fake_ensure.calls = []  # type: ignore[attr-defined]

    monkeypatch.setattr(main, "logger", logger)
    monkeypatch.setattr(main, "ensure_correlation_id", fake_ensure)
    return logger, fake_ensure


def test_chat_fn_emits_structured_logs(monkeypatch):
    logger, fake_ensure = _install_test_logger(monkeypatch)

    monkeypatch.setattr(main.limiter, "check", lambda _ip: True)

    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=11)

    class DummyChunk:
        def __init__(self, content: str | None, usage_obj=None):
            self.choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]
            if usage_obj is not None:
                self.usage = usage_obj

    def fake_stream(*_args, **_kwargs):
        yield DummyChunk("Hello", None)
        yield DummyChunk(None, usage)

    class DummyClient:
        def create(self, *args, **kwargs):
            fake_stream.calls.append((args, kwargs))
            return fake_stream()

    fake_stream.calls = []  # type: ignore[attr-defined]

    monkeypatch.setattr(main.client.chat.completions, "create", DummyClient().create)

    request = SimpleNamespace(client=SimpleNamespace(host="1.2.3.4"))

    output = list(main.chat_fn("Hello", [], "openai/test", 0.2, "", request))

    assert output[-1] == "Hello"
    assert fake_ensure.calls == [request]

    events = {(level, name): payload for level, name, payload in logger.records}
    assert ("info", "chat_request_received") in events
    assert events[("info", "chat_request_received")]["selected_model"] == "openai/test"
    assert ("info", "openrouter_request_started") in events
    assert ("info", "openrouter_request_completed") in events


def test_new_conversation_logs_creation(monkeypatch):
    logger, fake_ensure = _install_test_logger(monkeypatch)

    conversations, convo_id, _history, _dropdown = main.new_conversation([], None)

    assert conversations and convo_id
    assert fake_ensure.calls == [None]
    assert any(name == "conversation_created" for _level, name, _ in logger.records)


def test_delete_conversation_logs_state(monkeypatch):
    logger, fake_ensure = _install_test_logger(monkeypatch)

    convo = {"id": "abc", "title": "Conversation 1", "history": []}
    conversations = [convo]
    updated = main.delete_conversation("abc", conversations, "abc")

    assert fake_ensure.calls == [None]
    assert any(name == "conversation_delete_requested" for _level, name, _ in logger.records)
    assert any(name == "conversation_deleted" for _level, name, _ in logger.records)
    assert updated[1] is None


def test_export_and_import_handlers_log(monkeypatch, tmp_path):
    logger, fake_ensure = _install_test_logger(monkeypatch)

    export_path = tmp_path / "export.json"
    monkeypatch.setattr(main, "export_conversation", lambda history: str(export_path))

    result_path = main.export_handler([{"role": "user", "content": "hi"}])
    assert result_path == str(export_path)
    assert any(name == "conversation_export_started" for _level, name, _ in logger.records)
    assert any(name == "conversation_export_completed" for _level, name, _ in logger.records)

    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps([{ "role": "assistant", "content": "hello" }]), encoding="utf-8")

    records_count_before = len(logger.records)
    imported = main.import_handler(str(history_path), [])
    assert isinstance(imported, list) and imported[0]["role"] == "assistant"
    assert len(logger.records) > records_count_before
    assert any(name == "conversation_import_started" for _level, name, _ in logger.records)
    assert any(name == "conversation_import_completed" for _level, name, _ in logger.records)

    # ensure ensure_correlation_id invoked for each handler call
    assert fake_ensure.calls.count(None) >= 2
