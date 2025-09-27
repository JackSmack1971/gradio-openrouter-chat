from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Iterator, List

from types import SimpleNamespace

import pytest
import pytest_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

try:  # pragma: no cover - dependency shim for tests
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _ProcessorFormatter:
        wrap_for_formatter = staticmethod(lambda *args, **kwargs: None)

    structlog = SimpleNamespace(  # type: ignore[assignment]
        processors=SimpleNamespace(  # type: ignore[arg-type]
            TimeStamper=lambda fmt, key: (
                lambda logger, method, event_dict: event_dict
            ),
            JSONRenderer=lambda: (
                lambda logger, method, event_dict: event_dict
            ),
            add_log_level=lambda logger, method, event_dict: event_dict,
            StackInfoRenderer=lambda logger, method, event_dict: event_dict,
            format_exc_info=lambda logger, method, event_dict: event_dict,
        ),
        contextvars=SimpleNamespace(
            merge_contextvars=lambda logger=None, method=None, event_dict=None: event_dict
            or {},
            bind_contextvars=lambda **kwargs: None,
            clear_contextvars=lambda: None,
        ),
        dev=SimpleNamespace(
            ConsoleRenderer=lambda: (
                lambda logger, method, event_dict: event_dict
            )
        ),
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
    sys.modules.setdefault("structlog", structlog)
    sys.modules.setdefault("structlog.contextvars", structlog.contextvars)

import main  # noqa: E402
import utils  # noqa: E402
from utils import clear_correlation_id, sanitize_text  # noqa: E402


@pytest_asyncio.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Provide a dedicated asyncio loop for async tests.

    Ensures isolation from the global loop when pytest-asyncio is active.
    """

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def fixtures_path() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def conversation_state(fixtures_path: Path) -> List[dict[str, str]]:
    """Load a sanitized conversation history shared across tests."""

    json_path = fixtures_path / "minimal_conversation.json"
    yaml_path = fixtures_path / "minimal_conversation.yaml"

    with json_path.open("r", encoding="utf-8") as handle:
        json_history = json.load(handle)

    # JSON is valid YAML; reuse json parser to keep dependencies minimal.
    with yaml_path.open("r", encoding="utf-8") as handle:
        yaml_history = json.load(handle)

    assert yaml_history == json_history, "JSON and YAML fixtures must stay in sync"

    max_chars = getattr(main.settings, "max_input_chars", 4096)
    sanitized_history: List[dict[str, str]] = []
    for entry in json_history:
        sanitized_history.append(
            {
                "role": entry["role"],
                "content": sanitize_text(entry.get("content", ""), max_chars),
            }
        )
    return sanitized_history


@pytest.fixture
def fake_openai_stream() -> Callable[..., Iterator[SimpleNamespace]]:
    """Yield OpenAI-style chunks without performing external API calls for security.

    The generator mirrors the structure returned by ``client.chat.completions.create``
    so streaming code paths can be exercised safely during tests.
    """

    def _factory(
        contents: Iterable[str | None],
        *,
        usage: object | None = None,
    ) -> Iterator[SimpleNamespace]:
        tokens = list(contents)

        def _iterator() -> Iterator[SimpleNamespace]:
            for index, token in enumerate(tokens):
                delta = SimpleNamespace(content=token)
                choice = SimpleNamespace(delta=delta)
                chunk = SimpleNamespace(
                    id=f"test-chunk-{index}",
                    choices=[choice],
                )
                if usage is not None and index == len(tokens) - 1:
                    chunk.usage = usage
                yield chunk

        return _iterator()

    return _factory


class MockOpenAIClient:
    """Patchable OpenAI client that never leaves the test process for safety."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self._stream: Iterable[object] = ()
        self._exception: Exception | None = None

    def queue_stream(self, stream: Iterable[object]) -> None:
        """Configure the next streaming response while avoiding live requests."""

        self._stream = stream
        self._exception = None

    def queue_exception(self, exc: Exception) -> None:
        """Force the next call to raise ``exc`` instead of performing network I/O."""

        self._stream = ()
        self._exception = exc

    def create(self, *args, **kwargs) -> Iterable[object]:
        self.calls.append((args, kwargs))
        if self._exception is not None:
            raise self._exception
        return self._stream


@pytest.fixture
def mock_openai_client(monkeypatch) -> MockOpenAIClient:
    """Patch ``main.client`` and ``main.OpenAI`` so tests run without external traffic."""

    mock = MockOpenAIClient()
    client_stub = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=mock.create)
        )
    )

    monkeypatch.setattr(main, "client", client_stub, raising=False)
    monkeypatch.setattr(main, "OpenAI", lambda *args, **kwargs: client_stub)

    return mock


@pytest.fixture
def temporary_usage_log(tmp_path, monkeypatch) -> Path:
    """Redirect usage logging to a temp file to protect real analytics data."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    log_file = data_dir / "usage.csv"

    monkeypatch.setattr(utils, "DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(utils, "LOG_CSV", log_file, raising=False)

    return log_file


@pytest.fixture(autouse=True)
def reset_correlation_context():
    clear_correlation_id()
    yield
    clear_correlation_id()
