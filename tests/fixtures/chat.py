from __future__ import annotations

"""Factories that create offline chat artefacts for isolated testing."""

from typing import Iterable
from types import SimpleNamespace


def build_history(messages: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    """Assemble a chat history without contacting remote services for security."""

    return [{"role": role, "content": content} for role, content in messages]


def request_stub(host: str = "127.0.0.1") -> SimpleNamespace:
    """Create a minimal request object that never leaves the test process."""

    return SimpleNamespace(client=SimpleNamespace(host=host))


def stream_chunk(content: str | None = None, usage: object | None = None) -> SimpleNamespace:
    """Build a streaming delta chunk mirroring OpenAI responses without network I/O."""

    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta)
    chunk = SimpleNamespace(choices=[choice])
    if usage is not None:
        chunk.usage = usage
    return chunk
