from __future__ import annotations  # Fixed: double underscores, not asterisks

import contextlib
import contextvars
import csv
import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

SANITIZE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


# --- Correlation ID helpers -------------------------------------------------
_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)
_DEFAULT_HEADER_NAME = "x-request-id"


def generate_correlation_id() -> str:
    """Return a new UUID4 string suitable for trace/correlation identifiers."""

    return str(uuid.uuid4())


def validate_correlation_id(value: str | None) -> bool:
    """Return ``True`` when *value* looks like a valid UUID string."""

    if not value:
        return False
    try:
        uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def get_correlation_id(default: str | None = None) -> str | None:
    """Fetch the correlation ID stored in the current context.

    Parameters
    ----------
    default:
        Value returned when the context does not yet contain an identifier.
    """

    value = _correlation_id_var.get()
    return value or default


def set_correlation_id(correlation_id: str) -> str:
    """Persist *correlation_id* in the current context after validation."""

    if not validate_correlation_id(correlation_id):
        raise ValueError("correlation_id must be a valid UUID string")
    _correlation_id_var.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Remove the correlation identifier from the active context."""

    _correlation_id_var.set(None)


@contextlib.contextmanager
def correlation_context(correlation_id: str | None = None):
    """Context manager that binds a correlation ID for the duration of a block.

    Usage pattern::

        with correlation_context():
            logger = get_logger(__name__)
            logger.info("message")  # records auto-include the correlation_id

    Passing an explicit ``correlation_id`` allows upstream middleware to
    propagate a known identifier.  When omitted, a new UUID4 is generated.
    """

    if correlation_id and validate_correlation_id(correlation_id):
        new_id = correlation_id
    else:
        # Generate a fresh identifier to avoid propagating untrusted values.
        new_id = generate_correlation_id()
    token = _correlation_id_var.set(new_id)
    try:
        yield new_id
    finally:
        _correlation_id_var.reset(token)


class CorrelationIdAdapter(logging.LoggerAdapter):
    """Attach the active correlation ID to every log record."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        correlation_id = get_correlation_id()
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("correlation_id", correlation_id)
        return msg, kwargs


def get_logger(name: str | None = None) -> CorrelationIdAdapter:
    """Return a ``LoggerAdapter`` that injects the current correlation ID.

    The adapter preserves familiar ``logging`` semantics while ensuring
    downstream handlers can rely on ``record.correlation_id`` being present.
    Modules should call this helper rather than ``logging.getLogger`` directly.
    """

    base_logger = logging.getLogger(name or __name__)
    return CorrelationIdAdapter(base_logger, {})


def extract_correlation_id_from_request(
    request: Any | None,
    *,
    header_name: str = _DEFAULT_HEADER_NAME,
) -> str | None:
    """Fetch a correlation ID from the provided Gradio/Starlette request object."""

    if request is None:
        return None

    # ``gr.Request`` provides Starlette's ``headers`` and ``query_params``
    # interfaces. Both are case-insensitive mappings.
    header_value = None
    headers = getattr(request, "headers", None)
    if headers is not None:
        try:
            header_value = headers.get(header_name) or headers.get(header_name.upper())
        except AttributeError:
            header_value = (
                headers.get(header_name) if isinstance(headers, dict) else None
            )

    candidate = header_value or None
    if candidate and validate_correlation_id(candidate):
        return candidate

    # Fall back to query parameters when no header is present.
    params = getattr(request, "query_params", None)
    if params is not None:
        try:
            candidate = params.get(header_name) or params.get("correlation_id")
        except AttributeError:
            candidate = params.get(header_name) if isinstance(params, dict) else None
        if candidate and validate_correlation_id(candidate):
            return candidate

    return None


@contextlib.contextmanager
def bind_request_correlation_id(
    request: Any | None,
    *,
    header_name: str = _DEFAULT_HEADER_NAME,
):
    """Middleware-style helper to bind a request correlation ID.

    Example
    -------
    ```python
    def handler(request: gr.Request):
        with bind_request_correlation_id(request):
            logger = get_logger(__name__)
            logger.info("Handling request")
            ...
    ```

    The helper accepts ``None`` for compatibility with tests or Gradio
    callbacks that omit a request object.  A new UUID is generated when the
    incoming message lacks an identifier.
    """

    existing = extract_correlation_id_from_request(request, header_name=header_name)
    with correlation_context(existing):
        yield get_correlation_id()


def ensure_correlation_id(request: Any | None = None) -> str:
    """Ensure a correlation ID is available, storing it in the current context."""

    existing = extract_correlation_id_from_request(request)
    if existing:
        set_correlation_id(existing)
        return existing
    return set_correlation_id(generate_correlation_id())


def sanitize_text(s: str, max_chars: int) -> str:
    s = SANITIZE_RE.sub("", s or "")
    return s[:max_chars]


def trim_history(messages: list[dict], max_messages: int) -> list[dict]:
    # keep the latest N messages (system included if present)
    if len(messages) <= max_messages:
        return messages
    # preserve first system message if exists
    system = [m for m in messages if m["role"] == "system"][:1]
    rest = [m for m in messages if m["role"] != "system"]
    return (system + rest[-max_messages:]) if system else rest[-max_messages:]


# --- Simple token-bucket per-IP limiter (in-memory) ---
class RateLimiter:
    def __init__(
        self, capacity_per_min: int
    ):  # Fixed: double underscores, not asterisks
        self.capacity = capacity_per_min
        self.allowance: dict[str, float] = defaultdict(lambda: float(capacity_per_min))
        self.last_check: dict[str, float] = defaultdict(time.time)
        self.lock = threading.Lock()  # Correct usage per Python docs

    def check(self, key: str) -> bool:
        now = time.time()
        with self.lock:  # Using context manager - correct per Python threading docs
            last = self.last_check[key]
            elapsed = now - last
            self.last_check[key] = now
            # Token bucket algorithm: replenish tokens based on elapsed time
            self.allowance[key] = min(
                self.capacity, self.allowance[key] + elapsed * (self.capacity / 60.0)
            )
            if self.allowance[key] < 1.0:
                return False
            self.allowance[key] -= 1.0
            return True


# --- Persistence / Analytics ---
DATA_DIR = Path(".data")
DATA_DIR.mkdir(exist_ok=True)
LOG_CSV = DATA_DIR / "usage.csv"


def log_usage(row: dict) -> None:
    # Initialize CSV with headers if it doesn't exist
    if not LOG_CSV.exists():
        LOG_CSV.write_text(
            "ts,ip,model,input_tokens,output_tokens,latency_ms,cost_estimate\n",
            encoding="utf-8",
        )

    # Append usage data to CSV
    with LOG_CSV.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(
            f,
            fieldnames=[
                "ts",
                "ip",
                "model",
                "input_tokens",
                "output_tokens",
                "latency_ms",
                "cost_estimate",
            ],
        ).writerow(row)


def export_conversation(history: list[dict]) -> str:
    # Export conversation history as timestamped JSON file
    path = DATA_DIR / f"chat_{int(time.time())}.json"
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
