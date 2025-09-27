from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace

import pytest

import config


ENV_KEYS = {
    "APP_TITLE",
    "APP_REFERER",
    "APP_X_TITLE",
    "OPENROUTER_API_KEY",
    "DEFAULT_MODEL",
    "TEMPERATURE",
    "MAX_INPUT_CHARS",
    "MAX_HISTORY_MESSAGES",
    "RATE_LIMIT_REQUESTS_PER_MIN",
    "ENABLE_ANALYTICS",
    "HOST",
    "PORT",
    "LOG_LEVEL",
    "LOG_JSON",
    "LOG_PATH",
    "LOG_ROTATION_WHEN",
    "LOG_ROTATION_INTERVAL",
    "LOG_BACKUP_COUNT",
    "TRUSTED_PROXIES",
}


@pytest.fixture(autouse=True)
def clear_relevant_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def reload_config(monkeypatch: pytest.MonkeyPatch):
    def _reload() -> config:
        return importlib.reload(config)

    yield _reload

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    importlib.reload(config)


def test_as_bool_edge_cases() -> None:
    assert config._as_bool(" TRUE ", False) is True
    assert config._as_bool("0", True) is False
    assert config._as_bool("YeS", False) is True
    assert config._as_bool(" off ", True) is False
    assert config._as_bool(None, True) is True


def test_as_int_clamps_and_defaults() -> None:
    assert config._as_int("42", 10, minimum=0, maximum=50) == 42
    assert config._as_int("-100", 5, minimum=0, maximum=10) == 0
    assert config._as_int("9999", 5, minimum=0, maximum=10) == 10
    assert config._as_int("not-an-int", 7, minimum=0, maximum=5) == 5


def test_trusted_proxies_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTED_PROXIES", " 10.0.0.1 , , example.com ,, 127.0.0.1 ")
    assert config._trusted_proxies_default() == [
        "10.0.0.1",
        "example.com",
        "127.0.0.1",
    ]


def test_settings_initialization_with_env(monkeypatch: pytest.MonkeyPatch, reload_config) -> None:
    monkeypatch.setenv("APP_TITLE", "Configured App")
    monkeypatch.setenv("OPENROUTER_API_KEY", "live-key")
    monkeypatch.setenv("TEMPERATURE", "0.9")
    monkeypatch.setenv("MAX_INPUT_CHARS", "1200")
    monkeypatch.setenv("MAX_HISTORY_MESSAGES", "10")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MIN", "100")
    monkeypatch.setenv("ENABLE_ANALYTICS", "no")
    monkeypatch.setenv("TRUSTED_PROXIES", "1.1.1.1, 2.2.2.2")

    module = reload_config()
    settings = module.Settings()

    assert settings.app_title == "Configured App"
    assert settings.api_key == "live-key"
    assert pytest.approx(settings.temperature, rel=1e-9) == 0.9
    assert settings.max_input_chars == 1200
    assert settings.max_history_messages == 10
    assert settings.rate_limit_per_min == 100
    assert settings.enable_analytics is False
    assert settings.trusted_proxies == ["1.1.1.1", "2.2.2.2"]


def test_settings_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable is required"):
        config.Settings(api_key="")


def test_settings_invalid_base_url_raises() -> None:
    with pytest.raises(ValueError, match="Invalid base_url format: ftp://bad"):
        config.Settings(api_key="abc", base_url="ftp://bad")


def test_logging_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, SimpleNamespace] = {}

    def _raise_configure(**_: object) -> None:
        raise RuntimeError("boom")

    def _fake_basic_config(**kwargs: object) -> None:
        calls["basic_config"] = SimpleNamespace(kwargs=kwargs)

    captured = {}

    class _Logger:
        def __init__(self) -> None:
            self.handlers: list[object] = []

        def addHandler(self, handler: object) -> None:  # pragma: no cover - handler interactions
            self.handlers.append(handler)

        def removeHandler(self, handler: object) -> None:  # pragma: no cover - handler interactions
            if handler in self.handlers:
                self.handlers.remove(handler)

        def exception(self, message: str, *args: object, **kwargs: object) -> None:
            captured["message"] = message

    logger_stub = _Logger()

    monkeypatch.setattr(config.structlog, "configure", _raise_configure)
    monkeypatch.setattr(logging, "basicConfig", _fake_basic_config)
    monkeypatch.setattr(logging, "getLogger", lambda name=None: logger_stub)

    config._configure_logging_from_env()

    assert "basic_config" in calls
    assert captured["message"] == "Failed to configure structured logging"


def test_get_settings_clears_context(monkeypatch: pytest.MonkeyPatch, reload_config) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "context-key")
    module = reload_config()

    cleared: list[bool] = []
    monkeypatch.setattr(module.structlog_contextvars, "bind_contextvars", lambda **_: None)
    monkeypatch.setattr(module.structlog_contextvars, "clear_contextvars", lambda: cleared.append(True))

    settings = module.get_settings()

    assert settings.api_key == "context-key"
    assert cleared == [True]
