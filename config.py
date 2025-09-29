from __future__ import annotations  # FIXED: Double underscores, not asterisks

import logging
import logging.config
import logging.handlers
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import uuid4

import structlog
from dotenv import load_dotenv
from structlog import contextvars as structlog_contextvars

load_dotenv()


def _as_bool(value: str | None, default: bool) -> bool:
    """Normalize environment-sourced truthy values."""

    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _as_int(value: str | None, default: int, *, minimum: int, maximum: int) -> int:
    """Convert environment variable to a bounded integer."""

    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _as_float(
    value: str | None, default: float, *, minimum: float, maximum: float
) -> float:
    """Convert environment variable to a bounded float value."""

    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _trusted_proxies_default() -> list[str]:
    """Parse trusted proxy addresses from environment."""

    return [
        proxy.strip()
        for proxy in os.getenv("TRUSTED_PROXIES", "").split(",")
        if proxy.strip() and proxy.strip() != ""
    ]


def _configure_logging_from_env() -> None:
    """Configure structured logging using stdlib + structlog."""

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_json = _as_bool(os.getenv("LOG_JSON"), True)
    log_path = os.getenv("LOG_PATH", "logs/app.log")
    rotation_when = os.getenv("LOG_ROTATION_WHEN", "midnight")
    rotation_interval = _as_int(
        os.getenv("LOG_ROTATION_INTERVAL"), 1, minimum=1, maximum=1440
    )
    rotation_backup_count = _as_int(
        os.getenv("LOG_BACKUP_COUNT"), 7, minimum=1, maximum=365
    )

    try:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        timestamper = structlog.processors.TimeStamper(fmt="iso", key="timestamp")
        shared_processors = [
            structlog_contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            timestamper,
        ]

        processor = (
            structlog.processors.JSONRenderer()
            if log_json
            else structlog.dev.ConsoleRenderer()
        )

        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structlog": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": processor,
                    "foreign_pre_chain": shared_processors,
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "structlog",
                },
                "file": {
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "formatter": "structlog",
                    "filename": str(log_file),
                    "when": rotation_when,
                    "interval": rotation_interval,
                    "backupCount": rotation_backup_count,
                    "encoding": "utf-8",
                },
            },
            "root": {
                "level": log_level,
                "handlers": ["default", "file"],
            },
        }

        logging.config.dictConfig(logging_config)

        structlog.configure(
            processors=shared_processors
            + [
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    except (
        Exception
    ):  # SECURITY: Ensure logging always initializes even if structlog fails
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
        logging.getLogger(__name__).exception("Failed to configure structured logging")


_configure_logging_from_env()


logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Settings:
    app_title: str = os.getenv("APP_TITLE", "OpenRouter AI Assistant")

    # IMPROVED: More explicit None handling
    referer: Optional[str] = os.getenv("APP_REFERER")
    x_title: Optional[str] = os.getenv("APP_X_TITLE")

    # IMPROVED: Better validation for required API key
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = "https://openrouter.ai/api/v1"  # per OpenRouter quickstart
    default_model: str = os.getenv("DEFAULT_MODEL", "openai/gpt-4o")

    # IMPROVED: Added validation for numeric values
    temperature: float = max(0.0, min(2.0, float(os.getenv("TEMPERATURE", "0.3"))))

    # Safety/perf - IMPROVED: Added bounds checking
    max_input_chars: int = max(
        100, min(50000, int(os.getenv("MAX_INPUT_CHARS", "8000")))
    )
    max_history_messages: int = max(
        1, min(200, int(os.getenv("MAX_HISTORY_MESSAGES", "40")))
    )

    # Rate limit (approx per-IP) - IMPROVED: Added bounds checking
    rate_limit_per_min: int = max(
        1, min(1000, int(os.getenv("RATE_LIMIT_REQUESTS_PER_MIN", "60")))
    )

    # Analytics
    enable_analytics: bool = _as_bool(os.getenv("ENABLE_ANALYTICS"), True)

    # Health endpoint controls
    health_check_enabled: bool = _as_bool(os.getenv("HEALTHCHECK_ENABLED"), True)
    health_check_timeout: float = _as_float(
        os.getenv("HEALTHCHECK_TIMEOUT"),
        5.0,
        minimum=1.0,
        maximum=60.0,
    )

    # Deploy
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = max(
        1024, min(65535, int(os.getenv("PORT", "7860")))
    )  # IMPROVED: Valid port range

    # Logging configuration (environment-driven)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_json: bool = _as_bool(os.getenv("LOG_JSON"), True)
    log_path: str = os.getenv("LOG_PATH", "logs/app.log")
    log_rotation_when: str = os.getenv("LOG_ROTATION_WHEN", "midnight")
    log_rotation_interval: int = _as_int(
        os.getenv("LOG_ROTATION_INTERVAL"), 1, minimum=1, maximum=1440
    )
    log_backup_count: int = _as_int(
        os.getenv("LOG_BACKUP_COUNT"), 7, minimum=1, maximum=365
    )

    # IMPROVED: More robust list parsing with filtering
    trusted_proxies: list[str] = field(default_factory=_trusted_proxies_default)

    def __post_init__(self):
        """Validate critical settings after initialization."""

        # ADDED: Runtime validation for required fields
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )

        # ADDED: Validate base_url format
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid base_url format: {self.base_url}")


def get_settings() -> Settings:
    """Get validated settings instance with structured logging."""

    correlation_id = str(uuid4())
    structlog_contextvars.bind_contextvars(correlation_id=correlation_id)
    try:
        return Settings()
    except ValueError as exc:
        logger.error(
            "configuration_validation_error",
            message="Configuration validation failed",
            error=str(exc),
            correlation_id=correlation_id,
            exc_info=True,
        )
        raise
    except Exception as exc:
        logger.exception(
            "unexpected_configuration_error",
            correlation_id=correlation_id,
        )
        raise ValueError("Failed to load configuration") from exc
    finally:
        structlog_contextvars.clear_contextvars()


# Create settings instance
settings = get_settings()
