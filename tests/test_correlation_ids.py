from __future__ import annotations

import logging

from utils import (
    bind_request_correlation_id,
    clear_correlation_id,
    correlation_context,
    ensure_correlation_id,
    extract_correlation_id_from_request,
    generate_correlation_id,
    get_correlation_id,
    get_logger,
    validate_correlation_id,
)


class DummyRequest:
    def __init__(self, headers=None, query_params=None):
        self.headers = headers or {}
        self.query_params = query_params or {}


def test_generate_and_validate_roundtrip():
    correlation_id = generate_correlation_id()
    assert validate_correlation_id(correlation_id)


def test_correlation_context_resets_previous_value():
    with correlation_context("123e4567-e89b-12d3-a456-426614174000"):
        assert get_correlation_id() == "123e4567-e89b-12d3-a456-426614174000"
        with correlation_context():
            nested_id = get_correlation_id()
            assert validate_correlation_id(nested_id)
        # after exiting nested context original ID restored
        assert get_correlation_id() == "123e4567-e89b-12d3-a456-426614174000"
    assert get_correlation_id() is None


def test_get_logger_attaches_correlation_id(monkeypatch):
    records: list[logging.LogRecord] = []

    handler = logging.Handler()

    def _capture(record: logging.LogRecord) -> None:
        records.append(record)

    handler.emit = _capture

    base_logger = logging.getLogger("utils-test")
    base_logger.handlers = [handler]
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = False

    adapter = get_logger("utils-test")

    known_id = "123e4567-e89b-12d3-a456-426614174000"
    with correlation_context(known_id):
        adapter.info("testing")

    assert records, "Expected a log record to be emitted"
    assert getattr(records[0], "correlation_id", None) == known_id


def test_extract_correlation_id_from_request_prefers_header():
    request = DummyRequest(
        headers={"x-request-id": "123e4567-e89b-12d3-a456-426614174000"}
    )
    assert (
        extract_correlation_id_from_request(request)
        == "123e4567-e89b-12d3-a456-426614174000"
    )


def test_extract_correlation_id_from_request_falls_back_to_params():
    request = DummyRequest(
        query_params={"correlation_id": "123e4567-e89b-12d3-a456-426614174000"}
    )
    assert (
        extract_correlation_id_from_request(request)
        == "123e4567-e89b-12d3-a456-426614174000"
    )


def test_bind_request_correlation_id_sets_context():
    request = DummyRequest(
        headers={"x-request-id": "123e4567-e89b-12d3-a456-426614174000"}
    )
    with bind_request_correlation_id(request) as correlation_id:
        assert correlation_id == "123e4567-e89b-12d3-a456-426614174000"
        assert get_correlation_id() == correlation_id
    assert get_correlation_id() is None


def test_ensure_correlation_id_generates_when_missing():
    clear_correlation_id()
    generated = ensure_correlation_id(None)
    assert validate_correlation_id(generated)
    clear_correlation_id()
    assert get_correlation_id() is None
