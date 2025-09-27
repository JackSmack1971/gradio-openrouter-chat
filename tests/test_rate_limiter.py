"""Regression tests for the in-memory RateLimiter implementation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

# Ensure the application modules are importable when tests run from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils
from utils import RateLimiter


@dataclass
class _TimeStub:
    """Simple controllable clock for deterministic rate limiter tests."""

    now: float = 0.0

    def time(self) -> float:  # pragma: no cover - trivial getter
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def time_stub(monkeypatch: pytest.MonkeyPatch) -> _TimeStub:
    stub = _TimeStub()
    # Replace the time module used inside utils with our controllable stub.
    monkeypatch.setattr(utils, "time", stub, raising=False)
    return stub


def test_rate_limiter_per_ip_buckets_are_independent(time_stub: _TimeStub) -> None:
    """Requests from different IPs should not affect each other's allowance."""

    limiter = RateLimiter(capacity_per_min=2)

    ip_one = "203.0.113.1"
    ip_two = "203.0.113.2"

    assert limiter.check(ip_one) is True
    assert limiter.check(ip_one) is True
    assert limiter.check(ip_one) is False

    # A different IP address should still have a full bucket.
    assert limiter.check(ip_two) is True
    assert limiter.check(ip_two) is True
    assert limiter.check(ip_two) is False

    # After enough simulated time passes, the first IP should be allowed again.
    time_stub.advance(60.0)
    assert limiter.check(ip_one) is True
