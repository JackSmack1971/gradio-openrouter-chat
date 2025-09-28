from __future__ import annotations

import concurrent.futures
import threading

import pytest

from utils import RateLimiter, time

pytestmark = pytest.mark.integration


def test_rate_limiter_is_thread_safe(monkeypatch):
    """Multiple concurrent calls should respect the configured capacity."""

    limiter = RateLimiter(3)
    key = "198.51.100.8"

    base_time = time.time()
    lock = threading.Lock()

    def fixed_time() -> float:
        with lock:
            return base_time

    # SECURITY: ensure deterministic timing so replenishment cannot bypass limits.
    monkeypatch.setattr(time, "time", fixed_time)

    def invoke() -> bool:
        return limiter.check(key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _: invoke(), range(10)))

    # Allowing for scheduling jitter, we only require "at most" capacity successes.
    assert 0 < results.count(True) <= limiter.capacity
    assert results.count(False) >= len(results) - limiter.capacity
