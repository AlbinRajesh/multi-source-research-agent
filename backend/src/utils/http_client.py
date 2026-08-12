"""Shared HTTP client (connection pooling) + circuit breaker.

Adapted from the reference project's web_utils.py. Used by
content_extractor.py and any search provider that makes raw HTTP calls
(the Tavily/SearXNG SDKs manage their own connections, but page-content
fetching goes through this shared pooled client for speed).
"""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-service circuit breaker: CLOSED -> (N failures) -> OPEN ->
    (timeout elapses) -> HALF_OPEN -> (success) -> CLOSED."""

    name: str
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    half_open_max_calls: int = 1

    _failures: int = field(default=0, init=False)
    _successes: int = field(default=0, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and (time.time() - self._last_failure_time) >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"Circuit '{self.name}' -> HALF_OPEN")
        return self._state

    def can_execute(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        return False

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failures = 0
                self._successes = 0
                logger.info(f"Circuit '{self.name}' closed after recovery")
        else:
            self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit '{self.name}' reopened after half-open failure")
        elif self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit '{self.name}' opened after {self._failures} failures")

    def get_retry_after(self) -> float:
        if self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            return max(0, self.reset_timeout - elapsed)
        return self.reset_timeout


# =============================================================================
# Pooled HTTP client (singleton)
# =============================================================================

class HTTPClientManager:
    """Shared httpx.AsyncClient with connection pooling + HTTP/2.
    One instance per process — reused across all fetches for speed."""

    _instance: Optional["HTTPClientManager"] = None

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "HTTPClientManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=20,
                        keepalive_expiry=30.0,
                    ),
                    timeout=httpx.Timeout(15.0, connect=5.0),
                    follow_redirects=True,
                    http2=True,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
            return self._client

    async def close(self) -> None:
        async with self._lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                self._client = None


async def cleanup_http_client() -> None:
    """Call on application shutdown."""
    await HTTPClientManager.get_instance().close()


# =============================================================================
# URL safety validation
# =============================================================================

BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}


def is_valid_url(url: str) -> bool:
    """Reject internal/loopback targets and non-http(s) schemes before fetching."""
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        if result.scheme.lower() not in {"http", "https"}:
            return False
        hostname = (result.hostname or "").lower()
        if hostname in BLOCKED_HOSTS:
            return False
        return True
    except Exception:
        return False