"""
Resilience Patterns for DataMesh.AI A2A Communication

Provides:
- Retry with exponential backoff and jitter
- Circuit breaker to prevent cascading failures
- Request deduplication with idempotency keys
- Timeout management
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Retry with Exponential Backoff
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    retryable_status_codes: tuple = (502, 503, 504, 429)


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception


def calculate_delay(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    backoff_multiplier: float,
    jitter: bool,
) -> float:
    """Calculate delay for the next retry attempt with optional jitter."""
    delay = initial_delay * (backoff_multiplier ** attempt)
    delay = min(delay, max_delay)

    if jitter:
        # Add random jitter: 0.5x to 1.5x of calculated delay
        delay = delay * (0.5 + random.random())

    return delay


def with_retry(config: RetryConfig | None = None):
    """
    Decorator that adds retry logic with exponential backoff.

    Usage:
        @with_retry(RetryConfig(max_retries=3))
        def call_api():
            ...
    """
    config = config or RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # Check if result indicates a retryable status
                    if isinstance(result, dict):
                        status_code = result.get("status_code", 200)
                        if status_code in config.retryable_status_codes:
                            raise RetryableStatusError(f"Status {status_code}")

                    return result

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = calculate_delay(
                            attempt,
                            config.initial_delay_seconds,
                            config.max_delay_seconds,
                            config.backoff_multiplier,
                            config.jitter,
                        )
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {config.max_retries + 1} attempts failed")

            raise RetryError(
                f"All {config.max_retries + 1} retry attempts exhausted",
                last_exception,
            )

        return wrapper
    return decorator


class RetryableStatusError(Exception):
    """Exception for retryable HTTP status codes."""
    pass


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes before closing
    timeout_seconds: float = 30.0  # Time before trying again
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    State transitions:
    - CLOSED: Normal operation, tracks failures
    - OPEN: After failure_threshold failures, rejects all calls
    - HALF_OPEN: After timeout, allows one test call
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout_seconds:
                        logger.info(f"Circuit {self.name}: OPEN → HALF_OPEN (timeout passed)")
                        self._state = CircuitState.HALF_OPEN
                        self._success_count = 0

            return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(f"Circuit {self.name}: HALF_OPEN → CLOSED (recovered)")
                    self._state = CircuitState.CLOSED
                    self._success_count = 0

    def record_failure(self, exception: Exception | None = None) -> None:
        """Record a failed call."""
        # Check if exception is excluded
        if exception and isinstance(exception, self.config.excluded_exceptions):
            return

        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit {self.name}: HALF_OPEN → OPEN (test failed)")
                self._state = CircuitState.OPEN
                self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(f"Circuit {self.name}: CLOSED → OPEN (threshold reached)")
                    self._state = CircuitState.OPEN
                    self._failure_count = 0

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        state = self.state  # This may transition OPEN → HALF_OPEN
        return state != CircuitState.OPEN

    def stats(self) -> dict[str, Any]:
        """Return circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure": self._last_failure_time,
            }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def with_circuit_breaker(circuit: CircuitBreaker):
    """
    Decorator that wraps a function with circuit breaker protection.

    Usage:
        breaker = CircuitBreaker("my-service")

        @with_circuit_breaker(breaker)
        def call_service():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not circuit.allow_request():
                raise CircuitOpenError(
                    f"Circuit {circuit.name} is open, request rejected"
                )

            try:
                result = func(*args, **kwargs)

                # Check for error status in result
                if isinstance(result, dict):
                    if result.get("status") == "ERROR":
                        circuit.record_failure()
                        return result

                circuit.record_success()
                return result

            except Exception as e:
                circuit.record_failure(e)
                raise

        return wrapper
    return decorator


# =============================================================================
# Idempotency / Request Deduplication
# =============================================================================

@dataclass
class IdempotencyEntry:
    """Cached result for an idempotent request."""
    key: str
    result: Any
    timestamp: float
    expires_at: float


class IdempotencyStore:
    """
    In-memory store for idempotent request results.

    Ensures that duplicate requests return the same result
    without re-executing the operation.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self._cache: dict[str, IdempotencyEntry] = {}
        self._lock = threading.RLock()

    @staticmethod
    def generate_key(request: dict[str, Any]) -> str:
        """Generate an idempotency key from a request."""
        # Use explicit idempotency key if provided
        if "idempotencyKey" in request:
            return request["idempotencyKey"]

        # Otherwise, hash the request content
        content = {
            "capability": request.get("capability"),
            "payload": request.get("payload"),
            "user": request.get("user", {}).get("id"),
        }
        content_str = str(sorted(content.items()))
        return hashlib.sha256(content_str.encode()).hexdigest()[:32]

    def get(self, key: str) -> Any | None:
        """Get cached result if not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                if time.time() < entry.expires_at:
                    logger.debug(f"Idempotency hit for key: {key[:16]}...")
                    return entry.result
                else:
                    del self._cache[key]
            return None

    def set(self, key: str, result: Any) -> None:
        """Cache a result."""
        with self._lock:
            self._cache[key] = IdempotencyEntry(
                key=key,
                result=result,
                timestamp=time.time(),
                expires_at=time.time() + self.ttl,
            )

    def cleanup(self) -> int:
        """Remove expired entries."""
        with self._lock:
            now = time.time()
            expired_keys = [k for k, v in self._cache.items() if v.expires_at < now]
            for k in expired_keys:
                del self._cache[k]
            return len(expired_keys)

    def stats(self) -> dict[str, int]:
        """Return store statistics."""
        with self._lock:
            now = time.time()
            valid = sum(1 for v in self._cache.values() if v.expires_at > now)
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid,
                "expired_entries": len(self._cache) - valid,
            }


# =============================================================================
# Combined Resilient Client
# =============================================================================

class ResilientClient:
    """
    HTTP client with built-in resilience patterns.

    Combines:
    - Circuit breaker per endpoint
    - Retry with exponential backoff
    - Request deduplication
    - Timeout management
    """

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        idempotency_ttl: int = 3600,
    ):
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.idempotency_store = IdempotencyStore(ttl_seconds=idempotency_ttl)

        # Circuit breaker per endpoint
        self._circuits: dict[str, CircuitBreaker] = {}
        self._circuits_lock = threading.Lock()

    def get_circuit(self, endpoint: str) -> CircuitBreaker:
        """Get or create circuit breaker for an endpoint."""
        with self._circuits_lock:
            if endpoint not in self._circuits:
                self._circuits[endpoint] = CircuitBreaker(endpoint, self.circuit_config)
            return self._circuits[endpoint]

    def call(
        self,
        endpoint: str,
        request: dict[str, Any],
        call_fn: Callable[[dict[str, Any]], dict[str, Any]],
        use_idempotency: bool = True,
    ) -> dict[str, Any]:
        """
        Make a resilient call to an endpoint.

        Args:
            endpoint: Endpoint identifier (e.g., "sql-agent")
            request: Request payload
            call_fn: Function to make the actual call
            use_idempotency: Whether to use request deduplication

        Returns:
            Response from the endpoint
        """
        # Check idempotency cache first
        idempotency_key = None
        if use_idempotency:
            idempotency_key = self.idempotency_store.generate_key(request)
            cached = self.idempotency_store.get(idempotency_key)
            if cached is not None:
                return {**cached, "_idempotency_hit": True}

        # Get circuit breaker
        circuit = self.get_circuit(endpoint)

        # Check circuit
        if not circuit.allow_request():
            return {
                "status": "ERROR",
                "error": f"Circuit breaker open for {endpoint}",
                "errorCode": "CIRCUIT_OPEN",
            }

        # Make call with retry
        last_error = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = call_fn(request)

                # Check result status
                if isinstance(result, dict):
                    if result.get("status") == "ERROR":
                        error_code = result.get("errorCode", "")
                        # Don't retry validation errors
                        if error_code in ("VALIDATION_ERROR", "UNAUTHORIZED"):
                            circuit.record_success()  # Not a circuit-breaker failure
                            return result

                        # Record as failure and retry
                        circuit.record_failure()
                        if attempt < self.retry_config.max_retries:
                            delay = calculate_delay(
                                attempt,
                                self.retry_config.initial_delay_seconds,
                                self.retry_config.max_delay_seconds,
                                self.retry_config.backoff_multiplier,
                                self.retry_config.jitter,
                            )
                            logger.warning(
                                f"[{endpoint}] Attempt {attempt + 1} failed, "
                                f"retrying in {delay:.2f}s..."
                            )
                            time.sleep(delay)
                            continue

                        return result

                # Success
                circuit.record_success()

                # Cache result for idempotency
                if use_idempotency and idempotency_key:
                    self.idempotency_store.set(idempotency_key, result)

                return result

            except Exception as e:
                last_error = e
                circuit.record_failure(e)

                if attempt < self.retry_config.max_retries:
                    delay = calculate_delay(
                        attempt,
                        self.retry_config.initial_delay_seconds,
                        self.retry_config.max_delay_seconds,
                        self.retry_config.backoff_multiplier,
                        self.retry_config.jitter,
                    )
                    logger.warning(
                        f"[{endpoint}] Attempt {attempt + 1} failed with {e}, "
                        f"retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)

        # All retries exhausted
        return {
            "status": "ERROR",
            "error": f"All {self.retry_config.max_retries + 1} attempts failed: {last_error}",
            "errorCode": "RETRY_EXHAUSTED",
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all components."""
        return {
            "circuits": {
                name: breaker.stats()
                for name, breaker in self._circuits.items()
            },
            "idempotency": self.idempotency_store.stats(),
        }
