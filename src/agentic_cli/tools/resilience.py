"""Resilience patterns for tools.

Provides:
- retry: Decorator for automatic retry with exponential backoff
- RateLimiter: Token bucket rate limiter
- CircuitBreaker: Circuit breaker for failing services
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar, ParamSpec
from collections import deque

from agentic_cli.tools.registry import ToolError, ErrorCode


P = ParamSpec("P")
T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_errors: tuple[type[Exception], ...] | None = None,
    retryable_codes: tuple[str, ...] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for automatic retry with exponential backoff.

    Retries failed function calls with increasing delays between attempts.
    Only retries on specified exceptions or ToolError with recoverable=True.

    Args:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff (delay = base_delay * base^attempt)
        retryable_errors: Exception types to retry on (default: ConnectionError, TimeoutError)
        retryable_codes: ToolError codes to retry on

    Example:
        @retry(max_attempts=3, base_delay=1.0)
        def fetch_data(url: str) -> dict:
            ...

        @retry(retryable_codes=(ErrorCode.RATE_LIMITED, ErrorCode.SERVICE_UNAVAILABLE))
        def call_api() -> dict:
            ...
    """
    if retryable_errors is None:
        retryable_errors = (ConnectionError, TimeoutError, OSError)

    if retryable_codes is None:
        retryable_codes = (
            ErrorCode.RATE_LIMITED,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.TIMEOUT,
        )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except ToolError as e:
                    last_exception = e
                    if not _should_retry_tool_error(e, retryable_codes):
                        raise
                except retryable_errors as e:
                    last_exception = e
                except Exception:
                    raise  # Non-retryable exception

                if attempt < max_attempts - 1:
                    delay = min(base_delay * (exponential_base**attempt), max_delay)
                    time.sleep(delay)

            # All retries exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error")

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except ToolError as e:
                    last_exception = e
                    if not _should_retry_tool_error(e, retryable_codes):
                        raise
                except retryable_errors as e:
                    last_exception = e
                except Exception:
                    raise  # Non-retryable exception

                if attempt < max_attempts - 1:
                    delay = min(base_delay * (exponential_base**attempt), max_delay)
                    await asyncio.sleep(delay)

            # All retries exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def _should_retry_tool_error(error: ToolError, retryable_codes: tuple[str, ...]) -> bool:
    """Check if a ToolError should be retried."""
    return error.recoverable or error.error_code in retryable_codes


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    When a service fails repeatedly, the circuit "opens" and fails fast
    without calling the service. After a timeout, it allows a test request
    through to check if the service recovered.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open before closing
        timeout: Seconds to wait before transitioning from open to half-open
        name: Identifier for this circuit breaker

    Example:
        breaker = CircuitBreaker(failure_threshold=5, timeout=30)

        @breaker
        def call_external_api():
            ...
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    name: str = "default"

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout transition."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Use as decorator."""

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self.is_open:
                raise ToolError(
                    message=f"Circuit breaker '{self.name}' is open",
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                    recoverable=True,
                    details={"circuit_state": self.state.value},
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self.is_open:
                raise ToolError(
                    message=f"Circuit breaker '{self.name}' is open",
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                    recoverable=True,
                    details={"circuit_state": self.state.value},
                )

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper


@dataclass
class RateLimiter:
    """Token bucket rate limiter.

    Limits the rate of function calls using a token bucket algorithm.
    Tokens are added at a fixed rate, and each call consumes one token.

    Attributes:
        rate: Maximum calls per second
        burst: Maximum burst size (bucket capacity)
        name: Identifier for this rate limiter

    Example:
        limiter = RateLimiter(rate=10, burst=20)  # 10 calls/sec, burst of 20

        @limiter
        def api_call():
            ...
    """

    rate: float  # Tokens per second
    burst: int = 10  # Maximum bucket size
    name: str = "default"

    _tokens: float = field(default=0.0, init=False)
    _last_update: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        self._tokens = float(self.burst)
        self._last_update = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_update = now

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Acquire a token (synchronous).

        Args:
            blocking: If True, wait for token. If False, return immediately.
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if token acquired, False if not (only when blocking=False)
        """
        start_time = time.time()

        while True:
            self._refill()

            if self._tokens >= 1:
                self._tokens -= 1
                return True

            if not blocking:
                return False

            if timeout is not None and time.time() - start_time >= timeout:
                return False

            # Wait for next token
            wait_time = (1 - self._tokens) / self.rate
            time.sleep(min(wait_time, 0.1))

    async def acquire_async(
        self, blocking: bool = True, timeout: float | None = None
    ) -> bool:
        """Acquire a token (asynchronous).

        Args:
            blocking: If True, wait for token. If False, return immediately.
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if token acquired, False if not
        """
        start_time = time.time()

        async with self._lock:
            while True:
                self._refill()

                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

                if not blocking:
                    return False

                if timeout is not None and time.time() - start_time >= timeout:
                    return False

                # Wait for next token
                wait_time = (1 - self._tokens) / self.rate
                await asyncio.sleep(min(wait_time, 0.1))

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Use as decorator."""

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not self.acquire(blocking=True, timeout=30):
                raise ToolError(
                    message=f"Rate limit exceeded for '{self.name}'",
                    error_code=ErrorCode.RATE_LIMITED,
                    recoverable=True,
                )
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not await self.acquire_async(blocking=True, timeout=30):
                raise ToolError(
                    message=f"Rate limit exceeded for '{self.name}'",
                    error_code=ErrorCode.RATE_LIMITED,
                    recoverable=True,
                )
            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper


def with_timeout(seconds: float) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add timeout to a function.

    Args:
        seconds: Maximum execution time in seconds

    Example:
        @with_timeout(30)
        async def slow_operation():
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except asyncio.TimeoutError:
                raise ToolError(
                    message=f"Operation timed out after {seconds} seconds",
                    error_code=ErrorCode.TIMEOUT,
                    recoverable=True,
                    tool_name=func.__name__,
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore

        # For sync functions, we can't easily add timeout
        # Return as-is with a warning in docstring
        return func

    return decorator


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Use with tools to specify retry settings.
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_codes: tuple[str, ...] = (
        ErrorCode.RATE_LIMITED,
        ErrorCode.SERVICE_UNAVAILABLE,
        ErrorCode.TIMEOUT,
    )

    def as_decorator(self) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Convert config to retry decorator."""
        return retry(
            max_attempts=self.max_attempts,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            retryable_codes=self.retryable_codes,
        )
