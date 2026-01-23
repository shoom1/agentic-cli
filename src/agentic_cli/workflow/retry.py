"""Retry handling with exponential backoff.

Provides reusable retry logic for transient errors in async operations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, Awaitable, Callable, Any

from google.api_core import exceptions as google_exceptions

from agentic_cli.logging import Loggers

logger = Loggers.workflow()


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (exponential backoff)
        retryable_exceptions: Tuple of exception types to retry
    """

    max_retries: int = 3
    base_delay: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            google_exceptions.ServiceUnavailable,
            google_exceptions.ResourceExhausted,
        )
    )


@dataclass
class RetryState:
    """Tracks state during retry execution.

    Attributes:
        attempt: Current attempt number (0-indexed)
        last_error: Most recent exception encountered
    """

    attempt: int = 0
    last_error: Exception | None = None


class RetryHandler:
    """Handles retry logic with exponential backoff for async generators.

    Example:
        handler = RetryHandler(RetryConfig(max_retries=3))

        async def create_operation():
            return runner.run_async(...)

        async for event in handler.execute_with_retry(create_operation, on_retry):
            process(event)
    """

    def __init__(self, config: RetryConfig | None = None):
        """Initialize retry handler.

        Args:
            config: Retry configuration (uses defaults if not provided)
        """
        self.config = config or RetryConfig()

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt using exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        return self.config.base_delay * (2**attempt)

    def _is_retryable(self, error: Exception) -> bool:
        """Check if the error is retryable.

        Args:
            error: Exception to check

        Returns:
            True if the error should be retried
        """
        return isinstance(error, self.config.retryable_exceptions)

    def _should_retry(self, state: RetryState) -> bool:
        """Check if another retry should be attempted.

        Args:
            state: Current retry state

        Returns:
            True if another retry is allowed
        """
        return state.attempt < self.config.max_retries - 1

    async def execute_with_retry(
        self,
        create_operation: Callable[[], AsyncGenerator[Any, None]],
        on_retry: Callable[[int, float, Exception], Awaitable[None]] | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Execute an async generator operation with retry logic.

        Creates a new async generator on each attempt. When a retryable error
        occurs, calls the on_retry callback if provided, waits for the backoff
        period, and creates a new generator for the next attempt.

        Args:
            create_operation: Factory function that returns a new async generator
            on_retry: Optional callback invoked before each retry with
                     (attempt, delay, error) arguments

        Yields:
            Events from the async generator

        Raises:
            Exception: Re-raises the last error if all retries are exhausted
        """
        state = RetryState()

        while True:
            try:
                async_gen = create_operation()
                async for event in async_gen:
                    yield event
                # Success - exit retry loop
                return

            except Exception as e:
                state.last_error = e

                if not self._is_retryable(e):
                    raise

                if not self._should_retry(state):
                    logger.error(
                        "retry_exhausted",
                        attempts=state.attempt + 1,
                        max_retries=self.config.max_retries,
                        error=str(e),
                    )
                    raise

                delay = self._calculate_delay(state.attempt)

                logger.warning(
                    "retrying_after_error",
                    attempt=state.attempt + 1,
                    max_retries=self.config.max_retries,
                    delay=delay,
                    error_type=type(e).__name__,
                    error=str(e),
                )

                if on_retry:
                    await on_retry(state.attempt, delay, e)

                await asyncio.sleep(delay)
                state.attempt += 1
