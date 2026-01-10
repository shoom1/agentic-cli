"""Structured logging configuration for agentic CLI applications.

Uses structlog for structured, context-rich logging that supports
both human-readable console output and machine-readable JSON format.
"""

import logging
import sys
from typing import TYPE_CHECKING

import structlog
from structlog.types import Processor

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings


def configure_logging(settings: "BaseSettings | None" = None) -> None:
    """Configure structured logging based on settings.

    Args:
        settings: Application settings. If None, uses defaults.
    """
    # Determine log level
    log_level = logging.WARNING
    log_format = "console"

    if settings is not None:
        log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        log_format = settings.log_format

    # Common processors for all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        # JSON format for production/log aggregation
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging for third-party libs
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a logger instance.

    Args:
        name: Optional logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name) if name else structlog.get_logger()


def bind_context(**kwargs: object) -> None:
    """Bind context variables to all subsequent log calls in the current context.

    Example:
        bind_context(session_id="abc123", user_id="user1")
        logger.info("processing")  # Will include session_id and user_id

    Args:
        **kwargs: Key-value pairs to bind to logging context
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def unbind_context(*keys: str) -> None:
    """Remove specific keys from the logging context.

    Args:
        *keys: Keys to remove from context
    """
    structlog.contextvars.unbind_contextvars(*keys)


# Pre-configured loggers for common components
class Loggers:
    """Pre-configured logger instances for agentic CLI components."""

    @staticmethod
    def cli() -> structlog.stdlib.BoundLogger:
        """Logger for CLI components."""
        return get_logger("agentic_cli.cli")

    @staticmethod
    def workflow() -> structlog.stdlib.BoundLogger:
        """Logger for workflow/agent components."""
        return get_logger("agentic_cli.workflow")

    @staticmethod
    def persistence() -> structlog.stdlib.BoundLogger:
        """Logger for persistence layer."""
        return get_logger("agentic_cli.persistence")

    @staticmethod
    def config() -> structlog.stdlib.BoundLogger:
        """Logger for configuration."""
        return get_logger("agentic_cli.config")

    @staticmethod
    def tools() -> structlog.stdlib.BoundLogger:
        """Logger for tools."""
        return get_logger("agentic_cli.tools")

    @staticmethod
    def knowledge_base() -> structlog.stdlib.BoundLogger:
        """Logger for knowledge base."""
        return get_logger("agentic_cli.knowledge_base")
