"""Loggers for the memory package. The host configures structlog."""

from __future__ import annotations

import structlog


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """A logger named ``name`` (by convention ``agentic_cli.memory.<component>``)."""
    return structlog.get_logger(name)
