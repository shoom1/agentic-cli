"""Tests for ``agentic_cli.logging``: third-party noise reduction."""

import logging

from agentic_cli.logging import configure_logging


def test_trafilatura_dependency_loggers_are_silenced():
    """trafilatura's own stdlib loggers emit raw ERROR lines to stderr
    (e.g. "parsed tree length: 0, wrong data type or not valid HTML",
    "readability_lxml failed:") under the default logging setup.
    kb_convert's own ``kb_convert_html``/``kb_convert_failed`` events already
    record what happened, so these must not print over the CLI.
    """
    configure_logging()

    for name in ("trafilatura", "htmldate", "courlan", "justext"):
        assert logging.getLogger(name).level == logging.CRITICAL
