"""Tests for ``agentic_cli.logging``: third-party noise reduction, and where
log lines go (stderr, or a sink while a CLI's prompt runs)."""

import logging
import threading

import structlog

from agentic_cli.logging import configure_logging, log_lines_to


def test_trafilatura_dependency_loggers_are_silenced():
    """trafilatura's own stdlib loggers emit raw ERROR lines to stderr
    (e.g. "parsed tree length: 0, wrong data type or not valid HTML",
    "readability_lxml failed:") under the default logging setup.
    The converter (agentic_cli.memory.kb.convert) logs kb_convert_html (and
    kb_convert_trafilatura_failed / kb_convert_html_failed), and the ingest
    tools log kb_convert_failed, which already record what happened, so these
    must not print over the CLI.
    """
    configure_logging()

    for name in ("trafilatura", "htmldate", "courlan", "justext"):
        assert logging.getLogger(name).level == logging.CRITICAL


def test_while_a_sink_is_set_each_log_line_reaches_it_whole(logging_configured):
    """One call a line, without its newline, whichever thread logged it."""
    lines: list[str] = []
    with log_lines_to(lines.append):
        structlog.get_logger("t").warning("first_event", key="value")
        thread = threading.Thread(target=structlog.get_logger("t").warning, args=("thread_event",))
        thread.start()
        thread.join()

    assert len(lines) == 2
    assert "first_event" in lines[0] and "key" in lines[0]
    assert "thread_event" in lines[1]
    assert not any("\n" in line for line in lines)


def test_without_a_sink_log_lines_go_to_stderr(logging_configured, capsys):
    lines: list[str] = []
    with log_lines_to(lines.append):
        pass

    structlog.get_logger("t").warning("after_the_sink")

    assert lines == []
    assert "after_the_sink" in capsys.readouterr().err
