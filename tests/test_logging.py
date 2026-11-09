"""Tests for ``agentic_cli.logging``: third-party noise reduction, and where
log lines go (stderr, or a sink while a CLI's prompt runs)."""

import logging
import subprocess
import sys
import threading

import structlog

from agentic_cli.logging import LOG_STREAM, log_lines_to


def test_trafilatura_dependency_loggers_are_silenced(logging_configured):
    """trafilatura's own stdlib loggers emit raw ERROR lines to stderr
    (e.g. "parsed tree length: 0, wrong data type or not valid HTML",
    "readability_lxml failed:") under the default logging setup.
    The converter (agentic_cli.memory.kb.convert) logs kb_convert_html (and
    kb_convert_trafilatura_failed / kb_convert_html_failed), and the ingest
    tools log kb_convert_failed, which already record what happened, so these
    must not print over the CLI.
    """
    for name in ("trafilatura", "htmldate", "courlan", "justext"):
        assert logging.getLogger(name).level == logging.CRITICAL


def test_the_genai_sdks_non_text_parts_warning_is_silenced(logging_configured):
    """google-genai warns when a response holding a function call is read as
    text; the ADK event processor reads every part on its own."""
    assert logging.getLogger("google_genai.types").level == logging.ERROR


def test_importing_the_adk_backend_sets_no_logger_level():
    """A library import changes no process-wide logging: ``configure_logging``
    does, which the CLI calls. Run in a fresh interpreter, since this one has
    imported everything already."""
    probe = (
        "import logging, agentic_cli.workflow.adk.manager; "
        "print(logging.getLogger('google_genai.types').level)"
    )
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    assert run.stdout.strip() == str(logging.NOTSET)


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


def test_a_line_begun_under_a_sink_and_ended_without_one_is_written_whole(capsys):
    """print() writes a line and its newline apart: a sink removed between
    the two must not leave the newline alone on stderr."""
    lines: list[str] = []
    with log_lines_to(lines.append):
        LOG_STREAM.write("unfinished")
    LOG_STREAM.write("\n")

    assert lines == []
    assert capsys.readouterr().err == "unfinished\n"


def test_without_a_stderr_a_log_line_is_dropped_not_raised(logging_configured, monkeypatch):
    """A process with no stderr (pythonw, a daemon) still logs without failing."""
    monkeypatch.setattr("sys.stderr", None)

    structlog.get_logger("t").warning("nowhere_to_go")
    LOG_STREAM.flush()
