"""The raw LLM traffic log every backend writes through.

``workflow/llm_log.py`` holds the JSONL writer and the secret redaction that
ADK's ``LLMLoggingPlugin`` used to keep to itself. The log holds whole
conversations, so it lives under the user's home with owner-only permissions,
and a log that cannot be written never fails the caller.
"""

from __future__ import annotations

import json
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.llm_log import LLMLogWriter, redact_secrets

ANTHROPIC_KEY = "sk-ant-api03-ABCDEFGH1234567890ZYXW"
GOOGLE_KEY = "AIzaSyABCDEFGH1234567890ZYXW"
BEARER = "Bearer abcdefghij.KLMNOP-qrstu"


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _records(writer: LLMLogWriter) -> list[dict]:
    return [json.loads(line) for line in writer.path.read_text().splitlines()]


class TestLocation:
    def test_the_log_lives_under_the_home_directory(self, home):
        writer = LLMLogWriter("test_app")

        assert writer.path == home / ".test_app" / "logs" / "llm_events.jsonl"

    def test_the_directory_and_file_are_owner_only(self, home):
        writer = LLMLogWriter("test_app")

        assert _mode(writer.path.parent) == 0o700
        assert _mode(writer.path) == 0o600

    def test_an_existing_log_is_tightened(self, home):
        log_dir = home / ".test_app" / "logs"
        log_dir.mkdir(parents=True)
        log_dir.chmod(0o755)
        (log_dir / "llm_events.jsonl").write_text("")
        (log_dir / "llm_events.jsonl").chmod(0o644)

        writer = LLMLogWriter("test_app")

        assert _mode(log_dir) == 0o700
        assert _mode(writer.path) == 0o600


class TestRecords:
    def test_a_record_holds_the_timestamp_type_content_and_metadata(self, home):
        writer = LLMLogWriter("test_app")
        event = WorkflowEvent.llm_request(model="google:gemini-2.5-flash", tools=["glob"])

        writer.write(event)

        [record] = _records(writer)
        assert set(record) == {"timestamp", "type", "content", "metadata"}
        assert record["type"] == "llm_request"
        assert record["content"] == event.content
        assert record["metadata"] == event.metadata
        stamp = datetime.fromisoformat(record["timestamp"])
        assert stamp.tzinfo is not None and stamp.utcoffset() == timezone.utc.utcoffset(None)

    def test_each_event_appends_one_line(self, home):
        writer = LLMLogWriter("test_app")

        writer.write(WorkflowEvent.llm_request(model="m"))
        writer.write(WorkflowEvent.llm_response(model="m", content="hello"))

        assert [r["type"] for r in _records(writer)] == ["llm_request", "llm_response"]

    def test_values_json_cannot_hold_are_written_as_text(self, home):
        writer = LLMLogWriter("test_app")

        writer.write(WorkflowEvent.llm_request(model="m", config={"path": Path("/x/y")}))

        [record] = _records(writer)
        assert record["metadata"]["config"] == {"path": "/x/y"}

    @pytest.mark.parametrize("secret", [ANTHROPIC_KEY, GOOGLE_KEY, BEARER])
    def test_secrets_are_redacted_in_the_file(self, home, secret):
        writer = LLMLogWriter("test_app")

        writer.write(WorkflowEvent.llm_response(model="m", content=f"the key is {secret}"))

        text = writer.path.read_text()
        assert secret not in text
        assert "[REDACTED]" in text


class TestRedaction:
    @pytest.mark.parametrize(
        "line, kept",
        [
            (f"x-api-key: {ANTHROPIC_KEY}", "x-api-key: [REDACTED]"),
            (f"key={GOOGLE_KEY}&q=1", "key=[REDACTED]&q=1"),
            (f"authorization: {BEARER}", "authorization: [REDACTED]"),
            ("authorization: bearer abcdefghijkl", "authorization: [REDACTED]"),
        ],
    )
    def test_keys_and_tokens_are_masked(self, line, kept):
        assert redact_secrets(line) == kept

    @pytest.mark.parametrize("line", ["sk-short", "AIza-short", "Bearer short", "a skill"])
    def test_short_lookalikes_are_left_alone(self, line):
        assert redact_secrets(line) == line


class TestFailuresAreIgnored:
    def test_a_log_directory_that_cannot_be_made_is_ignored(self, tmp_path, monkeypatch):
        not_a_dir = tmp_path / "home"
        not_a_dir.write_text("a file where the home directory should be")
        monkeypatch.setenv("HOME", str(not_a_dir))

        writer = LLMLogWriter("test_app")
        writer.write(WorkflowEvent.llm_request(model="m"))

        assert not writer.path.exists()

    def test_a_writer_without_its_directory_stays_off(self, tmp_path, monkeypatch):
        """A file it created later would not be owner-only, so it never does."""
        home = tmp_path / "home"
        home.write_text("not a directory yet")
        monkeypatch.setenv("HOME", str(home))
        writer = LLMLogWriter("test_app")
        home.unlink()
        writer.path.parent.mkdir(parents=True)

        writer.write(WorkflowEvent.llm_request(model="m"))

        assert not writer.path.exists()

    def test_a_failed_write_is_ignored(self, home):
        writer = LLMLogWriter("test_app")
        writer.path.unlink()
        writer.path.mkdir()  # opening it for append now raises IsADirectoryError

        writer.write(WorkflowEvent.llm_request(model="m"))

        assert writer.path.is_dir()


class TestAdkPlugin:
    """ADK's ``LLMLoggingPlugin`` writes through the shared writer."""

    @pytest.fixture(autouse=True)
    def _adk(self):
        pytest.importorskip("google.adk")

    def test_the_plugin_logs_to_the_writers_file(self, home):
        from agentic_cli.workflow.adk.plugins import LLMLoggingPlugin

        plugin = LLMLoggingPlugin(app_name="test_app")

        assert plugin._log_file == LLMLogWriter("test_app").path

    def test_the_plugin_writes_through_the_writer(self, home, monkeypatch):
        from agentic_cli.workflow.adk.plugins import LLMLoggingPlugin

        written: list[WorkflowEvent] = []
        monkeypatch.setattr(LLMLogWriter, "write", lambda self, event: written.append(event))
        plugin = LLMLoggingPlugin(app_name="test_app")
        event = WorkflowEvent.llm_request(model="m")

        plugin._write_to_log(event)

        assert written == [event]

    def test_the_plugin_redacts_with_the_shared_function(self):
        from agentic_cli.workflow.adk import plugins

        assert plugins._redact_secrets is redact_secrets
