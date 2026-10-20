"""The raw LLM traffic log (``raw_llm_logging``), shared by every backend.

Each backend turns its model calls into ``LLM_REQUEST``/``LLM_RESPONSE``/
``LLM_USAGE`` events and writes them here, one JSON record per line, to
``~/.{app_name}/logs/llm_events.jsonl``. The log holds whole conversations, so
it lives under the user's home, never the (often git-tracked) project, with an
owner-only directory (0700) and file (0600), and obvious keys and tokens are
masked before a line is written. Logging is for debugging: a log that cannot
be written is skipped, so it never fails the caller.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from agentic_cli.logging import Loggers
from agentic_cli.workflow.events import WorkflowEvent

logger = Loggers.workflow()

# Best-effort redaction of provider secrets before they hit the log file.
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{16,}"      # Anthropic / OpenAI keys
    r"|AIza[A-Za-z0-9_\-]{16,}"     # Google API keys
    r"|Bearer\s+[A-Za-z0-9._\-]{10,})",  # bearer tokens
    re.IGNORECASE,
)


def redact_secrets(text: str) -> str:
    """Mask obvious API-key/token patterns in a serialized log line."""
    return _SECRET_RE.sub("[REDACTED]", text)


class LLMLogWriter:
    """Appends LLM events to ``~/.{app_name}/logs/llm_events.jsonl``.

    The directory and file are created (or tightened) owner-only when the
    writer is made. A directory that cannot be created turns the writer off,
    and a write that fails is skipped: nothing is logged, and nothing is
    raised.

    Attributes:
        path: The log file.
    """

    def __init__(self, app_name: str) -> None:
        log_dir = Path.home() / f".{app_name}" / "logs"
        self.path: Path = log_dir / "llm_events.jsonl"
        # Off for good when the directory cannot be made: a file created later
        # by ``open(..., "a")`` would not be owner-only.
        self._enabled = True
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._enabled = False
            logger.debug("llm_log_dir_unavailable", path=str(log_dir), exc_info=True)
            return
        try:
            log_dir.chmod(0o700)
        except OSError:
            pass
        # Create (or tighten) with owner-only perms before anything is written.
        try:
            self.path.touch(exist_ok=True)
            self.path.chmod(0o600)
        except OSError:
            pass

    def write(self, event: WorkflowEvent) -> None:
        """Append ``event`` as one redacted JSON line."""
        if not self._enabled:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event.type.value,
            "content": event.content,
            "metadata": event.metadata,
        }

        try:
            line = redact_secrets(json.dumps(record, default=str))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
