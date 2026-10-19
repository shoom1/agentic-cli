"""Write ``adk1_sessions.sql``: a session database as ADK 1.x leaves it.

0.6.x pinned ``google-adk<2``, so every durable session a 0.6.x install saved
was written by ADK 1.x. ``tests/workflow/test_adk1_session_compat.py`` loads
this dump and reads it back with whatever ADK is installed.

The dump is committed; this script is the record of how it was made. It was
run once, with ADK 1.36 installed, from the repository root:

    conda run -n agenticcli python tests/fixtures/make_adk1_sessions.py

Run on ADK 2 it would write a database ADK 2 made, which proves nothing, so it
refuses. An optional argument writes the dump somewhere else.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

import google.adk
from google.adk.events import Event
from google.adk.sessions import DatabaseSessionService
from google.genai import types

APP_NAME = "research_demo"
USER_ID = "tester"
SESSION_ID = "adk1-session"
STATE = {"plan": "1. Search arXiv", "user:lang": "en"}

OUT = Path(__file__).with_name("adk1_sessions.sql")


def _text(author: str, text: str) -> Event:
    role = "user" if author == "user" else "model"
    return Event(
        author=author,
        invocation_id="inv-1",
        content=types.Content(role=role, parts=[types.Part(text=text)]),
    )


def _call(call_id: str, name: str, args: dict, *, long_running: bool = False) -> Event:
    return Event(
        author="coordinator",
        invocation_id="inv-1",
        long_running_tool_ids={call_id} if long_running else None,
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))],
        ),
    )


def _response(call_id: str, name: str, response: dict) -> Event:
    return Event(
        author="coordinator",
        invocation_id="inv-1",
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call_id, name=name, response=response
                    )
                )
            ],
        ),
    )


async def _write(db: Path) -> None:
    service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db}")
    try:
        session = await service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID, state=dict(STATE)
        )
        for event in (
            _text("user", "Search arXiv for speculative decoding."),
            _call("call-1", "search_arxiv", {"query": "speculative decoding"}),
            _response("call-1", "search_arxiv", {"success": True, "count": 2}),
            _call("call-2", "sandbox_execute", {"code": "1 + 1"}, long_running=True),
            _text("coordinator", "Found 2 papers; the analysis is running."),
        ):
            await service.append_event(session, event)
    finally:
        await service.close()


def main() -> None:
    if not google.adk.__version__.startswith("1."):
        sys.exit(f"needs ADK 1.x to write a 0.6.x database; found {google.adk.__version__}")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "sessions.db"
        asyncio.run(_write(db))
        with sqlite3.connect(db) as conn:
            dump = "\n".join(conn.iterdump())
    dump = re.sub(r"[ \t]+$", "", dump, flags=re.MULTILINE)  # SQLite pads DDL lines
    out.write_text(
        f"-- A session database written by google-adk {google.adk.__version__}.\n"
        f"-- Made by tests/fixtures/make_adk1_sessions.py; do not edit by hand.\n"
        f"{dump}\n"
    )
    print(f"wrote {out} (google-adk {google.adk.__version__})")


if __name__ == "__main__":
    main()
