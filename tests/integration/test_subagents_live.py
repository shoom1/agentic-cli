"""LIVE: sub-agent delegation through the framework's ``task`` tool, on every
provider and both backends (real API calls).

Marked ``llm``; each case skips without its provider's key (the integration
``conftest`` loads keys from ``~/.research_demo/.env`` or
``$AGENTIC_TEST_ENV_FILE``). Run explicitly:

    python -m pytest tests/integration/test_subagents_live.py -v -m llm

Each case runs a coordinator and one sub-agent type on one provider's model,
on ADK and on Deep Agents, and checks what the offline contract suite
(``tests/workflow/test_subagents_contract.py``) cannot: that the real API
takes what the framework sends.

- Delegation: the provider accepts ``task``'s declaration (the
  ``agent_type`` enum, the brief's fields and their descriptions; on ADK a
  ``parameters_json_schema``), the run ends ``completed``, and the
  coordinator reads the envelope's ``success``.
- The step limit: a type with ``AgentLimits(max_model_calls=2)`` is given a
  task that needs more calls. Its last allowed call keeps its tools declared
  and ends with ``STEP_LIMIT_NOTE`` (read back from the raw LLM log): the
  provider accepts that request, the run ends ``completed``, partial,
  ``step limit reached``, and the coordinator reads that.
- The token limit: a type with a small ``AgentLimits(max_tokens=...)`` ends
  its run by the limit, or completes it when the answer that crosses it
  makes no tool calls.

Each test runs with HOME and the working directory in its own temporary
folder, so the raw LLM log (``raw_llm_logging``) is written there, never
under the user's home. The catalog lists each provider for real, as on a
first start, and caches nothing.

The coordinator is told to make one task call; a model may make more (Qwen
on OpenRouter splits the lookup by key, and re-delegates after a cancelled
run until ``max_runs_per_turn`` stops it). The checks read the first run.

Override a model with LIVE_SUBAGENTS_GEMINI_MODEL, LIVE_SUBAGENTS_CLAUDE_MODEL,
LIVE_SUBAGENTS_GPT_MODEL or LIVE_SUBAGENTS_OPENROUTER_MODEL (a
``provider:model`` reference).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.agents import RUN_CANCELLED, RUN_COMPLETED, AgentLimits  # noqa: E402
from agentic_cli.workflow.agents.control import STEP_LIMIT_NOTE  # noqa: E402
from agentic_cli.workflow.agents.envelope import (  # noqa: E402
    STEP_LIMIT_ERROR,
    TOKEN_LIMIT_ERROR,
    read_envelope,
)
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from tests.conftest import patch_catalogs  # noqa: E402
from tests.integration.helpers import (  # noqa: E402
    assert_no_errors,
    find_events,
    find_tool_calls,
    find_tool_results,
)
from tests.models.fakes import openrouter_spec  # noqa: E402

pytestmark = pytest.mark.llm

APP = "subagents-live"
USER = "u"
SESSION = "s"
#: The most seconds one turn may take: a hung call fails the test, not the run.
TURN_TIMEOUT = 300


def _provider(case_id: str, model: str, needs: str):
    """One provider: its model, and the variable its key is in."""
    return pytest.param(
        model,
        id=case_id,
        marks=pytest.mark.skipif(not os.environ.get(needs), reason=f"{needs} is not set"),
    )


PROVIDERS = [
    _provider(
        "gemini",
        os.getenv("LIVE_SUBAGENTS_GEMINI_MODEL", "google:gemini-2.5-flash"),
        "GOOGLE_API_KEY",
    ),
    _provider(
        "claude",
        os.getenv("LIVE_SUBAGENTS_CLAUDE_MODEL", "anthropic:claude-haiku-4-5"),
        "ANTHROPIC_API_KEY",
    ),
    _provider(
        "gpt",
        os.getenv("LIVE_SUBAGENTS_GPT_MODEL", "openai:gpt-5-mini"),
        "OPENAI_API_KEY",
    ),
    _provider(
        "openrouter",
        os.getenv("LIVE_SUBAGENTS_OPENROUTER_MODEL", "openrouter:qwen/qwen3.7-plus"),
        "OPENROUTER_API_KEY",
    ),
]
BACKENDS = ["adk", "deepagents"]

#: The keys each ``live_subagents_lookup`` call was asked for.
LOOKED_UP: list[str] = []


@register_tool(capabilities=EXEMPT, description="Look up the value filed under one key.")
def live_subagents_lookup(key: str) -> dict:
    """Look up the value filed under ``key`` (one key per call)."""
    LOOKED_UP.append(key)
    return {"success": True, "key": key, "value": f"{key.strip().lower()}-{len(key.strip()) * 7}"}


@pytest.fixture
def isolated(tmp_path, monkeypatch) -> Path:
    """HOME and the working directory in the test's folder: the raw LLM log
    is written under it. Real model listings, written to no cache file."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    patch_catalogs(monkeypatch)
    LOOKED_UP.clear()
    return home


def _manager(backend: str, model: str, configs: list[AgentConfig], workspace: Path):
    settings = BaseSettings(
        app_name=APP,
        workspace_dir=workspace,
        session_store="memory",
        thinking_effort="low",
        raw_llm_logging=True,
        model_providers={"openrouter": openrouter_spec()},
    )
    manager_class = GoogleADKWorkflowManager if backend == "adk" else DeepAgentsWorkflowManager
    return manager_class(agent_configs=configs, settings=settings, model=model)


async def _turn(manager, message: str) -> list:
    async def collect() -> list:
        return [event async for event in manager.process(message, USER, session_id=SESSION)]

    return await asyncio.wait_for(collect(), TURN_TIMEOUT)


def _root_text(events: list) -> str:
    """The coordinator's own reply: the TEXT of no sub-agent run."""
    return "".join(
        event.content or ""
        for event in find_events(events, EventType.TEXT)
        if not event.metadata.get("subagent") and (event.run is None or event.run.run is None)
    )


def _envelopes(events: list) -> list[dict]:
    """The envelope of each ``task`` result the coordinator read, in order."""
    found = [read_envelope(event.metadata.get("result")) for event in find_tool_results(events, "task")]
    assert all(envelope is not None for envelope in found), found
    return found


def _records(manager) -> list:
    return manager.agent_runs.records(manager.session_ref(SESSION, USER))


def _calls_of(manager, run: str) -> int:
    """How many model calls the run ``run`` made, as the usage ledger counts them."""
    return manager.usage_of(manager.session_ref(SESSION, USER)).totals(run=run).calls


def _sent_requests(home: Path, run: str | None, agent: str | None = None) -> list[dict]:
    """The model requests of the run ``run`` (None: the root's own, ``agent``'s),
    as the raw LLM log recorded them sent, in order."""
    path = home / f".{APP}" / "logs" / "llm_events.jsonl"
    assert path.is_file(), "no raw LLM log was written"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [
        record for record in records
        if record.get("type") == EventType.LLM_REQUEST.value
        and (record.get("run") or {}).get("run") == run
        and (agent is None or record.get("metadata", {}).get("agent") == agent)
    ]


def _parent_was_sent(home: Path, text: str) -> bool:
    """Whether the coordinator's last request held ``text``: the envelope it read."""
    sent = _sent_requests(home, None, "coordinator")
    return bool(sent) and text in json.dumps(sent[-1]["metadata"].get("messages"))


# -- 1. delegation through task ------------------------------------------------------


@pytest.mark.parametrize("model", PROVIDERS)
@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_delegation_through_task_completes_and_its_envelope_is_read(
    backend, model, isolated, tmp_path
):
    manager = _manager(backend, model, [
        AgentConfig(
            name="coordinator",
            prompt=(
                "You never calculate yourself: hand every calculation to the calculator "
                "sub-agent with one task call. When the task call returns, reply with one "
                "line: the product, then the word SUCCESS if the task result's success "
                "field is true, else the word FAILURE."
            ),
            sub_agents=["calculator"],
            include_state_tools=False,
        ),
        AgentConfig(
            name="calculator",
            prompt="You multiply numbers. Reply with the product only.",
            description="Multiplies numbers and replies with the product.",
            include_state_tools=False,
        ),
    ], tmp_path / "workspace")
    try:
        events = await _turn(manager, "What is 6 times 7?")
        records = _records(manager)
    finally:
        await manager.cleanup()

    assert_no_errors(events)
    delegations = find_tool_calls(events, "task")
    assert delegations, "the coordinator did not call task"
    arguments = delegations[0].metadata["tool_args"]
    assert arguments["agent_type"] == "calculator", arguments
    assert arguments.get("title") and arguments.get("objective"), arguments
    (envelope, *_) = _envelopes(events)
    assert envelope["success"] is True, envelope
    assert (envelope["status"], envelope["partial"], envelope["error"]) == (
        RUN_COMPLETED, False, None,
    ), envelope
    assert "42" in envelope["result"], envelope
    assert records and records[0].status == RUN_COMPLETED, records
    reply = _root_text(events)
    assert "42" in reply and "SUCCESS" in reply.upper(), reply
    assert "FAILURE" not in reply.upper(), reply


# -- 2. the step limit's last call ---------------------------------------------------

_KEYS = ("alpha", "beta", "gamma", "delta")

_RESEARCHER_PROMPT = (
    "You look values up. Look up every key the task names with live_subagents_lookup, "
    "one key per response: never call the tool more than once in a response, and look "
    "up the next key only after the previous result came back. When you have every "
    "value, reply with each key and its value."
)

_COORDINATOR_PROMPT = (
    "You never look anything up yourself: hand the whole lookup to the researcher "
    "sub-agent in exactly one task call whose brief names every key (never one call per "
    "key), and do not call task again whatever it returns. "
    "When the task call returns, reply with one line copying three fields of the task "
    "result exactly: status=<its status field> partial=<its partial field> "
    "error=<its error field>."
)


def _limited(limits: AgentLimits) -> list[AgentConfig]:
    """A coordinator and a researcher with ``limits`` that looks keys up."""
    return [
        AgentConfig(
            name="coordinator",
            prompt=_COORDINATOR_PROMPT,
            sub_agents=["researcher"],
            include_state_tools=False,
        ),
        AgentConfig(
            name="researcher",
            prompt=_RESEARCHER_PROMPT,
            description="Looks up the values filed under keys, one key per call.",
            tools=[live_subagents_lookup],
            include_state_tools=False,
            limits=limits,
        ),
    ]


_ASK = f"Find the values filed under the keys {', '.join(_KEYS)}."


@pytest.mark.parametrize("model", PROVIDERS)
@pytest.mark.parametrize("backend", BACKENDS)
async def test_the_step_limits_last_call_is_accepted_and_the_run_completes_partial(
    backend, model, isolated, tmp_path
):
    manager = _manager(backend, model, _limited(AgentLimits(max_model_calls=2)),
                       tmp_path / "workspace")
    try:
        events = await _turn(manager, _ASK)
        records = _records(manager)
        (envelope, *_) = _envelopes(events)
        calls = _calls_of(manager, envelope["run"])
    finally:
        await manager.cleanup()

    assert_no_errors(events)
    assert (envelope["success"], envelope["status"], envelope["partial"], envelope["error"]) == (
        True, RUN_COMPLETED, True, STEP_LIMIT_ERROR,
    ), envelope
    assert calls == 2, f"the run made {calls} model calls"
    assert LOOKED_UP, "the researcher looked nothing up"
    assert records[0].status == RUN_COMPLETED and records[0].error == STEP_LIMIT_ERROR, records

    # What the provider accepted: the last allowed request, its tools still
    # declared and its last message ending with the note.
    sent = _sent_requests(isolated, envelope["run"])
    assert len(sent) == 2, [record["metadata"].get("tools") for record in sent]
    first, last = (record["metadata"] for record in sent)
    assert STEP_LIMIT_NOTE not in json.dumps(first.get("messages"))
    assert STEP_LIMIT_NOTE in json.dumps(last["messages"][-1]), last["messages"][-1]
    assert "live_subagents_lookup" in (last.get("tools") or []), last.get("tools")

    # The parent reads it: its next request holds the envelope, the step
    # limit's error in it, and its reply says how the run ended. (What it
    # copies of the error is the model's: Gemini 2.5 Flash once wrote
    # ``error=None`` for the error it was sent.)
    assert _parent_was_sent(isolated, STEP_LIMIT_ERROR)
    reply = _root_text(events).lower()
    assert "completed" in reply, reply


# -- 3. the token limit --------------------------------------------------------------

#: Under any first request's input: the researcher's prompt and tool alone are more.
_SMALL_TOKEN_LIMIT = 100


@pytest.mark.parametrize("model", PROVIDERS)
@pytest.mark.parametrize("backend", BACKENDS)
async def test_the_token_limit_ends_the_run_or_its_last_answer_completes_it(
    backend, model, isolated, tmp_path
):
    manager = _manager(backend, model, _limited(AgentLimits(max_tokens=_SMALL_TOKEN_LIMIT)),
                       tmp_path / "workspace")
    try:
        events = await _turn(manager, _ASK)
        records = _records(manager)
        (envelope, *_) = _envelopes(events)
        calls = _calls_of(manager, envelope["run"])
    finally:
        await manager.cleanup()

    assert_no_errors(events)
    # The first answer is past the limit: it ends the run either way.
    assert calls == 1, f"the run made {calls} model calls"
    assert LOOKED_UP == [], f"a call past the limit ran: {LOOKED_UP}"
    if envelope["status"] == RUN_CANCELLED:
        assert (envelope["success"], envelope["error"], envelope["cancelled_by"]) == (
            False, TOKEN_LIMIT_ERROR, "limit",
        ), envelope
        assert records[0].cancelled_by == "limit", records
        assert _parent_was_sent(isolated, TOKEN_LIMIT_ERROR)
    else:
        # An answer that made no tool calls: the run's result, whole.
        assert (envelope["success"], envelope["status"], envelope["partial"]) == (
            True, RUN_COMPLETED, False,
        ), envelope
        assert envelope["error"] is None, envelope
    reply = _root_text(events).lower()
    assert envelope["status"] in reply, reply
