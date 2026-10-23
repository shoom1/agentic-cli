"""LIVE: context management against real providers.

Marked ``llm``; each case skips without the keys it needs (the integration
``conftest`` loads keys from ``~/.research_demo/.env`` or
``$AGENTIC_TEST_ENV_FILE``). Run explicitly:

    python -m pytest tests/integration/test_context_live.py -v -m llm

What only a real provider can answer:

- Deep Agents: Claude, Gemini and OpenAI each accept a summarized thread
  (``context_max_tokens`` at its 16,000 minimum makes it summarize within a
  few turns).
- A small-window OpenRouter model driven past its window: the conversation is
  summarized and goes on.
- One message larger than the window (the small OpenRouter model, and Claude
  Haiku's 200k) ends in the clear overflow message, whether Deep Agents' own
  budget check or the provider's overflow text stops it.
- A ``read_file`` result over ``tool_result_max_chars``, on Claude through ADK
  and through Deep Agents: the model reads the note and pages the output with
  ``read_file`` to find a line the cut left out. And a tool's result that no
  file holds: the model follows the note to the saved output.
- Sub-agents (ADK on Claude; Deep Agents on Claude and on OpenAI Responses):
  one that inherits sees the conversation, one that starts fresh sees only its
  brief, and each is delegated to and returns. A sub-agent that inherits a
  history holding tool calls for tools it lacks is still accepted.
- ADK summarizes the main agent: Claude, Gemini, OpenAI Responses and
  OpenRouter accept a conversation that starts with the user-role summary
  and goes on, and models that always think (Claude Opus 4.7, Gemini 3)
  accept the summary call with thinking off.
- A provider's real overflow on ADK (Claude Haiku), provoked with
  dense CJK text the character estimate counts low: the main agent's ends
  in the clear overflow message, and a sub-agent's in one warning while the
  coordinator answers.
- The context display: with a cache hit (Claude on both backends, Gemini on
  ADK, OpenAI), the prompt count the status bar shows is the whole prompt;
  and with the raw LLM log on, an ADK delegation keeps the main agent's
  ``ctx`` while the log names each call's own model.
- ``/compact``: on both backends, against Claude, Gemini and OpenAI, a summary
  is written now and the next request is accepted; a focus with braces and
  quotes reaches the prompt; and on Deep Agents a ``/compact`` near the
  trigger is not followed by a second summary on the next call.

Override the models with LIVE_CONTEXT_CLAUDE_MODEL (default
``anthropic:claude-sonnet-4-5``), LIVE_CONTEXT_GEMINI_MODEL (default
``google:gemini-2.5-flash``), LIVE_CONTEXT_OPENAI_MODEL (default
``openai:gpt-5-mini``), LIVE_CONTEXT_HAIKU_MODEL (default
``anthropic:claude-haiku-4-5``) and LIVE_CONTEXT_SMALL_MODEL (an ``openrouter:`` id;
by default the cheapest tool-calling model OpenRouter lists with a 32k to 64k
window).
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import httpx
import pytest

pytest.importorskip("deepagents")

from agentic_cli.cli.usage_tracker import UsageTracker  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.models import ModelCatalog  # noqa: E402
from agentic_cli.workflow import model_services  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import OVERFLOW_ERROR_CODE  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from tests.integration.conftest import collect_events  # noqa: E402
from tests.workflow.compacting import compact_now  # noqa: E402
from tests.integration.helpers import (  # noqa: E402
    assert_no_errors,
    find_events,
    find_tool_calls,
    find_tool_results,
)

pytestmark = pytest.mark.llm

CLAUDE = os.getenv("LIVE_CONTEXT_CLAUDE_MODEL", "anthropic:claude-sonnet-4-5")
GEMINI = os.getenv("LIVE_CONTEXT_GEMINI_MODEL", "google:gemini-2.5-flash")
OPENAI = os.getenv("LIVE_CONTEXT_OPENAI_MODEL", "openai:gpt-5-mini")
HAIKU = os.getenv("LIVE_CONTEXT_HAIKU_MODEL", "anthropic:claude-haiku-4-5")
OPUS = os.getenv("LIVE_CONTEXT_OPUS_MODEL", "anthropic:claude-opus-4-7")
GEMINI_3 = os.getenv("LIVE_CONTEXT_GEMINI_3_MODEL", "google:gemini-3-flash-preview")
OPENROUTER = os.getenv("LIVE_CONTEXT_OPENROUTER_MODEL", "openrouter:qwen/qwen3-235b-a22b")

_KEYS = {"google": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
         "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def _key_for(model: str) -> str:
    return _KEYS[model.partition(":")[0]]


def _needs(*variables: str):
    missing = [v for v in variables if not os.environ.get(v)]
    return pytest.mark.skipif(bool(missing), reason=f"{', '.join(missing)} not set")


@pytest.fixture
def listed(monkeypatch):
    """Real model listings, as on a first start, written to no cache file."""

    def build_catalog(settings, endpoints, *, transport=None):
        return ModelCatalog(endpoints=endpoints)

    monkeypatch.setattr(model_services, "build_catalog", build_catalog)


def _settings(tmp_path, **values) -> BaseSettings:
    return BaseSettings(workspace_dir=tmp_path, session_store="memory", **values)


def _assistant(**values) -> list[AgentConfig]:
    return [AgentConfig(name="assistant", prompt="You are a concise assistant.", **values)]


_FLOWERS = ("tulips", "roses", "daisies", "lilies", "asters", "peonies", "irises")
_STREETS = ("Elm", "Oak", "Birch", "Maple", "Cedar", "Willow", "Aspen", "Linden")


def _filler(tokens: int, seed: int) -> str:
    """About ``tokens`` tokens of varied, harmless garden notes (about 4 characters a token).

    Plain prose: Claude ended calls given a dense table of shipments with
    ``stop_reason: "refusal"`` and no text, summary calls included.
    """
    lines, n = [], 0
    while sum(len(line) + 1 for line in lines) < tokens * 4:
        n += 1
        flower = _FLOWERS[(n + seed) % len(_FLOWERS)]
        street = _STREETS[(n * 3 + seed) % len(_STREETS)]
        lines.append(
            f"Note {seed}-{n}: the garden on {street} Street planted {(n * 7 + seed) % 90 + 10} "
            f"{flower} this spring, and the neighbours came to see them in week {n % 52 + 1}."
        )
    return "\n".join(lines)


def _text(events) -> str:
    return "".join(event.content or "" for event in find_events(events, EventType.TEXT))


def _summaries(events):
    return [e for e in find_events(events, EventType.CONTEXT_TRIMMED)
            if e.metadata.get("source") == "summarization"]


@pytest.mark.parametrize("model", [
    pytest.param(CLAUDE, marks=_needs(_key_for(CLAUDE)), id="claude"),
    pytest.param(GEMINI, marks=_needs(_key_for(GEMINI)), id="gemini"),
    pytest.param(OPENAI, marks=_needs(_key_for(OPENAI)), id="openai"),
])
async def test_deep_agents_summarizes_and_the_provider_accepts_the_summarized_thread(
    listed, tmp_path, model
):
    manager = DeepAgentsWorkflowManager(
        agent_configs=_assistant(),
        settings=_settings(tmp_path, context_max_tokens=16_000, thinking_effort="low"),
        model=model,
    )
    turns = []
    try:
        turns.append(await collect_events(
            manager,
            "The project we are discussing is named Quillfeather. Here are its garden "
            f"notes; reply OK.\n\n{_filler(6_000, 1)}",
            session_id="summarized",
        ))
        for seed in (2, 3):
            turns.append(await collect_events(
                manager, f"More notes; reply OK.\n\n{_filler(6_000, seed)}",
                session_id="summarized",
            ))
        turns.append(await collect_events(
            manager, "What is the project's name? Reply with the name only.",
            session_id="summarized",
        ))
    finally:
        await manager.cleanup()

    events = [event for turn in turns for event in turn]
    assert_no_errors(events)
    summaries = _summaries(events)
    assert summaries, "no summary was written"
    assert summaries[0].metadata["tokens_after"] < summaries[0].metadata["tokens_before"]
    types = [e.type for e in events]
    assert types.index(EventType.CONTEXT_SUMMARIZING) < types.index(EventType.CONTEXT_TRIMMED)
    assert any(e.metadata.get("purpose") == "summarization"
               for e in find_events(events, EventType.LLM_USAGE))
    # Every turn after the summary was answered by the provider.
    after = next(i for i, turn in enumerate(turns) if _summaries(turn))
    for turn in turns[after:]:
        assert _text(turn).strip(), "a turn after the summary got no answer"


def _small_openrouter_model() -> tuple[str, int]:
    """The cheapest tool-calling OpenRouter model with a 32k to 64k window."""
    if os.getenv("LIVE_CONTEXT_SMALL_MODEL"):
        return os.environ["LIVE_CONTEXT_SMALL_MODEL"], 0
    try:
        listing = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
    except (httpx.HTTPError, KeyError, ValueError) as error:
        pytest.skip(f"OpenRouter's model listing is unavailable: {error}")
    candidates = []
    for record in listing:
        window = record.get("context_length") or 0
        price = float((record.get("pricing") or {}).get("prompt") or 0)
        modality = (record.get("architecture") or {}).get("modality") or ""
        if (32_000 <= window <= 66_000 and price > 0 and modality.endswith("->text")
                and "tools" in (record.get("supported_parameters") or [])
                and not record["id"].endswith(":free")):
            candidates.append((price, window, record["id"]))
    if not candidates:
        pytest.skip("OpenRouter lists no tool-calling model with a 32k to 64k window")
    price, window, model_id = min(candidates)
    return f"openrouter:{model_id}", window


@_needs("OPENROUTER_API_KEY")
async def test_a_small_window_model_driven_past_its_window_is_summarized_and_goes_on(
    listed, tmp_path
):
    model, window = _small_openrouter_model()
    per_turn = (window or 32_000) // 3
    manager = DeepAgentsWorkflowManager(
        agent_configs=_assistant(include_state_tools=False),
        settings=_settings(tmp_path, thinking_effort="none"),
        model=model,
    )
    turns = []
    try:
        for seed in range(1, 6):
            turns.append(await collect_events(
                manager, f"Garden notes; reply OK.\n\n{_filler(per_turn, seed)}",
                session_id="small",
            ))
    finally:
        await manager.cleanup()

    events = [event for turn in turns for event in turn]
    assert_no_errors(events)
    assert _summaries(events), f"{model} was never summarized"
    assert _text(turns[-1]).strip(), f"{model} gave the last turn no answer"


@pytest.mark.parametrize("which", [
    pytest.param("small", marks=_needs("OPENROUTER_API_KEY"), id="openrouter-small"),
    pytest.param("haiku", marks=_needs(_key_for(HAIKU)), id="claude-haiku"),
])
async def test_one_message_larger_than_the_window_ends_in_the_clear_message(
    listed, tmp_path, which
):
    model, window = _small_openrouter_model() if which == "small" else (HAIKU, 200_000)
    manager = DeepAgentsWorkflowManager(
        agent_configs=_assistant(include_state_tools=False),
        settings=_settings(tmp_path, thinking_effort="none"),
        model=model,
    )
    try:
        events = await collect_events(
            manager,
            f"Summarize these notes.\n\n{_filler(int((window or 32_000) * 1.3), 9)}",
            session_id="overflow",
        )
    finally:
        await manager.cleanup()

    errors = find_events(events, EventType.ERROR)
    assert len(errors) == 1, [e.content for e in errors]
    (error,) = errors
    assert error.metadata.get("error_code") == OVERFLOW_ERROR_CODE, error.content
    assert error.content.startswith("This conversation no longer fits "), error.content
    assert error.content.endswith("context window. Use /compact or /new."), error.content


def _numbered_file(path, lines: int) -> dict[int, str]:
    """A file of ``lines`` numbered lines, each naming its own word; the words by line."""
    words = {n: f"w{(n * 7919) % 100_003:05d}" for n in range(lines)}
    path.write_text("".join(f"Line {n:04d}: the word for this line is {w}.\n"
                            for n, w in words.items()))
    return words


@pytest.mark.parametrize("backend", ["adk", "deepagents"])
@_needs(_key_for(CLAUDE))
async def test_the_model_pages_a_cut_read_file_result(listed, tmp_path, monkeypatch, backend):
    monkeypatch.chdir(tmp_path)
    # 1,500 lines of about 43 characters: about 65,000 characters, cut to
    # 20,000. Line 900 is in the middle the cut leaves out.
    words = _numbered_file(tmp_path / "notes.txt", 1_500)
    manager_class = GoogleADKWorkflowManager if backend == "adk" else DeepAgentsWorkflowManager
    manager = manager_class(
        agent_configs=_assistant(tools=["read_file"], include_state_tools=False),
        settings=_settings(tmp_path, tool_result_max_chars=20_000, thinking_effort="none"),
        model=CLAUDE,
    )
    try:
        events = await collect_events(
            manager,
            "Read notes.txt with read_file. Then tell me the word on the line that starts "
            "with 'Line 0900:'. Reply with the word only.",
            session_id="paging",
        )
    finally:
        await manager.cleanup()

    assert_no_errors(events)
    first, *later = find_tool_results(events, "read_file")
    assert "Cut: " in str(first.metadata.get("result")), first.metadata.get("result")
    assert later, "the model never read the output again"
    pages = [c.metadata["tool_args"] for c in find_tool_calls(events, "read_file")[1:]]
    assert any(args.get("offset") or args.get("limit") for args in pages), pages
    assert list((tmp_path / ".agentic_cli" / "tool-output").glob("read_file-*.txt"))
    assert words[900] in _text(events), (_text(events), pages)


#: ``long_report``'s text: what ``_numbered_file`` writes, held by no file.
_REPORT_LINES = 1_500


def _report_word(n: int) -> str:
    return f"w{(n * 7919) % 100_003:05d}"


@register_tool(capabilities=EXEMPT)
def long_report() -> dict:
    """Return today's long report, one numbered line per entry."""
    text = "".join(f"Line {n:04d}: the word for this line is {_report_word(n)}.\n"
                   for n in range(_REPORT_LINES))
    return {"success": True, "report": text}


@pytest.mark.parametrize("backend", ["adk", "deepagents"])
@_needs(_key_for(CLAUDE))
async def test_the_model_follows_the_note_to_the_saved_output(
    listed, tmp_path, monkeypatch, backend
):
    monkeypatch.chdir(tmp_path)
    manager_class = GoogleADKWorkflowManager if backend == "adk" else DeepAgentsWorkflowManager
    manager = manager_class(
        agent_configs=_assistant(tools=[long_report, "read_file"], include_state_tools=False),
        settings=_settings(tmp_path, tool_result_max_chars=20_000, thinking_effort="none"),
        model=CLAUDE,
    )
    try:
        events = await collect_events(
            manager,
            "Call long_report. Then tell me the word on the line that starts with "
            "'Line 0900:'. Reply with the word only.",
            session_id="saved",
        )
    finally:
        await manager.cleanup()

    assert_no_errors(events)
    (report,) = find_tool_results(events, "long_report")
    assert "tool-output" in str(report.metadata.get("result"))
    paths = [c.metadata["tool_args"].get("path", "") for c in find_tool_calls(events, "read_file")]
    assert any("tool-output" in path for path in paths), paths
    assert _report_word(900) in _text(events), (_text(events), paths)


#: A coordinator that passes questions on word for word, so only an inheriting
#: sub-agent can know what was said earlier.
_COORDINATOR = (
    "You coordinate two sub-agents, the archivist and the clerk. When the user asks you "
    "to ask one of them something, delegate the user's question to that sub-agent word "
    "for word, adding nothing, then tell the user its answer."
)
_HELPER = "Answer the question you are given in one short sentence. If you don't know, say so."


def _team(**extra) -> list[AgentConfig]:
    return [
        AgentConfig(name="coordinator", prompt=_COORDINATOR, sub_agents=["archivist", "clerk"],
                    include_state_tools=False, **extra),
        AgentConfig(name="archivist", prompt=_HELPER, context="inherit", include_state_tools=False,
                    description="Answers questions about the conversation so far."),
        AgentConfig(name="clerk", prompt=_HELPER, context="fresh", include_state_tools=False,
                    description="Answers general questions."),
    ]


def _answers_of(events, agent: str, backend: str) -> list[str]:
    """What ``agent`` answered its coordinator (the delegation call's results)."""
    tool = agent if backend == "adk" else "task"
    return [str(e.metadata.get("result")) for e in find_tool_results(events, tool)]


def _reply(events) -> str:
    """The main agent's own text: a sub-agent's text is activity, not the reply."""
    return "".join(e.content or "" for e in find_events(events, EventType.TEXT)
                   if not e.metadata.get("subagent"))


def _manager_for(backend: str, configs, settings, model):
    manager_class = GoogleADKWorkflowManager if backend == "adk" else DeepAgentsWorkflowManager
    return manager_class(agent_configs=configs, settings=settings, model=model)


@pytest.mark.parametrize("backend, model", [
    pytest.param("adk", CLAUDE, marks=_needs(_key_for(CLAUDE)), id="adk-claude"),
    pytest.param("deepagents", CLAUDE, marks=_needs(_key_for(CLAUDE)), id="deepagents-claude"),
    pytest.param("deepagents", OPENAI, marks=_needs(_key_for(OPENAI)), id="deepagents-openai"),
])
async def test_an_inheriting_sub_agent_sees_the_conversation_and_a_fresh_one_does_not(
    listed, tmp_path, backend, model
):
    manager = _manager_for(backend, _team(), _settings(tmp_path, thinking_effort="none"), model)
    try:
        told = await collect_events(
            manager, "The project we are discussing is named Quillfeather. Reply OK.",
            session_id="team",
        )
        inherited = await collect_events(
            manager, "Ask the archivist: what is the name of the project we are discussing?",
            session_id="team",
        )
        fresh = await collect_events(
            manager, "Ask the clerk: what is the name of the project we are discussing?",
            session_id="team",
        )
    finally:
        await manager.cleanup()

    assert_no_errors(told + inherited + fresh)
    (archivist,) = _answers_of(inherited, "archivist", backend)
    assert "quillfeather" in archivist.lower(), archivist
    assert "quillfeather" in _reply(inherited).lower()
    (clerk,) = _answers_of(fresh, "clerk", backend)
    assert "quillfeather" not in clerk.lower(), clerk


@register_tool(capabilities=EXEMPT)
def lookup_codename() -> dict:
    """Look up the project's codename."""
    return {"success": True, "codename": "Marigold-7"}


@pytest.mark.parametrize("backend", ["adk", "deepagents"])
@_needs(_key_for(CLAUDE))
async def test_an_inherited_history_with_tool_calls_the_sub_agent_lacks_is_accepted(
    listed, tmp_path, backend
):
    manager = _manager_for(
        backend, _team(tools=[lookup_codename]), _settings(tmp_path, thinking_effort="none"),
        CLAUDE,
    )
    try:
        looked_up = await collect_events(
            manager, "Call lookup_codename, then reply OK.", session_id="tools"
        )
        asked = await collect_events(
            manager,
            "Ask the archivist: which codename did the lookup earlier in this conversation return?",
            session_id="tools",
        )
    finally:
        await manager.cleanup()

    assert_no_errors(looked_up + asked)
    assert find_tool_results(looked_up, "lookup_codename")
    (archivist,) = _answers_of(asked, "archivist", backend)
    assert "marigold" in archivist.lower(), archivist


@pytest.mark.parametrize("backend", ["adk", "deepagents"])
@_needs(_key_for(CLAUDE))
async def test_a_fork_after_a_tool_call_in_the_same_turn_with_thinking_on(
    listed, tmp_path, backend
):
    # The coordinator calls a tool, then delegates, in one turn, with thinking
    # on: a fork inherits that tool exchange without the coordinator's
    # thinking (another agent's reasoning is never passed on). Claude accepts
    # the request.
    manager = _manager_for(
        backend, _team(tools=[lookup_codename]), _settings(tmp_path, thinking_effort="low"),
        CLAUDE,
    )
    try:
        events = await collect_events(
            manager,
            "First call lookup_codename. Then, in this same turn, ask the archivist: which "
            "codename did the lookup return? Tell me its answer.",
            session_id="same-turn",
        )
    finally:
        await manager.cleanup()

    assert_no_errors(events)
    assert find_tool_results(events, "lookup_codename")
    (archivist,) = _answers_of(events, "archivist", backend)
    assert "marigold" in archivist.lower(), archivist



@pytest.mark.parametrize("model", [
    pytest.param(CLAUDE, marks=_needs(_key_for(CLAUDE)), id="claude"),
    pytest.param(GEMINI, marks=_needs(_key_for(GEMINI)), id="gemini"),
    pytest.param(OPENAI, marks=_needs(_key_for(OPENAI)), id="openai"),
    pytest.param(OPENROUTER, marks=_needs(_key_for(OPENROUTER)), id="openrouter"),
    # Models that always think: the summary call runs with thinking off.
    pytest.param(OPUS, marks=_needs(_key_for(OPUS)), id="claude-opus-4-7"),
    pytest.param(GEMINI_3, marks=_needs(_key_for(GEMINI_3)), id="gemini-3"),
])
async def test_adk_summarizes_and_the_provider_accepts_the_summarized_conversation(
    listed, tmp_path, model
):
    manager = GoogleADKWorkflowManager(
        agent_configs=_assistant(include_state_tools=False),
        settings=_settings(tmp_path, context_max_tokens=16_000, thinking_effort="low"),
        model=model,
    )
    turns = []
    try:
        turns.append(await collect_events(
            manager,
            "The project we are discussing is named Quillfeather. Here are its garden "
            f"notes; reply OK.\n\n{_filler(6_000, 1)}",
            session_id="adk-summarized",
        ))
        for seed in (2, 3):
            turns.append(await collect_events(
                manager, f"More notes; reply OK.\n\n{_filler(6_000, seed)}",
                session_id="adk-summarized",
            ))
        for question in ("What is the project's name? Reply with the name only.",
                         "Thank you. Reply OK."):
            turns.append(await collect_events(manager, question, session_id="adk-summarized"))
    finally:
        await manager.cleanup()

    events = [event for turn in turns for event in turn]
    assert_no_errors(events)
    summaries = _summaries(events)
    assert summaries, "no summary was written"
    types = [e.type for e in events]
    assert types.index(EventType.CONTEXT_SUMMARIZING) < types.index(EventType.CONTEXT_TRIMMED)
    assert any(e.metadata.get("purpose") == "summarization"
               for e in find_events(events, EventType.LLM_USAGE))
    after = next(i for i, turn in enumerate(turns) if _summaries(turn))
    for turn in turns[after:]:
        assert _reply(turn).strip(), "a turn after the summary got no answer"


#: Dense text the character estimate counts low: CJK is about a token a
#: character, the estimate's 3.3 to 4 characters a token.
def _cjk(chars: int) -> str:
    base = "春夏秋冬山川河海日月星辰风雨雷电花草树木"
    return "".join(base[(i * 7) % len(base)] for i in range(chars))


#: How many CJK characters ``dense_notes`` returns: set per test.
_DENSE = {"chars": 0}


@register_tool(capabilities=EXEMPT)
def dense_notes() -> dict:
    """Return the archive's notes."""
    return {"success": True, "notes": _cjk(_DENSE["chars"])}


@pytest.mark.parametrize("model, chars", [
    # Claude Haiku: a 200,000-token window. 400,000 CJK characters are well
    # over it, yet estimated at about 133,000 tokens, under its budget.
    # (Gemini's 1M window cannot be passed this way: a request that large is
    # over a usual account's input tokens per minute, a 429 before any overflow.)
    pytest.param(HAIKU, 400_000, marks=_needs(_key_for(HAIKU)), id="claude-haiku"),
])
async def test_a_real_overflow_of_the_main_agent_ends_in_the_clear_message(
    listed, tmp_path, monkeypatch, model, chars
):
    monkeypatch.setitem(_DENSE, "chars", chars)
    manager = GoogleADKWorkflowManager(
        agent_configs=_assistant(tools=[dense_notes], include_state_tools=False),
        settings=_settings(tmp_path, tool_result_max_chars=5_000_000, thinking_effort="none"),
        model=model,
    )
    try:
        events = await collect_events(
            manager, "Call dense_notes, then tell me how many characters the notes have.",
            session_id="overflow",
        )
    finally:
        await manager.cleanup()

    assert find_tool_results(events, "dense_notes")
    errors = find_events(events, EventType.ERROR)
    assert len(errors) == 1, [e.content for e in errors]
    (error,) = errors
    assert error.metadata.get("error_code") == OVERFLOW_ERROR_CODE, error.content
    assert error.content.startswith("This conversation no longer fits "), error.content


@_needs(_key_for(CLAUDE), _key_for(HAIKU))
async def test_a_sub_agents_real_overflow_is_one_warning_and_the_coordinator_answers(
    listed, tmp_path, monkeypatch
):
    monkeypatch.setitem(_DENSE, "chars", 400_000)
    configs = [
        AgentConfig(name="coordinator", prompt=(
            "You coordinate the reader sub-agent. Ask it to read the archive's notes and "
            "summarize them, then tell the user what it said, even if it failed."),
            sub_agents=["reader"], include_state_tools=False),
        AgentConfig(name="reader", prompt="Call dense_notes, then summarize the notes.",
                    description="Reads the archive's notes.", tools=[dense_notes],
                    model=HAIKU, include_state_tools=False),
    ]
    manager = GoogleADKWorkflowManager(
        agent_configs=configs,
        settings=_settings(tmp_path, tool_result_max_chars=5_000_000, thinking_effort="none"),
        model=CLAUDE,
    )
    try:
        events = await collect_events(
            manager, "Have the reader summarize the archive's notes.", session_id="sub-overflow"
        )
    finally:
        await manager.cleanup()

    errors = find_events(events, EventType.ERROR)
    assert [(e.metadata.get("recoverable"), e.content.split(":")[0]) for e in errors] == [
        (True, "reader")
    ], [e.content for e in errors]
    assert "no longer fits" in errors[0].content
    assert "I could not do this task" in " ".join(_answers_of(events, "reader", "adk"))
    assert _reply(events).strip(), "the coordinator did not answer"


@_needs(_key_for(CLAUDE))
async def test_a_job_resumed_after_its_call_was_summarized(listed, tmp_path, monkeypatch):
    """A long-running call stays pending while ADK summarizes, then its result resumes."""
    from tests.integration.test_live_job_resume import (
        _SHAPE,
        _TOKEN,
        _wait_terminal,
        live_resume_probe,
    )

    monkeypatch.setitem(_SHAPE, "return_none", False)
    monkeypatch.chdir(tmp_path)
    agent = AgentConfig(
        name="job_runner",
        prompt=(
            "You run background jobs. When asked, call live_resume_probe with a shell "
            f"command that prints {_TOKEN} (e.g. 'echo {_TOKEN}'), then tell the user you "
            "started it. When you are given notes, reply OK. Never ask for confirmation."
        ),
        tools=[live_resume_probe],
        include_state_tools=False,
    )
    manager = GoogleADKWorkflowManager(
        agent_configs=[agent],
        settings=_settings(tmp_path, context_max_tokens=16_000, thinking_effort="none",
                           permissions_enabled=False),
        model=CLAUDE,
    )
    try:
        started = await collect_events(manager, "Start a background job now.", session_id="job")
        (record,) = [r for r in manager.job_manager.list() if r.resume_on_complete]
        notes = [
            await collect_events(
                manager, f"Garden notes; reply OK.\n\n{_filler(6_000, seed)}", session_id="job"
            )
            for seed in (1, 2, 3, 4)
        ]
        record = _wait_terminal(manager.job_manager, record.job_id)
        resumed = [event async for event in manager.resume_with_job_result(record)]
    finally:
        await manager.cleanup()

    assert find_tool_calls(started, "live_resume_probe")
    assert any(_summaries(turn) for turn in notes), "no summary while the call was pending"
    assert_no_errors(started + [event for turn in notes for event in turn] + resumed)
    assert _reply(resumed).strip(), "the resumed turn got no answer"



#: A system prompt long enough to be cached by every provider (Claude's
#: minimum is 1,024 tokens, Gemini's explicit cache's a few thousand).
_HANDBOOK = " ".join(
    f"Rule {n}: answer briefly, state facts plainly, and never invent a source."
    for n in range(1, 700)
)


@pytest.mark.parametrize("backend, model", [
    pytest.param("adk", CLAUDE, marks=_needs(_key_for(CLAUDE)), id="adk-claude"),
    pytest.param("deepagents", CLAUDE, marks=_needs(_key_for(CLAUDE)), id="deepagents-claude"),
    pytest.param("adk", GEMINI, marks=_needs(_key_for(GEMINI)), id="adk-gemini"),
    pytest.param("adk", OPENAI, marks=_needs(_key_for(OPENAI)), id="adk-openai"),
])
async def test_with_a_cache_hit_the_prompt_count_is_the_whole_prompt(
    listed, tmp_path, backend, model
):
    manager = _manager_for(
        backend, [AgentConfig(name="assistant", prompt=f"You are concise. {_HANDBOOK}",
                              include_state_tools=False)],
        _settings(tmp_path, thinking_effort="none"), model,
    )
    try:
        first = await collect_events(manager, "Say hello.", session_id="cache")
        second = await collect_events(manager, "Say goodbye.", session_id="cache")
    finally:
        await manager.cleanup()

    assert_no_errors(first + second)
    (written,) = find_events(first, EventType.LLM_USAGE)
    (read,) = find_events(second, EventType.LLM_USAGE)
    assert (read.metadata.get("cached_tokens") or 0) > 0, read.metadata
    tracker = UsageTracker()
    tracker.record(read.metadata)
    # The whole prompt, cached part included: about the first call's.
    assert tracker.last_prompt_tokens >= read.metadata["cached_tokens"]
    assert tracker.last_prompt_tokens >= 0.9 * written.metadata["prompt_tokens"], (
        written.metadata, read.metadata)


@_needs(_key_for(CLAUDE), _key_for(HAIKU))
async def test_with_the_raw_log_on_a_delegation_keeps_the_main_agents_ctx(listed, tmp_path):
    app_name = f"ctx-live-{uuid.uuid4().hex[:8]}"
    log = Path.home() / f".{app_name}" / "logs" / "llm_events.jsonl"
    configs = [
        AgentConfig(name="coordinator", prompt=(
            "You coordinate the calculator sub-agent: hand it every calculation, then "
            "give the user its result."), sub_agents=["calculator"], include_state_tools=False),
        AgentConfig(name="calculator", prompt="You multiply numbers. Reply with the product.",
                    description="Multiplies numbers.", model=HAIKU, include_state_tools=False),
    ]
    manager = GoogleADKWorkflowManager(
        agent_configs=configs,
        settings=_settings(tmp_path, app_name=app_name, raw_llm_logging=True,
                           thinking_effort="none"),
        model=CLAUDE,
    )
    try:
        events = await collect_events(manager, "What is 6 times 7?", session_id="logged")
        records = [json.loads(line) for line in log.read_text().splitlines()]
    finally:
        await manager.cleanup()
        shutil.rmtree(Path.home() / f".{app_name}", ignore_errors=True)

    assert_no_errors(events)
    usage = find_events(events, EventType.LLM_USAGE)
    tracker = UsageTracker()
    for event in usage:
        tracker.record(event.metadata)
    coordinator = [e for e in usage if e.metadata.get("agent") == "coordinator"]
    calculator = [e for e in usage if e.metadata.get("agent") == "calculator"]
    assert calculator and all(e.metadata.get("purpose") == "subagent" for e in calculator)
    assert {e.metadata["model"] for e in calculator} == {HAIKU}
    # ctx is the coordinator's last prompt, never the sub-agent's.
    assert tracker.last_prompt_tokens == coordinator[-1].metadata["prompt_tokens"]
    # One usage per call: the raw log adds none to the stream.
    assert len(usage) == len([r for r in records if r.get("type") == "llm_response"]), records
    # The log names each call's own model.
    by_agent = {}
    for record in records:
        meta = record.get("metadata") or {}
        if record.get("type") == "llm_response":
            by_agent.setdefault(meta.get("author") or meta.get("agent"), set()).add(
                str(meta.get("model") or record.get("model")))
    assert any("haiku" in model for model in by_agent.get("calculator", set())), by_agent
    assert all("haiku" not in model for model in by_agent.get("coordinator", set())), by_agent



@pytest.mark.parametrize("backend, model", [
    pytest.param("adk", CLAUDE, marks=_needs(_key_for(CLAUDE)), id="adk-claude"),
    pytest.param("deepagents", CLAUDE, marks=_needs(_key_for(CLAUDE)), id="deepagents-claude"),
    pytest.param("adk", GEMINI, marks=_needs(_key_for(GEMINI)), id="adk-gemini"),
    pytest.param("deepagents", GEMINI, marks=_needs(_key_for(GEMINI)), id="deepagents-gemini"),
    pytest.param("adk", OPENAI, marks=_needs(_key_for(OPENAI)), id="adk-openai"),
    pytest.param("deepagents", OPENAI, marks=_needs(_key_for(OPENAI)), id="deepagents-openai"),
])
async def test_compact_writes_a_summary_now_and_the_next_request_is_accepted(
    listed, tmp_path, backend, model
):
    manager = _manager_for(
        backend, _assistant(include_state_tools=False),
        _settings(tmp_path, context_max_tokens=16_000, thinking_effort="none"), model,
    )
    try:
        for seed in (1, 2):
            await collect_events(
                manager, f"Garden notes; reply OK.\n\n{_filler(2_000, seed)}",
                session_id="compact", user_id="u",
            )
        # Braces and quotes in the focus: it is spliced into a str.format prompt.
        result, reported = await compact_now(
            manager, 'keep the {garden} names and "week" numbers',
            session_id="compact", user_id="u",
        )
        after = await collect_events(
            manager, "Which streets had gardens? Name two.", session_id="compact", user_id="u"
        )
    finally:
        await manager.cleanup()

    assert result is not None and result.messages_summarized > 0, reported
    trimmed = [e for e in reported if e.type is EventType.CONTEXT_TRIMMED]
    assert [e.metadata["source"] for e in trimmed] == ["compact"]
    assert any(e.metadata.get("purpose") == "summarization"
               for e in find_events(reported, EventType.LLM_USAGE))
    assert_no_errors(after)
    assert _reply(after).strip(), "the request after /compact got no answer"


@_needs(_key_for(CLAUDE))
async def test_a_deep_agents_compact_near_the_trigger_is_not_followed_by_a_second_summary(
    listed, tmp_path
):
    manager = _manager_for(
        "deepagents", _assistant(include_state_tools=False),
        _settings(tmp_path, context_max_tokens=16_000, thinking_effort="none"), CLAUDE,
    )
    try:
        # About 13,000 tokens: under the 16,000-token trigger, but the reported
        # count of the last answer is near it.
        for seed in (1, 2):
            await collect_events(
                manager, f"Garden notes; reply OK.\n\n{_filler(6_000, seed)}",
                session_id="near", user_id="u",
            )
        result, _ = await compact_now(manager, session_id="near", user_id="u")
        after = await collect_events(manager, "Reply OK.", session_id="near", user_id="u")
    finally:
        await manager.cleanup()

    assert result is not None
    assert_no_errors(after)
    assert not _summaries(after), "the next call summarized again"
    assert _reply(after).strip()
