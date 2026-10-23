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

Override the models with LIVE_CONTEXT_CLAUDE_MODEL (default
``anthropic:claude-sonnet-4-5``), LIVE_CONTEXT_GEMINI_MODEL (default
``google:gemini-2.5-flash``), LIVE_CONTEXT_OPENAI_MODEL (default
``openai:gpt-5-mini``), LIVE_CONTEXT_HAIKU_MODEL (default
``anthropic:claude-haiku-4-5``) and LIVE_CONTEXT_SMALL_MODEL (an ``openrouter:`` id;
by default the cheapest tool-calling model OpenRouter lists with a 32k to 64k
window).
"""

from __future__ import annotations

import os

import httpx
import pytest

pytest.importorskip("deepagents")

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
