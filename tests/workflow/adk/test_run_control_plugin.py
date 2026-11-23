"""ADK: a sub-agent run's limits, and the tools a narrowed run is offered (its
registered, skill and MCP tools, and ADK's built-ins), at its model calls
(``workflow/adk/run_control_plugin.py``).

What the plugin does to a request and an answer, alone, and one whole ADK run
that only ADK's built-ins concern. Whole turns, on both backends, are in the
contract suite (``TestTheStepLimit``, ``TestTheTokenLimit``,
``TestANarrowedRun``).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.agents.callback_context import CallbackContext  # noqa: E402
from google.adk.agents.invocation_context import InvocationContext  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.tools import BaseTool, FunctionTool, google_search  # noqa: E402
from google.adk.tools.google_search_tool import GoogleSearchTool  # noqa: E402
from google.adk.tools.tool_context import ToolContext  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.workflow.adk.run_control_plugin import (  # noqa: E402
    RunControlPlugin,
    RunToolsPlugin,
    drop_calls,
    narrow,
    with_note,
)
from agentic_cli.workflow.agents.control import STEP_LIMIT_NOTE, RunControl  # noqa: E402
from agentic_cli.workflow.agents.envelope import TOKEN_LIMIT_ERROR  # noqa: E402
from agentic_cli.workflow.agents.policy import AgentLimits  # noqa: E402
from agentic_cli.workflow.sessions import RunRef  # noqa: E402

REF = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)
#: A model ADK's built-ins run on.
GEMINI = "gemini-2.5-flash"


def user(*parts: types.Part) -> types.Content:
    return types.Content(role="user", parts=list(parts))


def model(*parts: types.Part) -> types.Content:
    return types.Content(role="model", parts=list(parts))


def text(value: str, **fields) -> types.Part:
    return types.Part(text=value, **fields)


def call(name: str) -> types.Part:
    return types.Part.from_function_call(name=name, args={})


def answered(name: str) -> types.Part:
    return types.Part.from_function_response(name=name, response={"success": True})


class TestTheNote:
    def test_it_is_a_part_of_a_last_user_content(self):
        result = user(answered("lookup"))
        contents = [user(text("# Task: Find X.")), model(call("lookup")), result]

        noted = with_note(contents, STEP_LIMIT_NOTE)

        assert noted[:2] == contents[:2]
        assert noted[2].parts[0] == answered("lookup")
        assert noted[2].parts[1].text == STEP_LIMIT_NOTE
        assert result.parts == [answered("lookup")]  # not changed in place

    def test_after_a_models_content_it_is_a_user_content_of_its_own(self):
        contents = [user(text("# Task: Find X.")), model(text("Hm."))]

        assert with_note(contents, STEP_LIMIT_NOTE)[-1] == user(text(STEP_LIMIT_NOTE))


class TestDroppedCalls:
    def test_the_text_and_thoughts_stay(self):
        answer = LlmResponse(content=model(text("why", thought=True), text("X is 42."),
                                           call("lookup")))

        drop_calls(answer)

        assert answer.content == model(text("why", thought=True), text("X is 42."))

    def test_calls_alone_leave_an_empty_text(self):
        answer = LlmResponse(content=model(text("why", thought=True), call("lookup")))

        drop_calls(answer)

        assert answer.content == model(text("why", thought=True), text(""))

    def test_an_answer_without_calls_is_left_as_it_is(self):
        content = model(text("X is 42."))
        answer = LlmResponse(content=content)

        drop_calls(answer)

        assert answer.content is content


class TestThePlugin:
    async def test_the_roots_calls_are_left_alone(self):
        plugin = RunControlPlugin(lambda: None, "anthropic")
        request = LlmRequest(contents=[user(text("Go."))])
        answer = LlmResponse(content=model(call("lookup")))

        assert await plugin.before_model_callback(callback_context=None, llm_request=request) is None
        assert await plugin.after_model_callback(callback_context=None, llm_response=answer) is None

        assert request.contents == [user(text("Go."))]
        assert answer.content == model(call("lookup"))

    async def test_a_call_past_the_token_limit_is_the_runs_last(self):
        control = RunControl(REF, AgentLimits(max_tokens=100))
        plugin = RunControlPlugin(lambda: control, "anthropic")
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=90, candidates_token_count=20
        )
        answer = LlmResponse(content=model(text("Found it."), call("lookup")), usage_metadata=usage)

        await plugin.before_model_callback(callback_context=None, llm_request=LlmRequest())
        await plugin.after_model_callback(callback_context=None, llm_response=answer)

        assert control.cancelled == ("limit", TOKEN_LIMIT_ERROR)
        assert answer.content == model(text("Found it."))

    async def test_a_finishing_answer_past_the_token_limit_is_left_as_it_is(self):
        control = RunControl(REF, AgentLimits(max_tokens=100))
        plugin = RunControlPlugin(lambda: control, "anthropic")
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=90, candidates_token_count=20
        )
        content = model(text("Found it."))
        answer = LlmResponse(content=content, usage_metadata=usage)

        await plugin.before_model_callback(callback_context=None, llm_request=LlmRequest())
        await plugin.after_model_callback(callback_context=None, llm_response=answer)

        assert (control.cancelled, control.final_call) == (None, False)
        assert answer.content is content

    async def test_an_answer_with_no_usage_counts_none_of_its_own_tokens(self):
        control = RunControl(REF, AgentLimits(max_tokens=1))
        plugin = RunControlPlugin(lambda: control, "anthropic")
        answer = LlmResponse(content=model(text("Found it."), call("lookup")))

        await plugin.before_model_callback(callback_context=None, llm_request=LlmRequest())
        await plugin.after_model_callback(callback_context=None, llm_response=answer)

        assert (control.cancelled, control.final_call) == (None, False)
        assert answer.content == model(text("Found it."), call("lookup"))

    async def test_an_answer_with_no_usage_still_counts_the_runs_side_calls(self):
        control = RunControl(REF, AgentLimits(max_tokens=100), side_tokens=lambda: 120)
        plugin = RunControlPlugin(lambda: control, "anthropic")
        answer = LlmResponse(content=model(text("Found it."), call("lookup")))

        await plugin.before_model_callback(callback_context=None, llm_request=LlmRequest())
        await plugin.after_model_callback(callback_context=None, llm_response=answer)

        assert control.cancelled == ("limit", TOKEN_LIMIT_ERROR)
        assert answer.content == model(text("Found it."))

    async def test_a_partial_answer_counts_nothing(self):
        control = RunControl(REF, AgentLimits(max_tokens=1))
        plugin = RunControlPlugin(lambda: control, "anthropic")
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=90, candidates_token_count=20
        )

        await plugin.after_model_callback(
            callback_context=None,
            llm_response=LlmResponse(content=model(text("Fo")), usage_metadata=usage, partial=True),
        )

        assert control.cancelled is None


class TestSteering:
    """Each steered message is the run's unstored one until its child
    session holds it: a run stopped while it stores them keeps the rest for
    its instance's next run."""

    class Stalling:
        """A session service whose second append never returns."""

        def __init__(self) -> None:
            self.stored: list[str] = []
            self.stalled = asyncio.Event()

        async def append_event(self, session, event):
            if self.stored:
                self.stalled.set()
                await asyncio.Event().wait()
            self.stored.append(event.content.parts[0].text)
            return event

    @staticmethod
    def context(service) -> SimpleNamespace:
        return SimpleNamespace(
            invocation_id="i1", session=object(),
            get_invocation_context=lambda: SimpleNamespace(session_service=service),
        )

    async def test_a_message_stored_is_no_more_unstored(self):
        control = RunControl(REF, AgentLimits())
        service = self.Stalling()
        control.steer("Also Z.")
        request = LlmRequest(contents=[user(text("# Task: Find X."))])

        await RunControlPlugin(lambda: control, "anthropic").before_model_callback(
            callback_context=self.context(service), llm_request=request
        )

        assert service.stored == ["Also Z."]
        assert control.unstored() == []
        assert request.contents[-1] == user(text("Also Z."))

    async def test_a_run_stopped_while_storing_keeps_the_rest_unstored(self):
        control = RunControl(REF, AgentLimits())
        service = self.Stalling()
        control.steer("Also Z.")
        control.steer("In metric units.")
        plugin = RunControlPlugin(lambda: control, "anthropic")

        call = asyncio.create_task(plugin.before_model_callback(
            callback_context=self.context(service),
            llm_request=LlmRequest(contents=[user(text("# Task: Find X."))]),
        ))
        await asyncio.wait_for(service.stalled.wait(), 5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        assert service.stored == ["Also Z."]
        assert control.unstored() == ["In metric units."]


def offering(*names: str) -> LlmRequest:
    """A request that offers a function tool of each of ``names``, and Google search."""
    request = LlmRequest(contents=[user(text("# Task: Find X."))])
    request.config.tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(name=name, description=name) for name in names
        ]),
        types.Tool(google_search=types.GoogleSearch()),
    ]
    request.tools_dict = {name: BaseTool(name=name, description=name) for name in names}
    return request


def lookup(q: str) -> dict:
    """Look ``q`` up."""
    return {"success": True}


async def agent_context(tools: list) -> CallbackContext:
    """A real callback context of a Gemini agent with ``tools``, as ADK passes
    a plugin's model callbacks."""
    service = InMemorySessionService()
    session = await service.create_session(app_name="app", user_id="u", session_id="s")
    agent = LlmAgent(name="researcher", model=GEMINI, tools=list(tools))
    return CallbackContext(InvocationContext(
        session_service=service, invocation_id="i", agent=agent, session=session,
    ))


async def processed(context: CallbackContext, tools: list) -> LlmRequest:
    """The request ADK builds once each of ``tools`` processed it, before the
    model callbacks."""
    request = LlmRequest(model=GEMINI, contents=[user(text("# Task: Find X."))])
    for tool in tools:
        await tool.process_llm_request(
            tool_context=ToolContext(context.get_invocation_context()), llm_request=request
        )
    return request


def declared(request: LlmRequest) -> list[str]:
    return [
        d.name for tool in request.config.tools or ()
        for d in tool.function_declarations or ()
    ]


class TestANarrowedRunsTools:
    """A run narrowed to some of its type's tools (``RunControl.tools``) is
    offered those and the tools every run keeps, in ``tools_dict`` (what ADK
    resolves a call from) and in what the model is sent."""

    async def test_its_tools_and_those_every_run_keeps_and_no_other(self):
        control = RunControl(REF, AgentLimits(), tools=frozenset({"lookup"}))
        plugin = RunToolsPlugin(lambda: control)
        request = offering("lookup", "fetch_page", "save_plan", "task", "check_agents")

        await plugin.before_model_callback(callback_context=None, llm_request=request)

        assert list(request.tools_dict) == ["lookup", "save_plan", "task",
                                            "check_agents"]
        assert declared(request) == ["lookup", "save_plan", "task", "check_agents"]
        # What no tool of the agent's is known to have made (here no agent) stays.
        assert request.config.tools[1].google_search is not None

    @pytest.mark.parametrize("kept, searches", [
        ({"lookup"}, False), ({"lookup", "google_search"}, True),
    ], ids=["without-it", "keeping-it"])
    async def test_a_built_in_goes_unless_the_run_keeps_it(self, kept, searches):
        """ADK's built-ins (``google_search``) run inside the model: what one adds
        to a request is no function declaration, and nothing in ``tools_dict``.
        A run narrowed without it is not sent it; one that keeps it is sent what
        ADK made, as it made it."""
        tools = [FunctionTool(lookup), google_search]
        context = await agent_context(tools)
        request = await processed(context, tools)
        made = list(request.config.tools)
        control = RunControl(REF, AgentLimits(), tools=frozenset(kept))

        await RunToolsPlugin(lambda: control).before_model_callback(
            callback_context=context, llm_request=request
        )

        assert any(tool.google_search is not None for tool in request.config.tools) is searches
        assert (declared(request), list(request.tools_dict)) == (["lookup"], ["lookup"])
        assert (made[1] in request.config.tools) is searches  # ADK's own entry, kept

    def test_narrow_leaves_out_the_entries_of_the_built_ins_it_refuses(self):
        request = offering("lookup")
        search = request.config.tools[1]
        code = types.Tool(code_execution=types.ToolCodeExecution())
        request.config.tools.append(code)

        narrow(request, lambda name: name == "lookup",
               built_ins={"google_search": [search], "url_context": []})

        # A code executor's entry is no tool's: it stays.
        assert request.config.tools[1:] == [code]
        assert declared(request) == ["lookup"]

    async def test_a_built_in_no_run_narrows_is_never_asked(self):
        """A run not narrowed has every tool as ADK made it, built-ins too."""
        tools = [FunctionTool(lookup), google_search]
        context = await agent_context(tools)
        request = await processed(context, tools)
        made = list(request.config.tools)
        control = RunControl(REF, AgentLimits())

        await RunToolsPlugin(lambda: control).before_model_callback(
            callback_context=context, llm_request=request
        )

        assert request.config.tools == made

    def test_a_tool_left_with_no_declaration_goes(self):
        request = LlmRequest(contents=[user(text("Go."))])
        request.config.tools = [types.Tool(function_declarations=[
            types.FunctionDeclaration(name="fetch_page", description="Fetch."),
        ])]
        request.tools_dict = {"fetch_page": BaseTool(name="fetch_page",
                                                     description="Fetch.")}

        narrow(request, lambda name: False)

        assert (request.tools_dict, request.config.tools) == ({}, None)

    async def test_a_run_not_narrowed_keeps_them_all(self):
        control = RunControl(REF, AgentLimits())
        plugin = RunToolsPlugin(lambda: control)
        request = offering("lookup", "fetch_page")

        await plugin.before_model_callback(callback_context=None, llm_request=request)

        assert (list(request.tools_dict), declared(request)) == (
            ["lookup", "fetch_page"], ["lookup", "fetch_page"]
        )


class CountedBuiltIn(BaseTool):
    """A built-in of an app's: it adds a request entry, and counts how often it was asked."""

    def __init__(self) -> None:
        super().__init__(name="counted_search", description="counted_search")
        self.asked = 0

    async def process_llm_request(self, *, tool_context, llm_request) -> None:
        self.asked += 1
        llm_request.config.tools = [
            *(llm_request.config.tools or ()), types.Tool(google_search=types.GoogleSearch()),
        ]


class TestABuiltInsEntriesLearned:
    """What a refused built-in adds to a request is learned once per model, for
    the build (the plugin, made again by a reinitialize): its
    ``process_llm_request`` may do work, and runs again at no model call."""

    async def test_once_for_two_model_calls_of_a_narrowed_run(self):
        counted = CountedBuiltIn()
        tools = [FunctionTool(lookup), counted]
        context = await agent_context(tools)
        control = RunControl(REF, AgentLimits(), tools=frozenset({"lookup"}))
        plugin = RunToolsPlugin(lambda: control)
        requests = [await processed(context, tools) for _ in range(2)]
        asked = counted.asked  # ADK's own processing of each request

        for request in requests:
            await plugin.before_model_callback(callback_context=context, llm_request=request)

        assert counted.asked - asked == 1
        assert all(not request.config.tools[1:] for request in requests)

    async def test_one_that_cannot_say_drops_nothing_and_is_logged_once(self, monkeypatch):
        """A built-in that raises on its own request (ADK's ``google_search`` with
        ``bypass_multi_tools_limit`` on a model it does not serve: ADK sends it
        as an agent tool instead) is taken to add nothing; its run goes on."""
        from agentic_cli.workflow.adk import run_control_plugin

        logged: list[tuple[str, dict]] = []
        monkeypatch.setattr(run_control_plugin, "logger", type("Recorder", (), {
            "debug": staticmethod(lambda event, **fields: logged.append((event, fields))),
        })())
        search = GoogleSearchTool(bypass_multi_tools_limit=True)
        context = await agent_context([FunctionTool(lookup), search])
        control = RunControl(REF, AgentLimits(), tools=frozenset({"lookup"}))
        plugin = RunToolsPlugin(lambda: control)

        for _ in range(2):
            request = offering("lookup")
            request.model = "claude-sonnet-4-6"
            await plugin.before_model_callback(callback_context=context, llm_request=request)
            assert request.config.tools[1].google_search is not None  # nothing dropped

        assert [(event, fields["tool"], fields["model"]) for event, fields in logged] == [
            ("built_in_entries_unknown", "google_search", "claude-sonnet-4-6"),
        ]
        assert "not supported" in logged[0][1]["error"]


# --- a whole run --------------------------------------------------------------------


@pytest.fixture
def claude_project(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, with a fake Claude key."""
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")


async def test_a_run_narrowed_without_a_bypass_google_search_on_claude_completes(
    claude_project, monkeypatch
):
    """ADK sends ``google_search`` with ``bypass_multi_tools_limit`` as an agent
    tool, and it serves Gemini alone: on Claude, its own request processing
    raises. A run narrowed without it runs as it did before narrowing knew of
    built-ins."""
    from tests.workflow.test_subagents_contract import (
        SIDE_CALLS, SUMMARIES, _adk_crew, agent, calls, contract_lookup, delegate,
        envelope_read, says,
    )

    crew = _adk_crew(
        [agent("coordinator", sub_agents=("researcher",)),
         agent("researcher", tools=(contract_lookup,
                                    GoogleSearchTool(bypass_multi_tools_limit=True)))],
        {"coordinator": [calls(delegate("researcher", "Find X.", "c1",
                                        tools=["contract_lookup"])), says("Done.")],
         "researcher": [says("X is 42.")], SIDE_CALLS: [], SUMMARIES: []},
        monkeypatch,
    )
    try:
        await crew.turn()
        envelope = envelope_read(crew.results_read("coordinator", 1)["c1"])
    finally:
        await crew.manager.cleanup()

    assert (envelope["status"], envelope["result"]) == ("completed", "X is 42.")
