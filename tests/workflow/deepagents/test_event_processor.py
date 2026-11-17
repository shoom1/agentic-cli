"""The event stream (``workflow/deepagents/event_processor.py``).

A turn streams the root agent's graph, ``astream(..., stream_mode=["updates"],
subgraphs=True)``: ``(namespace, mode, payload)`` items. A sub-agent's graph
streams apart, in its run's own task, through a processor made for that run
(``run=``). ``DeepAgentsEventProcessor`` turns each finished node output into
``WorkflowEvent``s, one row of the spec's mapping per test class, each the
processor's agent's and run's: the root's own, or the sub-agent run's.

Most tests feed the processor payloads shaped as the stream sends them; the
last runs a real manager on scripted models and reads its turn's events.
"""

from __future__ import annotations

import json
import uuid

import anthropic
import httpx
import pytest

pytest.importorskip("deepagents")

from langchain_core.callbacks import AsyncCallbackHandler  # noqa: E402
from langchain_core.exceptions import ContextOverflowError  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult  # noqa: E402
from langgraph.graph.message import REMOVE_ALL_MESSAGES  # noqa: E402

from agentic_cli.models import ModelCatalog, RoleMap, resolve_model  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.tools._core.tasks import task_progress_data  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import (  # noqa: E402
    OVERFLOW_ERROR_CODE,
    ContextBudget,
    overflow_message,
)
from agentic_cli.workflow.deepagents.event_processor import (  # noqa: E402
    DeepAgentsEventProcessor,
    RunCallback,
)
from agentic_cli.workflow.agents.envelope import envelope  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.live_events import LiveEvents, merged  # noqa: E402
from agentic_cli.workflow.sessions import RunRef  # noqa: E402
from tests.models.fakes import providers  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai, task  # noqa: E402

_PROVIDERS = providers(anthropic="a-test-key", google="g-test-key", openai="o-test-key")


def _model(agent: str, spec: str):
    """``spec`` resolved for ``agent`` against the shipped table."""
    return resolve_model(
        agent=agent,
        spec=spec,
        generation=None,
        thinking_effort="none",
        roles=RoleMap(default_model=None, model_roles={}),
        catalog=ModelCatalog(providers=_PROVIDERS),
        providers=_PROVIDERS,
    )


MODELS = {
    "coordinator": _model("coordinator", "anthropic:claude-sonnet-4-5"),
    "researcher": _model("researcher", "google:gemini-2.5-flash"),
    "worker": _model("worker", "openai:gpt-5"),
}

#: Each agent's context numbers, as the manager computes them when it builds
#: the agents: two distinct triggers, one of them on an unknown window.
BUDGETS = {
    "coordinator": ContextBudget(window=200_000, window_known=True, budget=181_808,
                                 trigger=154_536, keep=18_544),
    "researcher": ContextBudget(window=128_000, window_known=False, budget=121_600,
                                trigger=103_360, keep=12_403),
}
CONTEXT_KEYS = ("agent", "context_trigger", "window_known")

RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)


def _processor(**kwargs) -> DeepAgentsEventProcessor:
    return DeepAgentsEventProcessor(root="coordinator", agent_models=MODELS, **kwargs)


def _run_processor(**kwargs) -> DeepAgentsEventProcessor:
    """A processor made for the researcher's run, as ``_run_agent`` makes one."""
    return _processor(run=RESEARCHER, **kwargs)


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _model_update(*messages) -> dict:
    return {"model": {"messages": list(messages)}}


def _tools_update(*messages, **state) -> dict:
    return {"tools": {**state, "messages": list(messages)}}


def _types(events) -> list[EventType]:
    return [event.type for event in events]


class TestText:
    def test_a_model_answer_is_text(self):
        events = _processor(session_id="s1").process(
            (), "updates", _model_update(AIMessage("Hello."))
        )

        (event,) = [e for e in events if e.type is EventType.TEXT]
        assert event.content == "Hello."
        assert event.metadata["agent"] == "coordinator"
        assert event.metadata["session_id"] == "s1"

    def test_each_text_block_is_one_event(self):
        message = AIMessage(content=[
            {"type": "text", "text": "One."}, {"type": "text", "text": "Two."}
        ])

        events = _processor().process((), "updates", _model_update(message))

        assert [e.content for e in events if e.type is EventType.TEXT] == ["One.", "Two."]

    def test_an_empty_answer_has_no_text(self):
        events = _processor().process((), "updates", _model_update(AIMessage("")))

        assert EventType.TEXT not in _types(events)

    def test_the_roots_text_is_the_reply(self):
        events = _processor().process((), "updates", _model_update(AIMessage("Hello.")))

        (event,) = [e for e in events if e.type is EventType.TEXT]
        assert "subagent" not in event.metadata

    def test_a_sub_agents_text_is_its_activity(self):
        # Only the root replies to the user; a sub-agent's text is its work,
        # and its answer reaches the coordinator as the task call's result.
        events = _run_processor().process((), "updates", _model_update(AIMessage("It is 42.")))

        (event,) = [e for e in events if e.type is EventType.TEXT]
        assert (event.metadata["agent"], event.metadata["subagent"]) == ("researcher", True)
        assert event.run == RESEARCHER


class TestThinking:
    def test_reasoning_comes_before_the_text_it_precedes(self):
        message = AIMessage(content=[
            {"type": "reasoning", "reasoning": "Let me think."},
            {"type": "text", "text": "42."},
        ])

        events = _processor(session_id="s1").process((), "updates", _model_update(message))

        assert [(e.type, e.content) for e in events][:2] == [
            (EventType.THINKING, "Let me think."),
            (EventType.TEXT, "42."),
        ]
        assert events[0].metadata == {"session_id": "s1", "agent": "coordinator"}

    @pytest.mark.parametrize(
        "provider, block",
        [
            ("anthropic", {"type": "thinking", "thinking": "deep", "signature": "sig"}),
            ("google_genai", {"type": "thinking", "thinking": "deep"}),
            ("openai", {"type": "reasoning", "summary": [{"type": "summary_text", "text": "deep"}]}),
        ],
    )
    def test_every_providers_reasoning_is_thinking(self, provider, block):
        message = AIMessage(
            content=[block, {"type": "text", "text": "Done."}],
            response_metadata={"model_provider": provider},
        )

        events = _processor().process((), "updates", _model_update(message))

        assert [(e.type, e.content) for e in events][:2] == [
            (EventType.THINKING, "deep"),
            (EventType.TEXT, "Done."),
        ]


class TestToolCall:
    def test_one_event_per_call(self):
        message = AIMessage("", tool_calls=[
            _call("read_file", {"path": "a.txt"}, "c1"),
            _call("nope", {}, "c2"),  # unknown: answered by the gate, still a call
        ])

        events = _processor().process((), "updates", _model_update(message))

        calls = [e for e in events if e.type is EventType.TOOL_CALL]
        assert [(e.metadata["tool_name"], e.metadata["tool_args"]) for e in calls] == [
            ("read_file", {"path": "a.txt"}),
            ("nope", {}),
        ]
        assert [e.metadata["tool_call_id"] for e in calls] == ["c1", "c2"]
        assert all(e.metadata["agent"] == "coordinator" for e in calls)

    def test_delegation_is_a_task_call(self):
        brief = {"agent_type": "researcher", "title": "Find X", "objective": "Find X."}
        message = AIMessage("", tool_calls=[_call("task", brief, "c1")])

        (event,) = [
            e for e in _processor().process((), "updates", _model_update(message))
            if e.type is EventType.TOOL_CALL
        ]

        assert event.metadata["tool_name"] == "task"
        assert event.metadata["tool_args"] == brief

    def test_a_denied_call_is_a_call_then_a_failed_result(self):
        processor = _processor()
        processor.process((), "updates", _model_update(
            AIMessage("", tool_calls=[_call("write_file", {"path": "/etc/x"}, "c1")])
        ))
        denial = ToolMessage(
            json.dumps({"success": False, "error": "Permission denied: rule: builtin/deny"}),
            tool_call_id="c1", name="write_file", status="error",
        )

        (event,) = processor.process((), "updates", _tools_update(denial))

        assert event.type is EventType.TOOL_RESULT
        assert event.metadata["success"] is False
        assert event.metadata["error"] == "Permission denied: rule: builtin/deny"


class TestToolResult:
    def _result(self, message: ToolMessage, **state) -> WorkflowEvent:
        events = _processor().process((), "updates", _tools_update(message, **state))
        return next(e for e in events if e.type is EventType.TOOL_RESULT)

    def test_a_json_dict_is_parsed_back(self):
        event = self._result(ToolMessage(
            json.dumps({"success": True, "matches": 3}), tool_call_id="c1", name="grep"
        ))

        assert event.metadata["tool_name"] == "grep"
        assert event.metadata["result"] == {"success": True, "matches": 3}
        assert event.metadata["success"] is True
        assert event.metadata["tool_call_id"] == "c1"
        assert event.metadata["agent"] == "coordinator"

    def test_plain_text_stays_text(self):
        event = self._result(ToolMessage("X is 42.", tool_call_id="c1", name="task"))

        assert event.metadata["result"] == "X is 42."
        assert event.metadata["success"] is True

    def test_success_false_is_a_failure(self):
        event = self._result(ToolMessage(
            json.dumps({"success": False, "error": "no such file"}), tool_call_id="c1", name="read_file"
        ))

        assert event.metadata["success"] is False
        assert event.metadata["error"] == "no such file"

    def test_status_error_is_a_failure(self):
        event = self._result(ToolMessage(
            "Error: the arguments were malformed.", tool_call_id="c1", name="grep", status="error"
        ))

        assert event.metadata["success"] is False
        assert event.metadata["result"] == "Error: the arguments were malformed."
        assert event.metadata["error"] == "Error: the arguments were malformed."

    def test_a_result_without_a_name_takes_its_calls(self):
        processor = _processor()
        processor.process((), "updates", _model_update(
            AIMessage("", tool_calls=[_call("lookup", {"q": "x"}, "c1")])
        ))

        (event,) = processor.process(
            (), "updates", _tools_update(ToolMessage("Done.", tool_call_id="c1"))
        )

        assert event.metadata["tool_name"] == "lookup"

    def test_a_state_tools_command_is_its_tool_message(self):
        """``save_plan`` returns a ``Command``; the update carries its state
        and its ToolMessage, never the Command itself."""
        message = ToolMessage(
            json.dumps({"success": True, "message": "Plan saved"}),
            tool_call_id="c1", name="save_plan",
        )

        events = _processor().process((), "updates", _tools_update(message, plan="- [ ] a"))

        (event,) = events
        assert event.metadata["result"] == {"success": True, "message": "Plan saved"}
        assert "Command" not in event.content

    def test_a_task_command_is_the_runs_envelope(self):
        tasks = [{"id": "t1", "description": "find X", "status": "completed"}]
        found = envelope(RESEARCHER, "completed", result="X is 42.\nIt says so.")

        events = _processor().process((), "updates", _tools_update(
            ToolMessage(json.dumps(found), tool_call_id="c1", name="task"), tasks=tasks
        ))

        assert _types(events) == [EventType.TOOL_RESULT, EventType.TASK_PROGRESS]
        assert events[0].metadata["result"] == found
        assert (events[0].content, events[0].metadata["success"]) == ("X is 42.", True)

    def test_a_failed_runs_envelope_is_a_failed_result(self):
        found = envelope(RESEARCHER, "failed", error="overloaded")

        (event,) = _processor().process((), "updates", _tools_update(
            ToolMessage(json.dumps(found), tool_call_id="c1", name="task")
        ))

        assert event.content == "Failed: overloaded"
        assert (event.metadata["success"], event.metadata["error"]) == (False, "overloaded")

    def test_an_empty_answer_is_shown_as_no_answer(self):
        found = envelope(RESEARCHER, "completed", result="")

        (event,) = _processor().process((), "updates", _tools_update(
            ToolMessage(json.dumps(found), tool_call_id="c1", name="task")
        ))

        assert (event.content, event.metadata["success"]) == ("No answer", True)

    def test_a_tool_internal_model_call_is_not_text(self):
        """``generate_simple`` inside a tool is not a graph node: the stream
        has only the tool's result."""
        events = _processor().process((), "updates", _tools_update(
            ToolMessage(json.dumps({"success": True, "summary": "A page."}),
                        tool_call_id="c1", name="web_fetch")
        ))

        assert _types(events) == [EventType.TOOL_RESULT]


class TestTaskProgress:
    TASKS = [
        {"id": "t1", "description": "find X", "status": "completed"},
        {"id": "t2", "description": "write it up", "status": "in_progress"},
    ]

    def test_a_tasks_update_reports_progress_in_the_same_step(self):
        message = ToolMessage(json.dumps({"success": True}), tool_call_id="c1", name="save_tasks")

        events = _processor().process((), "updates", _tools_update(message, tasks=self.TASKS))

        assert _types(events) == [EventType.TOOL_RESULT, EventType.TASK_PROGRESS]
        progress = task_progress_data(self.TASKS)
        event = events[1]
        assert event.content == progress["display"]
        assert event.metadata["progress"] == progress["progress"]
        assert event.metadata["current_task_id"] == "t2"
        assert event.metadata["current_task_description"] == "write it up"
        assert event.metadata["agent"] == "coordinator"

    def test_a_sub_agents_list_is_reported_once_as_it_comes_back(self):
        """A sub-agent's list is the conversation's, which its run hands back
        with its result: the root's processor reports it then, once."""
        processor = _processor()
        message = ToolMessage(json.dumps({"success": True}), tool_call_id="r1", name="save_tasks")

        in_the_run = _run_processor().process((), "updates", _tools_update(message, tasks=self.TASKS))
        back = processor.process((), "updates", _tools_update(
            ToolMessage("Done.", tool_call_id="c1", name="task"), tasks=self.TASKS
        ))
        again = processor.process((), "updates", _tools_update(
            ToolMessage("Done.", tool_call_id="c2", name="task"), tasks=self.TASKS
        ))

        assert _types(in_the_run) == [EventType.TOOL_RESULT]
        assert _types(back) == [EventType.TOOL_RESULT, EventType.TASK_PROGRESS]
        assert back[1].metadata["agent"] == "coordinator"
        assert _types(again) == [EventType.TOOL_RESULT]

    def test_an_empty_list_reports_nothing(self):
        message = ToolMessage(json.dumps({"success": True}), tool_call_id="c1", name="save_tasks")

        events = _processor().process((), "updates", _tools_update(message, tasks=[]))

        assert _types(events) == [EventType.TOOL_RESULT]

    def test_a_plan_alone_reports_no_progress(self):
        message = ToolMessage(json.dumps({"success": True}), tool_call_id="c1", name="save_plan")

        events = _processor().process((), "updates", _tools_update(message, plan="- [ ] a"))

        assert _types(events) == [EventType.TOOL_RESULT]


class TestUsage:
    def test_every_token_key_is_mapped(self):
        message = AIMessage("Hi.", usage_metadata={
            "input_tokens": 1200,
            "output_tokens": 80,
            "total_tokens": 1280,
            "input_token_details": {"cache_read": 1000, "cache_creation": 150},
            "output_token_details": {"reasoning": 30},
        })

        (usage,) = [
            e for e in _processor().process((), "updates", _model_update(message))
            if e.type is EventType.LLM_USAGE
        ]

        assert usage.metadata == {
            "model": "anthropic:claude-sonnet-4-5",
            "prompt_tokens": 1200,
            "completion_tokens": 80,
            "total_tokens": 1280,
            "thinking_tokens": 30,
            "cached_tokens": 1000,
            "cache_creation_tokens": 150,
            "agent": "coordinator",
        }

    def test_cache_writes_broken_down_by_ttl_are_counted(self):
        # langchain-anthropic's mapping of the API's per-TTL breakdown.
        message = AIMessage("Hi.", usage_metadata={
            "input_tokens": 1550,
            "output_tokens": 7,
            "total_tokens": 1557,
            "input_token_details": {
                "cache_read": 1200, "cache_creation": 0,
                "ephemeral_5m_input_tokens": 250, "ephemeral_1h_input_tokens": 50,
            },
        })

        (usage,) = [
            e for e in _processor().process((), "updates", _model_update(message))
            if e.type is EventType.LLM_USAGE
        ]

        assert (usage.metadata["cached_tokens"], usage.metadata["cache_creation_tokens"]) == (
            1200, 300
        )

    def test_a_sub_agents_usage_is_on_its_own_model(self):
        message = AIMessage(
            "Found.", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        )

        (usage,) = [
            e for e in _run_processor().process((), "updates", _model_update(message))
            if e.type is EventType.LLM_USAGE
        ]

        assert usage.metadata["model"] == "google:gemini-2.5-flash"
        assert usage.metadata["agent"] == "researcher"
        assert "cached_tokens" not in usage.metadata

    def test_a_sub_agents_usage_is_not_the_conversations_context(self):
        message = AIMessage(
            "Found.", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        )

        (sub,) = [e for e in _run_processor().process((), "updates", _model_update(message))
                  if e.type is EventType.LLM_USAGE]
        (root,) = [e for e in _processor().process((), "updates", _model_update(message))
                   if e.type is EventType.LLM_USAGE]

        assert sub.metadata["purpose"] == "subagent"
        assert "purpose" not in root.metadata

    def test_a_message_without_usage_reports_none(self):
        events = _processor().process((), "updates", _model_update(AIMessage("Hi.")))

        assert EventType.LLM_USAGE not in _types(events)

    def test_each_agents_usage_carries_its_own_context_trigger(self):
        message = AIMessage(
            "Found.", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        )

        (sub,) = [e for e in _run_processor(budgets=BUDGETS).process(
            (), "updates", _model_update(message)) if e.type is EventType.LLM_USAGE]
        (root,) = [e for e in _processor(budgets=BUDGETS).process(
            (), "updates", _model_update(message)) if e.type is EventType.LLM_USAGE]

        assert {key: root.metadata.get(key) for key in CONTEXT_KEYS} == {
            "agent": "coordinator", "context_trigger": 154_536, "window_known": True,
        }
        assert {key: sub.metadata.get(key) for key in CONTEXT_KEYS} == {
            "agent": "researcher", "context_trigger": 103_360, "window_known": False,
        }


class TestError:
    def test_a_failed_turn_is_one_error_the_agents(self):
        error = RuntimeError("Error code: 400 - bad request")

        event = _processor().turn_failed(error)

        assert event.type is EventType.ERROR
        assert event.content == "Error code: 400 - bad request"
        assert event.metadata["error_code"] == "RuntimeError"
        assert event.metadata["recoverable"] is False
        assert event.metadata["agent"] == "coordinator"

    def test_without_a_message_it_is_named_by_its_type(self):
        event = _processor().turn_failed(TimeoutError())

        assert event.content == "TimeoutError"
        assert event.metadata["agent"] == "coordinator"

    def test_an_overflow_names_the_agents_own_model(self):
        """Deep Agents' own error when a summary still does not fit."""
        error = ContextOverflowError(
            "Context remains above the input budget after compaction; reduce input, "
            "tools, or configured output tokens."
        )

        event = _run_processor().turn_failed(error)

        assert event.type is EventType.ERROR
        assert event.content == overflow_message("google:gemini-2.5-flash")
        assert event.metadata["error_code"] == OVERFLOW_ERROR_CODE
        assert event.metadata["recoverable"] is False
        assert event.metadata["agent"] == "researcher"
        assert event.metadata["error_type"] == "ContextOverflowError"
        assert event.metadata["rate_limited"] is False

    def test_a_providers_overflow_at_the_root(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        error = anthropic.BadRequestError(
            "prompt is too long: 210000 tokens > 200000 maximum",
            response=httpx.Response(400, request=request),
            body={"type": "error", "error": {"type": "invalid_request_error",
                                             "message": "prompt is too long"}},
        )

        event = _processor().turn_failed(error)

        assert event.content == overflow_message("anthropic:claude-sonnet-4-5")
        assert event.metadata["error_code"] == OVERFLOW_ERROR_CODE
        assert event.metadata["agent"] == "coordinator"


class TestAttribution:
    def test_a_sub_agents_events_are_its_runs(self):
        events = _run_processor().process(
            (), "updates", _model_update(AIMessage("", tool_calls=[_call("grep", {}, "r1")]))
        )

        assert {(e.metadata["agent"], e.run) for e in events} == {("researcher", RESEARCHER)}

    def test_an_item_of_a_nested_graph_is_none_of_its_events(self):
        """Only its own graph's items (the root namespace) are its events: a
        graph a tool runs under the config it inherits streams in a
        namespace of its own, and its model's text is no agent's reply."""
        processor = _processor()

        assert processor.process(
            ("tools:unseen",), "updates", _model_update(AIMessage("Hi."))
        ) == []
        (said,) = processor.process((), "updates", _model_update(AIMessage("Hi.")))
        assert (said.type, said.metadata["agent"]) == (EventType.TEXT, "coordinator")

    def test_other_nodes_emit_nothing(self):
        """A middleware hook's update (here, a patched history) is not output."""
        patched = {"messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage("Hi."), AIMessage("Old answer."),
        ]}

        processor = _processor()
        assert processor.process((), "updates", {"PatchToolCallsMiddleware.before_agent": patched}) == []
        assert processor.process((), "updates", {"PatchToolCallsMiddleware.before_agent": None}) == []
        assert processor.process((), "custom", {"anything": 1}) == []
        assert processor.process((), "tasks", {"id": "t1", "name": "tools"}) == []


async def _reported(live: LiveEvents, act) -> list[WorkflowEvent]:
    """What ``act`` (an async callable) reports through ``live``, as a turn merges it."""

    async def source():
        await act()
        if False:  # pragma: no cover - a source that yields nothing itself
            yield

    return [item async for item in merged(source(), live)]


class TestRunCallback:
    async def test_a_summary_call_reports_that_it_started(self):
        live = LiveEvents()
        callback = RunCallback(root="coordinator", agent_models=MODELS, live=live)

        async def act():
            await callback.on_chat_model_start(
                {}, [[HumanMessage("Summarize.")]], run_id=uuid.uuid4(),
                metadata={"lc_source": "summarization", "lc_agent_name": "researcher"},
            )
            await callback.on_chat_model_start(  # an agent's own step reports nothing
                {}, [[HumanMessage("Hi.")]], run_id=uuid.uuid4(),
                metadata={"lc_agent_name": "researcher"},
            )

        (event,) = await _reported(live, act)
        assert event.type is EventType.CONTEXT_SUMMARIZING
        assert event.content == "Summarizing the earlier conversation…"
        assert event.metadata["agent"] == "researcher"

    async def test_everything_goes_out_through_the_live_events(self):
        live = LiveEvents()
        callback = RunCallback(root="coordinator", agent_models=MODELS, live=live)
        assert isinstance(callback, AsyncCallbackHandler)
        run = uuid.uuid4()

        async def source():
            await callback.on_chat_model_start(
                {}, [[HumanMessage("Summarize.")]], run_id=run,
                metadata={"lc_source": "summarization", "lc_agent_name": "coordinator"},
            )
            yield "the stream's next item"

        items = [item async for item in merged(source(), live)]

        assert [getattr(item, "type", item) for item in items] == [
            EventType.CONTEXT_SUMMARIZING, "the stream's next item"
        ]

    async def test_a_summarys_usage_is_the_summarized_agents(self):
        live = LiveEvents()
        callback = RunCallback(
            root="coordinator", agent_models=MODELS, live=live, budgets=BUDGETS
        )
        run = uuid.uuid4()
        answer = AIMessage("A summary.", usage_metadata={
            "input_tokens": 900, "output_tokens": 40, "total_tokens": 940,
        })

        async def act():
            await callback.on_chat_model_start(
                {}, [[HumanMessage("Summarize.")]], run_id=run,
                metadata={"lc_source": "summarization", "lc_agent_name": "researcher"},
            )
            await callback.on_llm_end(
                LLMResult(generations=[[ChatGeneration(message=answer)]]), run_id=run
            )

        _, usage = await _reported(live, act)
        assert usage.type is EventType.LLM_USAGE
        assert {key: usage.metadata.get(key) for key in ("purpose", "model", *CONTEXT_KEYS)} == {
            "purpose": "summarization", "model": "google:gemini-2.5-flash",
            "agent": "researcher", "context_trigger": 103_360, "window_known": False,
        }

    async def test_a_summary_reports_its_tokens(self):
        live = LiveEvents()
        callback = RunCallback(root="coordinator", agent_models=MODELS, live=live)

        async def act():
            await callback.on_custom_event(
                "agentic_cli.context_summarized",
                {"agent": "researcher", "messages_before": 11, "messages_after": 2,
                 "messages_summarized": 10, "tokens_before": 18_000, "tokens_after": 3_100},
                run_id=uuid.uuid4(),
            )

        (event,) = await _reported(live, act)
        assert event.type is EventType.CONTEXT_TRIMMED
        assert {key: event.metadata[key] for key in (
            "source", "agent", "messages_summarized", "tokens_before", "tokens_after"
        )} == {"source": "summarization", "agent": "researcher", "messages_summarized": 10,
               "tokens_before": 18_000, "tokens_after": 3_100}


@register_tool(capabilities=EXEMPT, description="Run a graph of its own (a processor test tool).")
async def processor_probe_nested_graph() -> dict:
    """Run a LangGraph graph whose "model" node answers, under the config the
    tool inherits: its updates reach the turn's stream as a subgraph's."""
    from langgraph.graph import START, MessagesState, StateGraph

    graph = StateGraph(MessagesState)
    graph.add_node("model", lambda state: {"messages": [AIMessage("The nested graph's text.")]})
    graph.add_edge(START, "model")
    await graph.compile().ainvoke({"messages": [HumanMessage("Go.")]})
    return {"success": True}


class TestARealTurn:
    """A real manager on scripted models: the turn's events end to end, the
    sub-agent's run among them, as its own stream reports them."""

    async def test_a_graph_a_tool_runs_is_none_of_the_turns_text(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
        from agentic_cli.config import BaseSettings
        from agentic_cli.workflow.deepagents import manager as manager_module

        models = ScriptedModels(
            solo=[ai(calls=[("processor_probe_nested_graph", {}, "c1")]), ai("Done.")]
        )
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        manager = manager_module.DeepAgentsWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="p",
                                       tools=[processor_probe_nested_graph],
                                       model="anthropic:claude-sonnet-4-5")],
            settings=BaseSettings(session_store="memory", default_model="anthropic:claude-sonnet-4-5"),
        )
        try:
            events = [e async for e in manager.process("Go.", "u", session_id="s1")]
        finally:
            await manager.cleanup()

        assert [e.content for e in events if e.type is EventType.TEXT] == ["Done."]
        assert [e.metadata.get("tool_name") for e in events
                if e.type in (EventType.TOOL_CALL, EventType.TOOL_RESULT)] == [
            "processor_probe_nested_graph", "processor_probe_nested_graph",
        ]

    async def test_a_delegating_turn(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
        from agentic_cli.config import BaseSettings
        from agentic_cli.workflow.deepagents import manager as manager_module

        models = ScriptedModels(
            coordinator=[
                ai(calls=[("save_plan", {"content": "- [ ] find X"}, "c0")]),
                ai("Delegating.", calls=[task("researcher", "Find X.", "c1")]),
                ai("X is 42."),
            ],
            researcher=[
                ai(calls=[("save_tasks", {"tasks": [
                    {"description": "find X", "status": "in_progress"}
                ]}, "r1")], reasoning="Plan first."),
                ai("It is 42."),
            ],
        )
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        manager = manager_module.DeepAgentsWorkflowManager(
            agent_configs=[
                AgentConfig(name="coordinator", prompt="p", sub_agents=["researcher"],
                            model="anthropic:claude-sonnet-4-5"),
                AgentConfig(name="researcher", prompt="p", description="Finds facts.",
                            model="anthropic:claude-haiku-4-5"),
            ],
            settings=BaseSettings(session_store="memory", default_model="anthropic:claude-sonnet-4-5"),
        )
        try:
            events = [e async for e in manager.process("What is X?", "u", session_id="s1")]
        finally:
            await manager.cleanup()

        seen = [
            (e.type.value, e.metadata["agent"], e.metadata.get("tool_name") or e.content)
            for e in events
            if e.type is not EventType.LLM_USAGE
        ]
        progress = task_progress_data(
            [{"id": "x", "description": "find X", "status": "in_progress"}]
        )
        assert seen == [
            ("tool_call", "coordinator", "save_plan"),
            ("tool_result", "coordinator", "save_plan"),
            ("text", "coordinator", "Delegating."),
            ("tool_call", "coordinator", "task"),
            ("agent_run", "researcher", "researcher-1 started"),
            ("thinking", "researcher", "Plan first."),
            ("tool_call", "researcher", "save_tasks"),
            ("tool_result", "researcher", "save_tasks"),
            ("text", "researcher", "It is 42."),
            ("agent_run", "researcher", "researcher-1 completed"),
            ("tool_result", "coordinator", "task"),
            ("task_progress", "coordinator", progress["display"]),
            ("text", "coordinator", "X is 42."),
        ]
        assert [(e.metadata["agent"], e.metadata.get("subagent")) for e in events
                if e.type is EventType.TEXT] == [
            ("coordinator", None), ("researcher", True), ("coordinator", None),
        ]
        usage = [(e.metadata["agent"], e.metadata["model"]) for e in events
                 if e.type is EventType.LLM_USAGE]
        assert usage == [
            ("coordinator", "anthropic:claude-sonnet-4-5"),
            ("coordinator", "anthropic:claude-sonnet-4-5"),
            ("researcher", "anthropic:claude-haiku-4-5"),
            ("researcher", "anthropic:claude-haiku-4-5"),
            ("coordinator", "anthropic:claude-sonnet-4-5"),
        ]
        task_result = next(
            e for e in events
            if e.type is EventType.TOOL_RESULT and e.metadata["tool_name"] == "task"
        )
        assert task_result.metadata["result"]["result"] == "It is 42."
        assert all("Command(" not in e.content for e in events)
