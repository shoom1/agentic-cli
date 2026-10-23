"""The event stream (``workflow/deepagents/event_processor.py``).

A turn streams ``astream(..., stream_mode=["updates", "tasks"],
subgraphs=True)``: ``(namespace, mode, payload)`` items, ``()`` being the
root. ``DeepAgentsEventProcessor`` turns each finished node output into
``WorkflowEvent``s, one row of the spec's mapping per test class, and
attributes every event to its agent through the ``tasks`` mode, which maps a
sub-agent's ``tools:<task-id>`` namespace to the ``task`` call that started
it.

Most tests feed the processor payloads shaped as the stream sends them; the
last runs a real agent tree on scripted models and feeds it the real stream.
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
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph.message import REMOVE_ALL_MESSAGES  # noqa: E402

from agentic_cli.models import ModelCatalog, RoleMap, resolve_model  # noqa: E402
from agentic_cli.tools._core.tasks import task_progress_data  # noqa: E402
from agentic_cli.tools.langgraph.state_tools import save_plan, save_tasks  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import (  # noqa: E402
    OVERFLOW_ERROR_CODE,
    ContextBudget,
    overflow_message,
)
from agentic_cli.workflow.deepagents.agent_builder import build_agent_tree  # noqa: E402
from agentic_cli.workflow.deepagents.event_processor import (  # noqa: E402
    DeepAgentsEventProcessor,
    RunCallback,
)
from agentic_cli.workflow.deepagents.tools import to_langchain_tools  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402
from agentic_cli.workflow.live_events import LiveEvents, merged  # noqa: E402
from tests.models.fakes import endpoints  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat, ai, task  # noqa: E402

_ENDPOINTS = endpoints(anthropic="a-test-key", google="g-test-key", openai="o-test-key")


def _model(agent: str, spec: str):
    """``spec`` resolved for ``agent`` against the shipped table."""
    return resolve_model(
        agent=agent,
        spec=spec,
        generation=None,
        thinking_effort="none",
        roles=RoleMap(default_model=None, model_roles={}),
        catalog=ModelCatalog(endpoints=_ENDPOINTS),
        endpoints=_ENDPOINTS,
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

SUB = "tools:4f2c"  # a sub-agent's namespace segment: its task's id
NESTED = "tools:9a1e"


def _processor(**kwargs) -> DeepAgentsEventProcessor:
    return DeepAgentsEventProcessor(root="coordinator", agent_models=MODELS, **kwargs)


def _task_started(task_id: str, call: dict, name: str = "tools") -> dict:
    """A ``tasks``-mode start payload: one tool call sent to the tools node."""
    return {"id": task_id, "name": name, "input": [call], "triggers": ("__pregel_push",)}


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _delegated(processor, namespace: tuple, segment: str, subagent: str) -> None:
    """Tell the processor that ``segment`` under ``namespace`` runs ``subagent``."""
    call = _call("task", {"description": "brief", "subagent_type": subagent}, "c-" + segment)
    processor.process(namespace, "tasks", _task_started(segment.removeprefix("tools:"), call))


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
        processor = _processor()
        _delegated(processor, (), SUB, "researcher")

        events = processor.process((SUB,), "updates", _model_update(AIMessage("It is 42.")))

        (event,) = [e for e in events if e.type is EventType.TEXT]
        assert (event.metadata["agent"], event.metadata["subagent"]) == ("researcher", True)


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
        message = AIMessage("", tool_calls=[
            _call("task", {"description": "Find X.", "subagent_type": "researcher"}, "c1"),
        ])

        (event,) = [
            e for e in _processor().process((), "updates", _model_update(message))
            if e.type is EventType.TOOL_CALL
        ]

        assert event.metadata["tool_name"] == "task"
        assert event.metadata["tool_args"] == {
            "description": "Find X.", "subagent_type": "researcher"
        }

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
            AIMessage("", tool_calls=[_call("task", {"description": "d", "subagent_type": "researcher"}, "c1")])
        ))

        (event,) = processor.process(
            (), "updates", _tools_update(ToolMessage("Done.", tool_call_id="c1"))
        )

        assert event.metadata["tool_name"] == "task"

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

    def test_a_task_command_is_the_sub_agents_answer(self):
        tasks = [{"id": "t1", "description": "find X", "status": "completed"}]

        events = _processor().process((), "updates", _tools_update(
            ToolMessage("X is 42.", tool_call_id="c1", name="task"), tasks=tasks
        ))

        assert _types(events) == [EventType.TOOL_RESULT, EventType.TASK_PROGRESS]
        assert events[0].metadata["result"] == "X is 42."

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

    def test_the_same_list_again_reports_nothing_new(self):
        """A sub-agent's list flows back with its result: one report, not two."""
        processor = _processor()
        _delegated(processor, (), SUB, "researcher")
        message = ToolMessage(json.dumps({"success": True}), tool_call_id="r1", name="save_tasks")

        first = processor.process((SUB,), "updates", _tools_update(message, tasks=self.TASKS))
        back = processor.process((), "updates", _tools_update(
            ToolMessage("Done.", tool_call_id="c1", name="task"), tasks=self.TASKS
        ))

        assert _types(first) == [EventType.TOOL_RESULT, EventType.TASK_PROGRESS]
        assert first[1].metadata["agent"] == "researcher"
        assert _types(back) == [EventType.TOOL_RESULT]

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
        processor = _processor()
        _delegated(processor, (), SUB, "researcher")
        message = AIMessage(
            "Found.", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        )

        (usage,) = [
            e for e in processor.process((SUB,), "updates", _model_update(message))
            if e.type is EventType.LLM_USAGE
        ]

        assert usage.metadata["model"] == "google:gemini-2.5-flash"
        assert usage.metadata["agent"] == "researcher"
        assert "cached_tokens" not in usage.metadata

    def test_a_sub_agents_usage_is_not_the_conversations_context(self):
        processor = _processor()
        _delegated(processor, (), SUB, "researcher")
        message = AIMessage(
            "Found.", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        )

        (sub,) = [e for e in processor.process((SUB,), "updates", _model_update(message))
                  if e.type is EventType.LLM_USAGE]
        (root,) = [e for e in processor.process((), "updates", _model_update(message))
                   if e.type is EventType.LLM_USAGE]

        assert sub.metadata["purpose"] == "subagent"
        assert "purpose" not in root.metadata

    def test_a_message_without_usage_reports_none(self):
        events = _processor().process((), "updates", _model_update(AIMessage("Hi.")))

        assert EventType.LLM_USAGE not in _types(events)

    def test_each_agents_usage_carries_its_own_context_trigger(self):
        processor = _processor(budgets=BUDGETS)
        _delegated(processor, (), SUB, "researcher")
        message = AIMessage(
            "Found.", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        )

        (sub,) = [e for e in processor.process((SUB,), "updates", _model_update(message))
                  if e.type is EventType.LLM_USAGE]
        (root,) = [e for e in processor.process((), "updates", _model_update(message))
                   if e.type is EventType.LLM_USAGE]

        assert {key: root.metadata.get(key) for key in CONTEXT_KEYS} == {
            "agent": "coordinator", "context_trigger": 154_536, "window_known": True,
        }
        assert {key: sub.metadata.get(key) for key in CONTEXT_KEYS} == {
            "agent": "researcher", "context_trigger": 103_360, "window_known": False,
        }


class TestError:
    def test_a_failed_turn_is_one_error_from_the_agent_that_failed(self):
        processor = _processor()
        _delegated(processor, (), SUB, "researcher")
        error = RuntimeError("Error code: 400 - bad request")
        # The sub-agent's model task fails first, then the coordinator's tools task.
        processor.process((SUB,), "tasks", {"id": "m1", "name": "model", "error": error,
                                            "interrupts": [], "result": None})
        processor.process((), "tasks", {"id": SUB.removeprefix("tools:"), "name": "tools",
                                        "error": error, "interrupts": [], "result": None})

        event = processor.turn_failed(error)

        assert event.type is EventType.ERROR
        assert event.content == "Error code: 400 - bad request"
        assert event.metadata["error_code"] == "RuntimeError"
        assert event.metadata["recoverable"] is False
        assert event.metadata["agent"] == "researcher"

    def test_without_a_failed_task_it_is_the_roots(self):
        event = _processor().turn_failed(TimeoutError())

        assert event.content == "TimeoutError"
        assert event.metadata["agent"] == "coordinator"

    def test_an_overflow_names_the_failing_agents_model(self):
        # Deep Agents' own error when a summary still does not fit.
        processor = _processor()
        _delegated(processor, (), SUB, "researcher")
        error = ContextOverflowError(
            "Context remains above the input budget after compaction; reduce input, "
            "tools, or configured output tokens."
        )
        processor.process((SUB,), "tasks", {"id": "m1", "name": "model", "error": error,
                                            "interrupts": [], "result": None})

        event = processor.turn_failed(error)

        assert event.type is EventType.ERROR
        assert event.content == overflow_message("google:gemini-2.5-flash")
        assert event.metadata["error_code"] == OVERFLOW_ERROR_CODE
        assert event.metadata["recoverable"] is False
        assert event.metadata["agent"] == "researcher"

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
    def test_a_sub_agents_events_are_its_own(self):
        processor = _processor()
        _delegated(processor, (), SUB, "researcher")

        events = processor.process(
            (SUB,), "updates", _model_update(AIMessage("", tool_calls=[_call("grep", {}, "r1")]))
        )

        assert {e.metadata["agent"] for e in events} == {"researcher"}

    def test_nested_namespaces_resolve_through_the_chain(self):
        processor = _processor()
        _delegated(processor, (), SUB, "researcher")
        _delegated(processor, (SUB,), NESTED, "worker")

        message = AIMessage(
            "Done.", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        )

        events = processor.process((SUB, NESTED), "updates", _model_update(message))

        assert {e.metadata["agent"] for e in events} == {"worker"}
        assert next(e for e in events if e.type is EventType.LLM_USAGE).metadata["model"] == (
            "openai:gpt-5"
        )

    def test_an_unknown_namespace_falls_back_to_the_root(self):
        events = _processor().process(("tools:unseen",), "updates", _model_update(AIMessage("Hi.")))

        assert {e.metadata["agent"] for e in events} == {"coordinator"}

    def test_a_tools_task_that_is_not_delegation_maps_nothing(self):
        """Only ``task`` delegates, whatever another tool's arguments say."""
        processor = _processor()
        lookalike = _call("assign", {"subagent_type": "researcher"}, "c1")
        processor.process((), "tasks", _task_started("ab12", lookalike))

        events = processor.process(("tools:ab12",), "updates", _model_update(AIMessage("Hi.")))

        assert {e.metadata["agent"] for e in events} == {"coordinator"}

    def test_other_nodes_emit_nothing(self):
        """A middleware hook's update (here, a patched history) is not output."""
        patched = {"messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage("Hi."), AIMessage("Old answer."),
        ]}

        processor = _processor()
        assert processor.process((), "updates", {"PatchToolCallsMiddleware.before_agent": patched}) == []
        assert processor.process((), "updates", {"PatchToolCallsMiddleware.before_agent": None}) == []
        assert processor.process((), "custom", {"anything": 1}) == []


class TestRunCallback:
    def test_drain_hands_over_what_was_collected_once(self):
        callback = RunCallback(root="coordinator", agent_models=MODELS)
        first, second = WorkflowEvent.text("a"), WorkflowEvent.text("b")

        callback._emit(first)
        callback._emit(second)

        assert isinstance(callback, AsyncCallbackHandler)
        assert callback.drain() == [first, second]
        assert callback.drain() == []

    async def test_a_summary_call_reports_that_it_started(self):
        callback = RunCallback(root="coordinator", agent_models=MODELS)

        await callback.on_chat_model_start(
            {}, [[HumanMessage("Summarize.")]], run_id=uuid.uuid4(),
            metadata={"lc_source": "summarization", "lc_agent_name": "researcher"},
        )
        await callback.on_chat_model_start(  # an agent's own step reports nothing
            {}, [[HumanMessage("Hi.")]], run_id=uuid.uuid4(),
            metadata={"lc_agent_name": "researcher"},
        )

        (event,) = callback.drain()
        assert event.type is EventType.CONTEXT_SUMMARIZING
        assert event.content == "Summarizing the earlier conversation…"
        assert event.metadata["agent"] == "researcher"

    async def test_with_live_events_everything_goes_out_through_them(self):
        live = LiveEvents()
        callback = RunCallback(root="coordinator", agent_models=MODELS, live=live)
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
        assert callback.drain() == []

    async def test_a_summarys_usage_is_the_summarized_agents(self):
        callback = RunCallback(root="coordinator", agent_models=MODELS, budgets=BUDGETS)
        run = uuid.uuid4()
        answer = AIMessage("A summary.", usage_metadata={
            "input_tokens": 900, "output_tokens": 40, "total_tokens": 940,
        })

        await callback.on_chat_model_start(
            {}, [[HumanMessage("Summarize.")]], run_id=run,
            metadata={"lc_source": "summarization", "lc_agent_name": "researcher"},
        )
        await callback.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=answer)]]), run_id=run
        )

        _, usage = callback.drain()
        assert usage.type is EventType.LLM_USAGE
        assert {key: usage.metadata.get(key) for key in ("purpose", "model", *CONTEXT_KEYS)} == {
            "purpose": "summarization", "model": "google:gemini-2.5-flash",
            "agent": "researcher", "context_trigger": 103_360, "window_known": False,
        }

    async def test_a_summary_reports_its_tokens(self):
        callback = RunCallback(root="coordinator", agent_models=MODELS)

        await callback.on_custom_event(
            "agentic_cli.context_summarized",
            {"agent": "researcher", "messages_before": 11, "messages_after": 2,
             "messages_summarized": 10, "tokens_before": 18_000, "tokens_after": 3_100},
            run_id=uuid.uuid4(),
        )

        (event,) = callback.drain()
        assert event.type is EventType.CONTEXT_TRIMMED
        assert {key: event.metadata[key] for key in (
            "source", "agent", "messages_summarized", "tokens_before", "tokens_after"
        )} == {"source": "summarization", "agent": "researcher", "messages_summarized": 10,
               "tokens_before": 18_000, "tokens_after": 3_100}


class TestARealStream:
    """A real tree on scripted models: the stream's own shapes, end to end."""

    async def test_a_delegating_turn(self):
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["researcher"]),
            AgentConfig(name="researcher", prompt="p", description="Finds facts."),
        ]
        models = {
            "coordinator": ScriptedChat(label="coordinator", script=[
                ai(calls=[("save_plan", {"content": "- [ ] find X"}, "c0")]),
                ai("Delegating.", calls=[task("researcher", "Find X.", "c1")]),
                ai("X is 42."),
            ]),
            "researcher": ScriptedChat(label="researcher", script=[
                ai(calls=[("save_tasks", {"tasks": [
                    {"description": "find X", "status": "in_progress"}
                ]}, "r1")], reasoning="Plan first."),
                ai("It is 42."),
            ]),
        }
        root = build_agent_tree(
            configs,
            chat_models=models,
            resolved=MODELS,
            tools={
                "coordinator": to_langchain_tools([save_plan]),
                "researcher": to_langchain_tools([save_tasks]),
            },
            prompts={"coordinator": "Coordinate.", "researcher": "Research."},
            settings=None,
            checkpointer=InMemorySaver(),
        )
        processor = _processor(session_id="s1")
        events: list[WorkflowEvent] = []
        async for namespace, mode, payload in root.astream(
            {"messages": [HumanMessage("What is X?")]},
            {"configurable": {"thread_id": "t"}},
            stream_mode=["updates", "tasks"],
            subgraphs=True,
        ):
            events += processor.process(namespace, mode, payload)

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
            ("thinking", "researcher", "Plan first."),
            ("tool_call", "researcher", "save_tasks"),
            ("tool_result", "researcher", "save_tasks"),
            ("task_progress", "researcher", progress["display"]),
            ("text", "researcher", "It is 42."),
            ("tool_result", "coordinator", "task"),
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
            ("researcher", "google:gemini-2.5-flash"),
            ("researcher", "google:gemini-2.5-flash"),
            ("coordinator", "anthropic:claude-sonnet-4-5"),
        ]
        task_result = next(
            e for e in events
            if e.type is EventType.TOOL_RESULT and e.metadata["tool_name"] == "task"
        )
        assert task_result.metadata["result"] == "It is 42."
        assert all("Command(" not in e.content for e in events)
