"""Deep Agents: a sub-agent's refused or cut answer, said so (``SubAgentOutcomeMiddleware``).

Each LangChain adapter reports how an answer stopped in its
``response_metadata``: ChatAnthropic's ``stop_reason`` (``refusal``,
``max_tokens``), ChatOpenAI's ``finish_reason`` (``content_filter``,
``length``) or the Responses API's ``incomplete_details``, and
ChatGoogleGenerativeAI's ``finish_reason`` (``SAFETY``, ``MAX_TOKENS``);
ChatOpenAI also gives a refusal its own block. The middleware, on every
sub-agent's model side and never the root's, rewrites a refused or cut final
answer and marks it. The whole turn, on both backends, is in the contract
suite.
"""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("deepagents")

from google.genai import types  # noqa: E402
from langchain.agents.middleware import ExtendedModelResponse, ModelResponse  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.agents.outcome import CUT, REFUSED, refusal_notice  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as deepagents_manager  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import (  # noqa: E402
    ActiveToolCallMiddleware,
    PermissionMiddleware,
    SubAgentOutcomeMiddleware,
    ToolCallErrorMiddleware,
    answer_outcome,
    marked_answer,
)
from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat, ScriptedModels, ai, task  # noqa: E402
from tests.workflow.deepagents.test_agent_builder import _tree, created  # noqa: E402,F401

CUT_TEXT = "X is\n\n[cut off at the output limit]"


def stopped(content="", **metadata) -> AIMessage:
    return AIMessage(content=content, response_metadata=metadata)


def blocks(message: AIMessage) -> list[tuple[str, str | None]]:
    """``message``'s standard blocks as (type, reasoning or text) pairs."""
    return [(b["type"], b.get("reasoning") or b.get("text")) for b in message.content_blocks]


def _refusal_in(message: object) -> dict | None:
    """deepagents 0.7.34's check (``middleware/subagents.py``), copied: the
    refusal block a ``task`` result is reworded from."""
    if not isinstance(message, BaseMessage):
        return None
    return next((b for b in message.content_blocks if b["type"] == "refusal"), None)


class TestMarkedAnswer:
    @pytest.mark.parametrize("metadata", [
        {"stop_reason": "refusal"},                         # ChatAnthropic
        {"finish_reason": "content_filter"},                # ChatOpenAI, OpenRouter
        {"finish_reason": "SAFETY"},                        # ChatGoogleGenerativeAI
        {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
    ])
    def test_a_refusal_becomes_the_notice(self, metadata):
        marked = marked_answer(stopped(**metadata))

        assert marked.text == refusal_notice()
        assert answer_outcome(marked) == REFUSED
        assert marked.response_metadata == {**metadata, "agentic_cli": {"outcome": REFUSED}}

    @pytest.mark.parametrize("refused", [
        # Chat Completions: the refusal beside empty content.
        AIMessage(content="", additional_kwargs={"refusal": "I can't help with that."},
                  response_metadata={"finish_reason": "stop", "model_provider": "openai"}),
        # The Responses API: a refusal block (non_standard before langchain-core 1.7).
        AIMessage(content=[{"type": "refusal", "refusal": "I can't help with that."}],
                  response_metadata={"status": "completed", "model_provider": "openai"}),
    ])
    def test_an_openai_refusal_is_the_explanation_and_leaves_none(self, refused):
        marked = marked_answer(refused)

        assert marked.text == refusal_notice(explanation="I can't help with that.")
        # Deep Agents 0.7.34+ finds no refusal to word again: one notice.
        assert _refusal_in(marked) is None
        assert [b["type"] for b in marked.content_blocks] == ["text"]
        assert "refusal" not in marked.additional_kwargs

    def test_a_streamed_gemini_prompt_block_becomes_the_notice(self):
        """Streamed, langchain-google-genai gives a blocked prompt an empty
        answer whose metadata holds only the prompt's feedback."""
        feedback = types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.PROHIBITED_CONTENT, block_reason_message="Blocked.",
        )

        marked = marked_answer(stopped(prompt_feedback=feedback.model_dump()))

        assert marked.text == refusal_notice()
        assert answer_outcome(marked) == REFUSED

    async def test_an_unstreamed_gemini_prompt_block_becomes_the_notice(self):
        """Unstreamed, as Deep Agents calls its models, langchain-google-genai
        gives a blocked prompt an empty answer and the prompt's feedback as
        the call's ``llm_output``, which langchain-core puts in the answer's
        metadata."""
        genai = pytest.importorskip("langchain_google_genai")
        blocked = types.GenerateContentResponse(
            prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason=types.BlockedReason.PROHIBITED_CONTENT,
                block_reason_message="Blocked.",
            ),
        )
        model = genai.ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="a-test-key")

        async def generate_content(**request):
            return blocked

        with mock.patch.object(model.async_client.models, "generate_content", generate_content):
            answer = await model.ainvoke("Find X.")

        assert (answer.content, answer.tool_calls) == ("", [])
        marked = marked_answer(answer)
        assert marked.text == refusal_notice()
        assert answer_outcome(marked) == REFUSED

    def test_a_standard_refusal_block_gives_its_category(self):
        refused = AIMessage(content=[{
            "type": "refusal", "refusal": "No.", "extras": {"category": "cyber"},
        }])

        marked = marked_answer(refused)

        # langchain-core before 1.7 keeps the block non_standard, extras and all.
        assert marked.text == refusal_notice(category="cyber", explanation="No.")

    @pytest.mark.parametrize("metadata", [
        {"stop_reason": "max_tokens"},
        {"finish_reason": "length"},
        {"finish_reason": "MAX_TOKENS"},
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    ])
    def test_a_cut_keeps_its_text_then_the_note(self, metadata):
        marked = marked_answer(stopped([{"type": "text", "text": "X is"}], **metadata))

        assert marked.text == CUT_TEXT
        assert answer_outcome(marked) == CUT

    @pytest.mark.parametrize("reasoning, metadata", [
        ({"type": "reasoning", "reasoning": "Thinking."}, {}),
        ({"type": "thinking", "thinking": "Thinking.", "signature": "s"},
         {"model_provider": "anthropic"}),
    ])
    def test_a_cut_keeps_its_text_and_its_reasoning(self, reasoning, metadata):
        cut = AIMessage(
            content=[reasoning, {"type": "text", "text": "X "}, {"type": "text", "text": "is"}],
            response_metadata={**metadata, "stop_reason": "max_tokens"},
        )

        marked = marked_answer(cut)

        assert marked.content[0] == reasoning
        assert blocks(marked) == [
            ("reasoning", "Thinking."), ("text", CUT_TEXT),
        ]

    def test_a_refusal_keeps_its_reasoning_and_leaves_no_refusal(self):
        refused = AIMessage(content=[
            {"type": "reasoning", "reasoning": "Thinking."},
            {"type": "refusal", "refusal": "No."},
        ])

        marked = marked_answer(refused)

        assert blocks(marked) == [
            ("reasoning", "Thinking."), ("text", refusal_notice(explanation="No.")),
        ]
        assert _refusal_in(marked) is None

    def test_an_answer_that_calls_a_tool_is_not_final(self):
        calling = AIMessage(
            content="", tool_calls=[{"name": "lookup", "args": {}, "id": "r1"}],
            response_metadata={"stop_reason": "max_tokens"},
        )

        assert marked_answer(calling) is calling

    @pytest.mark.parametrize("metadata", [
        {"stop_reason": "end_turn"}, {"finish_reason": "stop"}, {},
        {"finish_reason": "STOP",
         "prompt_feedback": {"block_reason": None, "safety_ratings": []}},
    ])
    def test_any_other_answer_is_itself(self, metadata):
        answer = stopped("X is 42.", **metadata)

        assert marked_answer(answer) is answer
        assert answer_outcome(answer) is None

    def test_only_an_ai_message_has_an_outcome(self):
        assert answer_outcome(ToolMessage("x", tool_call_id="c1")) is None


class TestTheMiddleware:
    @pytest.mark.parametrize("wrap", [
        lambda message: message,
        lambda message: ModelResponse(result=[message]),
        lambda message: ExtendedModelResponse(model_response=ModelResponse(result=[message])),
    ])
    async def test_it_marks_whatever_form_the_model_call_returns(self, wrap):
        async def handler(request):
            return wrap(stopped([{"type": "text", "text": "X is"}], stop_reason="max_tokens"))

        returned = await SubAgentOutcomeMiddleware().awrap_model_call(None, handler)

        if isinstance(returned, ExtendedModelResponse):
            returned = returned.model_response
        message = returned.result[0] if isinstance(returned, ModelResponse) else returned
        assert message.text == CUT_TEXT

    async def test_a_model_call_that_raises_is_not_its_business(self):
        """A sub-agent's failure goes on to ``task``, untouched here."""

        async def handler(request):
            raise RuntimeError("overloaded")

        with pytest.raises(RuntimeError, match="overloaded"):
            await SubAgentOutcomeMiddleware().awrap_model_call(None, handler)


class TestTheStacks:
    def test_every_sub_agent_has_it_and_the_root_has_not(self, created):
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["lead"]),
            AgentConfig(name="lead", prompt="p", sub_agents=["worker"], description="l"),
            AgentConfig(name="worker", prompt="p", description="w"),
        ]

        _tree(configs, {config.name: ScriptedChat() for config in configs})

        def outcome(name: str) -> list[type]:
            return [type(m) for m in created[name]["middleware"]
                    if isinstance(m, SubAgentOutcomeMiddleware)]

        assert outcome("coordinator") == []
        assert outcome("lead") == outcome("worker") == [SubAgentOutcomeMiddleware]
        # Outermost on the model side: before the tool side.
        worker = [type(m) for m in created["worker"]["middleware"]]
        assert worker[-4:] == [
            SubAgentOutcomeMiddleware, PermissionMiddleware, ToolCallErrorMiddleware,
            ActiveToolCallMiddleware,
        ]


class TestTheRawLog:
    """With ``raw_llm_logging``, the log keeps a sub-agent's answer as the model
    gave it, and its parent reads the rewritten one: ``RunCallback.on_llm_end``
    fires inside the model call, as the chat model returns, before the call
    comes back out through ``SubAgentOutcomeMiddleware``."""

    @pytest.mark.parametrize("stop, text, read", [
        ("refusal", "", refusal_notice()),
        ("max_tokens", "X is", "X is\n\n[cut off at the output limit]"),
    ])
    async def test_the_log_has_the_answer_as_the_model_gave_it(
        self, tmp_path, monkeypatch, stop, text, read
    ):
        (tmp_path / "home").mkdir()
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
        answer = ai(text).model_copy(update={"response_metadata": {"stop_reason": stop}})
        models = ScriptedModels(
            coordinator=[ai(calls=[task("researcher", "Find X.", "c1")]), ai("Done.")],
            researcher=[answer],
        )
        monkeypatch.setattr(deepagents_manager, "build_chat_model", models)
        manager = deepagents_manager.DeepAgentsWorkflowManager(
            agent_configs=[
                AgentConfig(name="coordinator", prompt="p", sub_agents=["researcher"]),
                AgentConfig(name="researcher", prompt="p", description="Finds facts."),
            ],
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                                  raw_llm_logging=True),
        )
        try:
            events = [e async for e in manager.process("Find X.", "u", session_id="s")]
        finally:
            await manager.cleanup()

        (logged,) = [e for e in events if e.type is EventType.LLM_RESPONSE
                     and e.metadata.get("agent") == "researcher"]
        assert logged.content == f"LLM Response: {text}"
        assert logged.metadata["finish_reason"] == stop
        (result,) = [e for e in events if e.type is EventType.TOOL_RESULT
                     and e.metadata.get("tool_call_id") == "c1"]
        assert result.metadata["result"] == read
