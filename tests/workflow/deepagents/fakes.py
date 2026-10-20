"""A scripted chat model for Deep Agents unit tests.

``ScriptedChat`` answers each model call with the next scripted ``AIMessage``
and records what it was sent, so a test can drive a whole agent tree offline
and then read every agent's requests. It stands in for the provider models
``build_chat_model`` builds (``ScriptedModels`` stands in for the function
itself, for a manager); tests of what reaches a provider use
``tests/workflow/langchain_fakes.py`` and ``FakeAPI`` instead.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

#: The usage a scripted answer reports unless it carries its own.
USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

_call_ids = itertools.count(1)


def ai(
    text: str = "",
    calls: Sequence[tuple[str, dict] | tuple[str, dict, str]] = (),
    *,
    reasoning: str | None = None,
    usage: dict | None = None,
) -> AIMessage:
    """A scripted answer: text, tool calls ``(name, args[, id])`` and reasoning."""
    tool_calls = []
    for call in calls:
        name, args, *rest = call
        call_id = rest[0] if rest else f"call_{next(_call_ids)}"
        tool_calls.append({"name": name, "args": args, "id": call_id, "type": "tool_call"})
    content: Any = text
    if reasoning is not None:
        content = [{"type": "reasoning", "reasoning": reasoning}]
        if text:
            content.append({"type": "text", "text": text})
    return AIMessage(content=content, tool_calls=tool_calls, usage_metadata=usage or USAGE)


def task(subagent: str, brief: str, call_id: str | None = None) -> tuple:
    """A ``task`` call delegating ``brief`` to ``subagent``."""
    args = {"description": brief, "subagent_type": subagent}
    return ("task", args, call_id) if call_id else ("task", args)


class ScriptedChat(BaseChatModel):
    """Answers with ``script``, in order, and records every request.

    Attributes:
        label: Names the model in failures.
        script: The answers, one per model call. An exception in it is
            raised by that call instead, as a provider error would be.
        requests: The messages of each call, system message included.
        bound: The tool schemas of each ``bind_tools`` call.
    """

    label: str = "model"
    # AIMessages and exceptions (Any: pydantic would coerce a union to AIMessage).
    script: list[Any] = Field(default_factory=list)
    requests: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[list[dict]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChat":  # type: ignore[override]
        self.bound.append([convert_to_openai_tool(tool)["function"] for tool in tools])
        return self

    def _answer(self, messages: list[BaseMessage]) -> ChatResult:
        self.requests.append(list(messages))
        if len(self.requests) > len(self.script):
            raise AssertionError(
                f"{self.label}: no scripted answer for call {len(self.requests)}"
            )
        answer = self.script[len(self.requests) - 1]
        if isinstance(answer, Exception):
            raise answer
        return ChatResult(generations=[ChatGeneration(message=answer)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return self._answer(messages)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return self._answer(messages)

    # -- reading what it was sent -------------------------------------------

    def conversation(self, call: int) -> list[BaseMessage]:
        """Call ``call``'s messages without the system prompt."""
        return [m for m in self.requests[call] if not isinstance(m, SystemMessage)]

    def system_prompt(self, call: int) -> str | None:
        system = [m for m in self.requests[call] if isinstance(m, SystemMessage)]
        return system[0].text if system else None

    def tool_names(self, call: int = 0) -> list[str]:
        return [schema["name"] for schema in self.bound[call]]

    def tool_schema(self, name: str, call: int = 0) -> dict:
        return next(schema for schema in self.bound[call] if schema["name"] == name)


class ScriptedModels:
    """Stands in for ``build_chat_model``: each agent's ``ScriptedChat``, by name.

    Patch ``agentic_cli.workflow.deepagents.manager.build_chat_model`` with an
    instance, and a manager builds every agent on its scripted model. A
    rebuild (a model switch) gets the same instance back, so one script runs
    across the whole test.

    Attributes:
        chats: Each agent's model.
        built: The resolved model of every build, in order.
    """

    def __init__(self, **scripts: Sequence[AIMessage | Exception]) -> None:
        self.chats = {
            name: ScriptedChat(label=name, script=list(script))
            for name, script in scripts.items()
        }
        self.built: list[Any] = []

    def __call__(self, resolved: Any, *, settings: Any, app_name: str) -> ScriptedChat:
        self.built.append(resolved)
        return self.chats[resolved.agent]
