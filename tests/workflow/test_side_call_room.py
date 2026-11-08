"""The workflow's side calls never send more than their model takes.

``BaseWorkflowManager.summarize`` (the ``llm_summarizer`` service: web_fetch's
summaries, the knowledge base's summary and sidecar) sent its prompt to
``generate_simple`` however long it was, and session fact extraction sent the
last 20 messages whole: a long PDF or a long session was a context overflow,
or a very expensive call. A prompt over the room the default model leaves (95%
of its window, or of the 128k floor when the window is unknown, less what the
call may write, at the provider's characters per token with the estimates'
10% margin) is now cut in the middle, its start and its end kept. What the
call may write is the resolved ``max_tokens``: the resolver clamps the 12,000
a summary asks for to the model's max output, so a small model keeps its
room. A room too small to hold a useful prompt sends the prompt as it is,
for the provider to answer.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.memory import MemoryStore
from agentic_cli.models import ModelCatalog, ModelEntry, ModelRef, RoleMap, resolve_model
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.service_registry import MEMORY_STORE
from tests.models.fakes import providers as fake_providers


class _Manager(BaseWorkflowManager):
    """A manager whose ``generate_simple`` records each prompt."""

    @property
    def backend_type(self) -> str:
        return "test"

    async def _do_initialize(self) -> None:
        pass

    async def process(
        self, message: str, user_id: str, session_id: str | None = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        if False:
            yield  # type: ignore[misc]

    async def reinitialize(self, model: str | None = None, preserve_sessions: bool = True) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    def _get_state_tools(self) -> list:
        return []

    async def generate_simple(self, prompt: str, max_tokens: int = 500) -> str:
        self.sent.append((prompt, max_tokens))
        return self.answer


SONNET = ModelRef("anthropic", "claude-sonnet-4-5")


def _manager(tmp_path, window: int | None, max_output: int | None = None) -> _Manager:
    """A manager whose default model is Claude Sonnet 4.5 with ``window``
    tokens (and ``max_output``, else the table's), as a ``models`` entry
    describes it; None leaves it uninitialized, with no model yet."""
    manager = _Manager(
        agent_configs=[],
        settings=BaseSettings(workspace_dir=tmp_path, auto_extract_session_facts=True),
    )
    manager.sent = []
    manager.answer = "A fact."
    manager._services[MEMORY_STORE] = MemoryStore(tmp_path / "user")
    if window is not None:
        providers = fake_providers(anthropic="a-test-key")
        entry = ModelEntry(context_window=window, max_output=max_output)
        manager._catalog = ModelCatalog(providers=providers, entries={SONNET: entry})
        manager._default_model = resolve_model(
            agent=None, spec=str(SONNET), generation=None,
            thinking_effort="none", roles=RoleMap(default_model=None, model_roles={}),
            catalog=manager._catalog, providers=providers,
        )
    return manager


#: (window, characters per token): a 32,000-token Claude model (3.3), and no
#: model yet, counted as the 128k floor at the default 4.0.
MODELS = [(32_000, 3.3), (None, 4.0)]


def _room(window: int | None, ratio: float, max_tokens: int) -> int:
    return int(((window or 128_000) * 95 // 100 - max_tokens) * ratio / 1.1)


@pytest.mark.parametrize(("window", "ratio"), MODELS)
async def test_a_summary_prompt_over_the_window_is_cut_in_the_middle(tmp_path, window, ratio):
    manager = _manager(tmp_path, window)
    content = "".join(f"line {n} of the page\n" for n in range(200_000))
    prompt = f"Web page content:\n---\n{content}---\n\nWhat does it say?\n\nAnswer from the content."

    await manager.summarize(content, prompt)

    (sent, max_tokens), = manager.sent
    assert len(sent) <= _room(window, ratio, max_tokens)
    assert sent.startswith("Web page content:\n---\nline 0 of the page\n")
    assert sent.endswith("line 199999 of the page\n---\n\nWhat does it say?\n\nAnswer from the content.")
    assert "omitted" in sent


async def test_a_prompt_within_the_room_is_sent_as_it_is(tmp_path):
    manager = _manager(tmp_path, 32_000)

    await manager.summarize("short", "Summarize: short")

    assert manager.sent == [("Summarize: short", 12000)]


@pytest.mark.parametrize(("window", "max_output"), [(8_192, 2_048), (12_000, 4_096)])
async def test_a_small_model_s_summary_keeps_its_instructions_and_content(
    tmp_path, window, max_output
):
    """The call writes at most the model's max output, not the 12,000 asked
    for: the room is what is left of the window after that."""
    manager = _manager(tmp_path, window, max_output)
    content = "A short page about volcanoes. " * 20
    prompt = f"Web page content:\n---\n{content}---\n\nWhat does it say?"

    await manager.summarize(content, prompt)

    assert manager.sent == [(prompt, 12000)]


async def test_a_room_too_small_for_a_useful_prompt_sends_it_as_it_is(tmp_path):
    """8,192 tokens less a 12,000-token output leave nothing: cutting the
    prompt to fit would send only the marker. The provider answers instead."""
    manager = _manager(tmp_path, 8_192, 64_000)
    prompt = "Summarize this page.\n\n" + "lava " * 20_000

    await manager.summarize("lava", prompt)

    assert manager.sent == [(prompt, 12000)]


@pytest.mark.parametrize(("window", "ratio"), MODELS)
async def test_fact_extraction_sends_a_bounded_prompt(tmp_path, window, ratio):
    manager = _manager(tmp_path, window)
    manager.answer = json.dumps([
        {"action": "new", "description": "A fact.", "content": "A fact.", "type": "user", "scope": "user"}
    ])
    messages = [
        {"role": "user" if n % 2 else "assistant", "content": f"message {n}: " + "x" * 100_000 + " END"}
        for n in range(30)
    ]

    assert [p.description for p in await manager.on_session_end(messages)] == ["A fact."]

    (sent, max_tokens), = manager.sent
    assert len(sent) <= _room(window, ratio, max_tokens)
    assert sent.startswith("You keep a user's long-term memory")
    assert "assistant: message 10: x" in sent  # the first of the last 20
    assert sent.endswith("x END")
