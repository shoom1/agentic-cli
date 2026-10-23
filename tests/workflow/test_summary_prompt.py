"""The prompt a conversation is summarized with, on both backends.

Both backends summarize with LangChain's ``DEFAULT_SUMMARY_PROMPT``. The ADK
backend runs without LangChain (an extra), so the prompt is a copy in
``workflow/summary_prompt.py``; Deep Agents uses the same copy, and a test
holds it to LangChain's, so a change there shows. ``summary_prompt`` fills
the messages in without ``str.format``, and splices an optional focus in
before them; ``with_focus`` is the splice alone, for LangChain's helper, which
fills its prompt with ``str.format``.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from agentic_cli.workflow.summary_prompt import (
    SUMMARY_PREFIX,
    SUMMARY_PROMPT,
    summary_prompt,
    with_focus,
)

FOCUS_BLOCK = "<focus>\nFocus the summary on: the API decisions\n</focus>\n\n<messages>"


def test_it_is_langchains_prompt_word_for_word():
    summarization = pytest.importorskip("langchain.agents.middleware.summarization")

    assert SUMMARY_PROMPT == summarization.DEFAULT_SUMMARY_PROMPT


def test_the_prefix_a_summary_starts_with():
    assert SUMMARY_PREFIX == "Here is a summary of the conversation to date:"


class TestSummaryPrompt:
    def test_the_messages_go_where_the_placeholder_was(self):
        prompt = summary_prompt("user: Hi.\nsolo: Hello.")

        head, _, tail = SUMMARY_PROMPT.partition("{messages}")
        assert prompt == f"{head}user: Hi.\nsolo: Hello.{tail}"
        assert prompt.endswith("Messages to summarize:\nuser: Hi.\nsolo: Hello.\n</messages>")
        assert "<focus>" not in prompt

    def test_braces_in_the_messages_are_text(self):
        text = 'read_file returned {"content": "{messages} {0} {x}"}'

        prompt = summary_prompt(text)

        assert prompt.count(text) == 1
        assert prompt.startswith("<role>\nContext Extraction Assistant\n</role>")

    def test_a_focus_goes_just_before_the_messages(self):
        prompt = summary_prompt("user: Hi.", "the API decisions")

        assert prompt.count(FOCUS_BLOCK) == 1
        assert prompt.endswith(f"{FOCUS_BLOCK}\nMessages to summarize:\nuser: Hi.\n</messages>")

    def test_braces_in_the_focus_are_text(self):
        prompt = summary_prompt("user: Hi.", "{messages} and {0}")

        assert "Focus the summary on: {messages} and {0}\n" in prompt
        assert prompt.count("user: Hi.") == 1

    @pytest.mark.parametrize("focus", [None, "", "   "])
    def test_no_focus_splices_nothing(self, focus):
        assert summary_prompt("user: Hi.", focus) == summary_prompt("user: Hi.")


class TestWithFocus:
    def test_the_block_is_spliced_before_the_messages(self):
        prompt = with_focus(SUMMARY_PROMPT, "the API decisions")

        assert prompt.count(FOCUS_BLOCK) == 1
        assert prompt.format(messages="user: Hi.") == summary_prompt(
            "user: Hi.", "the API decisions"
        )

    def test_braces_in_the_focus_survive_str_format(self):
        prompt = with_focus(SUMMARY_PROMPT, "keep {this} and {0}")

        assert "Focus the summary on: keep {this} and {0}\n" in prompt.format(messages="m")

    @pytest.mark.parametrize("focus", [None, "", "   "])
    def test_no_focus_is_the_prompt_unchanged(self, focus):
        assert with_focus(SUMMARY_PROMPT, focus) == SUMMARY_PROMPT


def test_deep_agents_summarizes_with_the_same_prompt():
    pytest.importorskip("deepagents")
    from agentic_cli.config import BaseSettings
    from agentic_cli.models import ModelCatalog, RoleMap, resolve_model
    from agentic_cli.workflow.deepagents.summarization import build_summarization
    from tests.models.fakes import endpoints as fake_endpoints
    from tests.workflow.deepagents.fakes import ScriptedChat

    endpoints = fake_endpoints(anthropic="a-test-key")
    model = resolve_model(
        agent="solo", spec="anthropic:claude-sonnet-4-5", generation=None,
        thinking_effort="none", roles=RoleMap(default_model=None, model_roles={}),
        catalog=ModelCatalog(endpoints=endpoints), endpoints=endpoints,
    )

    middleware = build_summarization(
        ScriptedChat(label="solo"), model, BaseSettings(session_store="memory")
    )

    assert middleware._lc_helper.summary_prompt == SUMMARY_PROMPT


def test_the_adk_side_loads_no_langchain():
    """The prompt is copied so that ADK, LangChain's extra not installed, has it."""
    probe = (
        "import json, sys; import agentic_cli.workflow.summary_prompt, "
        "agentic_cli.workflow.adk.compaction; "
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('langchain', 'langchain_core', 'deepagents', 'langgraph'))))"
    )
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    assert json.loads(run.stdout.strip().splitlines()[-1]) == []
