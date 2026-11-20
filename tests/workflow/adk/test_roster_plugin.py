"""ADK: the main agent's roster opens its requests after the memory index, is
never dropped, and is folded into the first user text with it.

``RosterPlugin`` puts the roster (``workflow/agents/roster.py``) first in the
main agent's requests; it comes before ``MemoryIndexPlugin`` in the chain, so
the index goes ahead of it. What opens a request is the index, the roster,
then a summary (``protected_head``): the fit guard never drops them, folds them
into the next user text in that order (``fold_summary``), and the overflow
handler's retry keeps the index and the roster ahead of its new summary.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_request import LlmRequest  # noqa: E402

from agentic_cli.workflow.adk.fit_guard_plugin import (  # noqa: E402
    FIT_NOTE,
    drop_turns,
    fold_summary,
    is_roster,
    protected_head,
)
from agentic_cli.workflow.adk.memory_index_plugin import MemoryIndexPlugin, index_content  # noqa: E402
from agentic_cli.workflow.adk.overflow_plugin import _with_summary  # noqa: E402
from agentic_cli.workflow.adk.roster_plugin import RosterPlugin, roster_content  # noqa: E402
from agentic_cli.workflow.agents.roster import ROSTER_CLOSE, ROSTER_HEADER, ROSTER_OPEN  # noqa: E402
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX  # noqa: E402
from tests.workflow.adk.test_fit_guard import (  # noqa: E402
    CLAUDE,
    LONG,
    NOTE_ROOM,
    call_context,
    model,
    tokens,
    user,
)
from tests.workflow.adk.test_memory_index_plugin import INDEX, texts  # noqa: E402

ROSTER = "\n".join([ROSTER_OPEN, ROSTER_HEADER,
                    "- researcher-1 (researcher, idle): Find X. — X is 42.", ROSTER_CLOSE])
SUMMARY = f"{SUMMARY_PREFIX}\n\nThe user is Ada."


class TestThePlugin:
    async def test_the_roster_goes_first_for_the_main_agent_only(self):
        plugin = RosterPlugin(root="lead", roster_for=lambda: ROSTER)
        mine, theirs = LlmRequest(contents=[user("Hi.")]), LlmRequest(contents=[user("Hi.")])

        await plugin.before_model_callback(callback_context=call_context("lead"), llm_request=mine)
        await plugin.before_model_callback(
            callback_context=call_context("researcher"), llm_request=theirs
        )

        assert [texts(c) for c in mine.contents] == [[ROSTER], ["Hi."]]
        assert [texts(c) for c in theirs.contents] == [["Hi."]]
        assert is_roster(mine.contents[0]) and not is_roster(user("Hi."))

    async def test_no_roster_no_change(self):
        plugin = RosterPlugin(root="lead", roster_for=lambda: "")
        request = LlmRequest(contents=[user("Hi.")])

        await plugin.before_model_callback(callback_context=call_context("lead"), llm_request=request)

        assert [texts(c) for c in request.contents] == [["Hi."]]

    async def test_before_the_memory_index_in_the_chain_the_index_goes_ahead_of_it(self):
        request = LlmRequest(contents=[user(SUMMARY), user("Hi.")])
        chain = [RosterPlugin(root="lead", roster_for=lambda: ROSTER),
                 MemoryIndexPlugin(root="lead", index_for=lambda: INDEX)]

        for plugin in chain:
            await plugin.before_model_callback(
                callback_context=call_context("lead"), llm_request=request
            )

        assert [texts(c) for c in request.contents] == [[INDEX], [ROSTER], [SUMMARY], ["Hi."]]


class TestWhatOpensARequest:
    def test_the_index_the_roster_then_a_summary(self):
        opening = [index_content(INDEX), roster_content(ROSTER), user(SUMMARY)]

        assert protected_head([*opening, user("Hi.")]) == opening
        assert protected_head([roster_content(ROSTER), user("Hi.")]) == [roster_content(ROSTER)]
        # Out of that order it opens nothing.
        assert protected_head([roster_content(ROSTER), index_content(INDEX), user("Hi.")]) == [
            roster_content(ROSTER)
        ]

    def test_folded_into_the_next_users_text_in_that_order(self):
        folded = fold_summary([index_content(INDEX), roster_content(ROSTER), user(SUMMARY),
                               user("Two?"), model("A2.")])

        assert [texts(c) for c in folded] == [[INDEX, ROSTER, SUMMARY, "Two?"], ["A2."]]
        assert fold_summary(folded) == folded

    def test_dropping_turns_keeps_them(self):
        contents = [index_content(INDEX), roster_content(ROSTER), user(SUMMARY), user(LONG),
                    model("A1."), user(LONG), model("A2."), user("Three?")]
        target = tokens([*contents[:3], *contents[5:]]) + NOTE_ROOM

        kept, dropped = drop_turns(contents, target_tokens=target, kind=CLAUDE)

        assert dropped == 1
        assert kept[:3] == contents[:3]
        assert texts(kept[3])[0] == FIT_NOTE

    def test_the_overflow_retry_keeps_the_index_and_the_roster_first(self):
        old = user(f"{SUMMARY_PREFIX}\n\nOld summary.")
        new = f"{SUMMARY_PREFIX}\n\nNew summary."

        retry = _with_summary([index_content(INDEX), roster_content(ROSTER), old, user("Two?")], new)
        alone = _with_summary([roster_content(ROSTER), user("Two?")], new)

        assert [texts(c) for c in retry] == [[INDEX, ROSTER, new, "Two?"]]
        assert [texts(c) for c in alone] == [[ROSTER, new, "Two?"]]
