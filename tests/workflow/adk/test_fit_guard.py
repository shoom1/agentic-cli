"""The fit guard: a request over its agent's input budget goes out under the trigger.

``FitGuardPlugin`` (``workflow/adk/fit_guard_plugin.py``) estimates each
request (system prompt, tool declarations and contents, at the provider's
characters per token, plus 10%). Over the calling agent's budget B, it drops
the oldest whole turns from that request only, never parting a call from its
result, always keeping the newest turn and a summary that opens the
conversation, until the request is under the agent's trigger, and folds a
note into the first kept user content. It reports that once a turn. When the
newest turn alone is over the budget, nothing it could drop would help, and
the request goes as it is: the overflow handler answers.

These tests drive the pieces on hand-built requests; turns on the real
manager are in ``tests/workflow/test_adk_fit_guard.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.models import ProviderKind  # noqa: E402
from agentic_cli.workflow.adk import fit_guard_plugin  # noqa: E402
from agentic_cli.workflow.adk.fit_guard_plugin import (  # noqa: E402
    FIT_NOTE,
    FitGuardPlugin,
    drop_turns,
    estimate_request_tokens,
)
from agentic_cli.workflow.context import ContextBudget  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402
from agentic_cli.workflow.live_events import LiveEvents, reporting  # noqa: E402
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX  # noqa: E402

CLAUDE = ProviderKind.ANTHROPIC
LONG = "word " * 2_400  # 12,000 characters


def user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def model(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part.from_text(text=text)])


def call(name: str, call_id: str) -> types.Content:
    return types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(id=call_id, name=name, args={}))
    ])


def result(name: str, call_id: str, **response) -> types.Content:
    return types.Content(role="user", parts=[
        types.Part(function_response=types.FunctionResponse(
            id=call_id, name=name, response=response
        ))
    ])


def tokens(contents: list[types.Content], kind: ProviderKind = CLAUDE) -> int:
    return estimate_request_tokens(LlmRequest(contents=list(contents)), kind)


#: Room enough for the note.
NOTE_ROOM = 50


class TestDropTurns:
    def test_the_oldest_whole_turns_go_first(self):
        contents = [user(LONG), model("A1."), user(LONG), model("A2."), user("Three?")]

        kept, dropped = drop_turns(
            contents, target_tokens=tokens(contents[2:]) + NOTE_ROOM, kind=CLAUDE
        )

        assert dropped == 1
        assert kept[1:] == contents[3:]
        assert [part.text for part in kept[0].parts] == [FIT_NOTE, LONG]

    def test_until_the_request_is_under_the_target(self):
        contents = [user(LONG), model("A1."), user(LONG), model("A2."), user("Three?")]

        kept, dropped = drop_turns(
            contents, target_tokens=tokens(contents[4:]) + NOTE_ROOM, kind=CLAUDE
        )

        assert dropped == 2
        assert [part.text for part in kept[0].parts] == [FIT_NOTE, "Three?"]
        assert tokens(kept) < tokens(contents[4:]) + NOTE_ROOM

    def test_the_newest_turn_is_always_kept(self):
        contents = [user(LONG), model("A1."), user("Two?"), call("get_plan", "c1"),
                    result("get_plan", "c1", plan="")]

        kept, dropped = drop_turns(contents, target_tokens=0, kind=CLAUDE)

        assert dropped == 1
        assert kept[1:] == contents[3:]
        assert [part.text for part in kept[0].parts] == [FIT_NOTE, "Two?"]

    def test_a_call_is_never_parted_from_its_result(self):
        # A user's text right after a call is no turn start: the cut would
        # leave the result without its call.
        contents = [user(LONG), call("lookup", "c1"), user("While you look: hurry."),
                    result("lookup", "c1", n=1), model("A1."), user(LONG), model("A2."),
                    user("Three?")]

        # Were the interjection a start, the cut there would fit the target.
        kept, dropped = drop_turns(
            contents, target_tokens=tokens(contents[2:]) + NOTE_ROOM, kind=CLAUDE
        )

        assert dropped == 1
        assert kept[1:] == contents[6:]
        assert [part.text for part in kept[0].parts] == [FIT_NOTE, LONG]

    def test_a_summary_that_opens_the_conversation_stays(self):
        summary = user(f"{SUMMARY_PREFIX}\n\nThe user is Ada.")
        contents = [summary, user(LONG), model("A2."), user("Three?")]

        kept, dropped = drop_turns(contents, target_tokens=0, kind=CLAUDE)

        assert dropped == 1
        assert kept[0] == summary
        assert [part.text for part in kept[1].parts] == [FIT_NOTE, "Three?"]

    def test_the_contents_given_are_left_alone(self):
        contents = [user(LONG), model("A1."), user("Two?")]
        before = [content.model_copy(deep=True) for content in contents]

        drop_turns(contents, target_tokens=0, kind=CLAUDE)

        assert contents == before

    def test_each_content_is_measured_once(self, monkeypatch):
        """However many turns go, each content is serialized once: at a 1M-token
        session the guard would otherwise spend seconds a call."""
        contents = [c for n in range(30) for c in (user(f"{n}: {LONG}"), model(f"A{n}."))]
        contents.append(user("Now?"))
        measured: list[int] = []
        dump = types.Content.model_dump_json

        def counted(self, **kwargs):
            measured.append(id(self))
            return dump(self, **kwargs)

        monkeypatch.setattr(types.Content, "model_dump_json", counted)

        kept, dropped = drop_turns(contents, target_tokens=0, kind=CLAUDE)

        assert dropped == 30
        assert len(measured) <= len(contents)

    def test_one_turn_is_nothing_to_drop(self):
        contents = [user(LONG), call("get_plan", "c1"), result("get_plan", "c1", plan=""),
                    model("A1.")]

        kept, dropped = drop_turns(contents, target_tokens=0, kind=CLAUDE)

        assert (kept, dropped) == (contents, 0)


class TestFoldSummary:
    """A summarized conversation opens with the summary, a user content of its
    own: folded into the next user's text, no provider sees two user messages
    in a row (a strict chat template refuses them)."""

    def test_a_leading_summary_joins_the_next_users_text(self):
        summary = user(f"{SUMMARY_PREFIX}\n\nThe user is Ada.")
        contents = [summary, user("Two?"), model("A2."), user("Three?")]

        folded = fit_guard_plugin.fold_summary(contents)

        assert [part.text for part in folded[0].parts] == [summary.parts[0].text, "Two?"]
        assert folded[0].role == "user"
        assert folded[1:] == contents[2:]
        assert contents[1].parts[0].text == "Two?"  # the contents given are left alone

    @pytest.mark.parametrize("after", ["result", "model"])
    def test_not_into_a_tool_result_or_a_model_turn(self, after):
        summary = user(f"{SUMMARY_PREFIX}\n\nThe user is Ada.")
        nxt = result("get_plan", "c1", plan="") if after == "result" else model("A1.")
        contents = [summary, nxt, user("Two?")]

        assert fit_guard_plugin.fold_summary(contents) == contents

    def test_without_a_summary_nothing_changes(self):
        contents = [user("One."), user("Two?")]

        assert fit_guard_plugin.fold_summary(contents) == contents


class TestEstimateRequestTokens:
    def test_the_system_prompt_the_tools_and_the_contents_count_with_a_margin(self):
        tool = types.Tool(function_declarations=[
            types.FunctionDeclaration(name="lookup", description="d" * 1_000)
        ])
        bare = LlmRequest(config=types.GenerateContentConfig(system_instruction="s" * 3_300))
        full = LlmRequest(
            contents=[user(LONG)],
            config=types.GenerateContentConfig(system_instruction="s" * 3_300, tools=[tool]),
        )

        assert estimate_request_tokens(bare, CLAUDE) == 1_100  # 3,300 / 3.3, plus 10%
        assert estimate_request_tokens(full, CLAUDE) > 1_100 + 1_000 / 3.3 + 12_000 / 3.3
        assert estimate_request_tokens(bare, ProviderKind.OPENAI) == round(3_300 / 4 * 1.1)


class Recorded(LiveEvents):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[WorkflowEvent] = []

    def emit(self, event: WorkflowEvent) -> None:
        self.events.append(event)


def budget_for(contents: list[types.Content]) -> ContextBudget:
    """A budget the whole of ``contents`` is over, and which all but its first
    turn fits under the trigger of."""
    over = tokens(contents) - 1
    return ContextBudget(window=over * 2, window_known=True, budget=over,
                         trigger=tokens(contents[2:]) + NOTE_ROOM, keep=0)


async def guarded(plugin: FitGuardPlugin, request: LlmRequest, live: LiveEvents,
                  agent: str = "solo") -> None:
    with reporting(live):
        answer = await plugin.before_model_callback(
            callback_context=SimpleNamespace(agent_name=agent, invocation_id="i1"),
            llm_request=request,
        )
    assert answer is None  # the request still goes to the model


class TestThePlugin:
    async def test_over_the_budget_the_request_goes_out_under_the_trigger(self):
        contents = [user(LONG), model("A1."), user(LONG), model("A2."), user("Three?")]
        budget = budget_for(contents)
        plugin = FitGuardPlugin({"solo": budget}, {"solo": CLAUDE})
        request, live = LlmRequest(contents=list(contents)), Recorded()

        await guarded(plugin, request, live)

        assert estimate_request_tokens(request, CLAUDE) < budget.trigger
        assert request.contents[0].parts[0].text == FIT_NOTE
        (trimmed,) = live.events
        assert trimmed.type is EventType.CONTEXT_TRIMMED
        assert trimmed.metadata == {
            "messages_before": 5, "messages_after": 3, "messages_removed": 2,
            "source": "fit_guard", "agent": "solo", "turns_dropped": 1,
        }

    async def test_within_the_budget_it_is_left_alone(self):
        contents = [user(LONG), model("A1."), user(LONG), model("A2."), user("Three?")]
        roomy = ContextBudget(window=1_000_000, window_known=True, budget=tokens(contents),
                              trigger=10, keep=0)
        plugin = FitGuardPlugin({"solo": roomy}, {"solo": CLAUDE})
        request, live = LlmRequest(contents=list(contents)), Recorded()

        await guarded(plugin, request, live)

        assert (request.contents, live.events) == (contents, [])

    async def test_it_is_reported_once_a_turn(self):
        contents = [user(LONG), model("A1."), user(LONG), model("A2."), user("Three?")]
        plugin = FitGuardPlugin({"solo": budget_for(contents)}, {"solo": CLAUDE})
        turn, next_turn = Recorded(), Recorded()

        for live in (turn, turn, next_turn):
            request = LlmRequest(contents=list(contents))
            await guarded(plugin, request, live)
            assert request.contents[0].parts[0].text == FIT_NOTE

        assert (len(turn.events), len(next_turn.events)) == (1, 1)

    async def test_a_newest_turn_over_the_budget_alone_is_left_alone(self):
        contents = [user("One."), model("A1."), user(LONG * 3)]
        too_small = ContextBudget(window=10_000, window_known=True, budget=tokens(contents[2:]) - 1,
                                  trigger=100, keep=0)
        plugin = FitGuardPlugin({"solo": too_small}, {"solo": CLAUDE})
        request, live = LlmRequest(contents=list(contents)), Recorded()

        await guarded(plugin, request, live)

        assert (request.contents, live.events) == (contents, [])

    async def test_a_summary_is_folded_into_the_next_user_text_within_the_budget(self):
        summary = user(f"{SUMMARY_PREFIX}\n\nThe user is Ada.")
        contents = [summary, user("Two?")]
        roomy = ContextBudget(window=1_000_000, window_known=True, budget=900_000,
                              trigger=10, keep=0)
        plugin = FitGuardPlugin({"solo": roomy}, {"solo": CLAUDE})
        request, live = LlmRequest(contents=list(contents)), Recorded()

        await guarded(plugin, request, live)

        assert [[p.text for p in c.parts] for c in request.contents] == [
            [summary.parts[0].text, "Two?"]
        ]
        assert live.events == []

    async def test_it_keeps_the_request_as_it_saw_it_until_the_call_ends(self):
        """The overflow handler retries from this copy: a model's adapter may
        rewrite the request in place (Gemini's context cache moves the system
        instruction, the tools and the cached contents out of it)."""
        contents = [user(LONG), model("A1."), user(LONG), model("A2."), user("Three?")]
        plugin = FitGuardPlugin({"solo": budget_for(contents)}, {"solo": CLAUDE})
        context = SimpleNamespace(agent_name="solo", invocation_id="i1")
        config = types.GenerateContentConfig(system_instruction="You help.")
        request = LlmRequest(contents=list(contents), config=config)

        await plugin.before_model_callback(callback_context=context, llm_request=request)
        request.config.system_instruction = None  # as Gemini's cache manager does
        request.contents = request.contents[-1:]
        seen_contents, seen_config = plugin.request_as_seen(context)

        assert seen_contents == contents  # as it came in, before the guard's trim
        assert seen_config.system_instruction == "You help."
        await plugin.after_model_callback(callback_context=context, llm_response=LlmResponse())
        assert plugin.request_as_seen(context) is None

    async def test_an_agent_it_has_no_budget_for_is_left_alone(self):
        contents = [user(LONG), model("A1."), user(LONG), model("A2."), user("Three?")]
        plugin = FitGuardPlugin({"solo": budget_for(contents)}, {"solo": CLAUDE})
        request, live = LlmRequest(contents=list(contents)), Recorded()

        await guarded(plugin, request, live, agent="native_helper")

        assert (request.contents, live.events) == (contents, [])
