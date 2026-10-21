"""The recording session fakes have only what thinking-prompt has.

``tests/event_replay.py``'s ``RecordingSession`` and ``RecordingThinkingBox``
stand in for ``ThinkingPromptSession`` and ``ThinkingBox`` in the render
tests. A fake that grows a method thinking-prompt lacks lets the code calling
it pass and then fail on every real session, as ``input_dialog`` did. So
every public method and property a fake defines must be the real class's,
sync or async as it is, and take only parameters the real one takes, by the
same name, of the same kind and, when positional, in the same place.
"""

from __future__ import annotations

import inspect

import pytest
from thinking_prompt import ThinkingBox, ThinkingPromptSession

from tests.event_replay import RecordingSession, RecordingThinkingBox

# How tests read the record. They are the fake's own, not the session's API.
QUERIES = {"kinds", "of", "responses", "rich", "warnings", "errors"}

PAIRS = [(RecordingSession, ThinkingPromptSession), (RecordingThinkingBox, ThinkingBox)]


def _members(fake: type) -> list[str]:
    """The public methods and properties ``fake`` itself defines."""
    return sorted(
        name
        for name, value in vars(fake).items()
        if not name.startswith("_")
        and name not in QUERIES
        and (callable(value) or isinstance(value, property))
    )


def _cases() -> list:
    return [
        pytest.param(fake, real, name, id=f"{fake.__name__}.{name}")
        for fake, real in PAIRS
        for name in _members(fake)
    ]


@pytest.mark.parametrize(("fake", "real", "name"), _cases())
def test_each_member_is_the_real_class_s(fake, real, name):
    fake_member = vars(fake)[name]
    assert hasattr(real, name), f"{real.__name__} has no {name}"
    real_member = inspect.getattr_static(real, name)

    if isinstance(fake_member, property):
        assert isinstance(real_member, property), f"{real.__name__}.{name} is not a property"
        return
    assert callable(real_member) and not isinstance(real_member, property), (
        f"{real.__name__}.{name} is not a method"
    )
    assert inspect.iscoroutinefunction(fake_member) == inspect.iscoroutinefunction(
        real_member
    ), f"{name}: one is async and the other is not"

    fake_params = list(inspect.signature(fake_member).parameters.values())[1:]
    real_params = list(inspect.signature(real_member).parameters.values())[1:]
    by_name = {p.name: (i, p) for i, p in enumerate(real_params)}
    for index, param in enumerate(fake_params):
        assert param.name in by_name, f"{real.__name__}.{name} takes no {param.name!r}"
        real_index, real_param = by_name[param.name]
        assert param.kind == real_param.kind, f"{name}({param.name}): {param.kind} vs {real_param.kind}"
        if param.kind is param.POSITIONAL_OR_KEYWORD:
            assert index == real_index, f"{name}({param.name}) is not in its real place"


def test_only_the_queries_are_left_out():
    """The names left out are the fake's own queries, none of them a session
    method a fake could get wrong."""
    assert QUERIES <= set(vars(RecordingSession))
    assert not QUERIES & set(dir(ThinkingPromptSession))
    assert _members(RecordingSession) and _members(RecordingThinkingBox)
