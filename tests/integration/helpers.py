"""Event assertion utilities for integration tests.

Provides helper functions for inspecting and asserting properties
of WorkflowEvent streams produced by workflow managers.
"""

from __future__ import annotations

from agentic_cli.workflow.events import EventType, WorkflowEvent


def find_events(
    events: list[WorkflowEvent],
    event_type: EventType,
) -> list[WorkflowEvent]:
    """Filter events by EventType.

    Args:
        events: List of workflow events to search.
        event_type: The event type to filter for.

    Returns:
        List of events matching the given type.
    """
    return [e for e in events if e.type == event_type]


def find_tool_calls(
    events: list[WorkflowEvent],
    tool_name: str | None = None,
) -> list[WorkflowEvent]:
    """Filter TOOL_CALL events, optionally by tool name.

    Args:
        events: List of workflow events to search.
        tool_name: Optional tool name to match.

    Returns:
        List of TOOL_CALL events, optionally filtered by name.
    """
    calls = find_events(events, EventType.TOOL_CALL)
    if tool_name is not None:
        calls = [e for e in calls if e.metadata.get("tool_name") == tool_name]
    return calls


def find_tool_results(
    events: list[WorkflowEvent],
    tool_name: str | None = None,
) -> list[WorkflowEvent]:
    """Filter TOOL_RESULT events, optionally by tool name.

    Args:
        events: List of workflow events to search.
        tool_name: Optional tool name to match.

    Returns:
        List of TOOL_RESULT events, optionally filtered by name.
    """
    results = find_events(events, EventType.TOOL_RESULT)
    if tool_name is not None:
        results = [e for e in results if e.metadata.get("tool_name") == tool_name]
    return results


def assert_tool_called(
    events: list[WorkflowEvent],
    tool_name: str,
    args_contain: dict | None = None,
) -> WorkflowEvent:
    """Assert that a specific tool was called.

    Args:
        events: List of workflow events to search.
        tool_name: Expected tool name.
        args_contain: Optional dict of key/value pairs that must appear
                      in the tool_args metadata.

    Returns:
        The first matching TOOL_CALL event.

    Raises:
        AssertionError: If no matching tool call is found.
    """
    calls = find_tool_calls(events, tool_name)
    assert calls, f"Expected tool call '{tool_name}' not found in events"

    if args_contain:
        for call in calls:
            tool_args = call.metadata.get("tool_args", {})
            if all(tool_args.get(k) == v for k, v in args_contain.items()):
                return call
        assert False, (
            f"Tool '{tool_name}' was called but no call matched "
            f"args_contain={args_contain}. "
            f"Actual calls: {[c.metadata.get('tool_args') for c in calls]}"
        )

    return calls[0]


def assert_event_order(
    events: list[WorkflowEvent],
    expected_types: list[EventType],
) -> None:
    """Assert that events appear in the expected order.

    Checks that each type in expected_types appears in events
    in order, but allows other events in between.

    Args:
        events: List of workflow events.
        expected_types: Ordered list of expected EventTypes.

    Raises:
        AssertionError: If the expected order is not satisfied.
    """
    actual_types = [e.type for e in events]
    idx = 0
    for expected in expected_types:
        found = False
        while idx < len(actual_types):
            if actual_types[idx] == expected:
                found = True
                idx += 1
                break
            idx += 1
        if not found:
            assert False, (
                f"Expected event type {expected} not found in remaining events. "
                f"Expected order: {expected_types}, "
                f"Actual types: {actual_types}"
            )


def assert_no_errors(events: list[WorkflowEvent]) -> None:
    """Assert that no ERROR events are present.

    Args:
        events: List of workflow events.

    Raises:
        AssertionError: If any ERROR events are found.
    """
    errors = find_events(events, EventType.ERROR)
    assert not errors, (
        f"Found {len(errors)} error event(s): "
        f"{[e.content for e in errors]}"
    )
