"""The example in ``BaseWorkflowManager``'s docstring is a backend that works.

The docstring's example is what a backend author starts from. It once could
not be instantiated (it left five abstract members out), and nothing noticed,
since it is text. The test takes the example from the docstring itself, builds
the class and runs it through the lifecycle the docstring describes, so it
cannot rot again.
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Any

import pytest

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE
from tests.conftest import MockContext


def _example_class() -> type[BaseWorkflowManager]:
    """The class the docstring's ``Example:`` section defines."""
    _, marker, example = inspect.getdoc(BaseWorkflowManager).partition("Example:")
    assert marker, "the class docstring has no Example: section"
    # The section is prose, ``::``, then the code to the end of the docstring.
    _, literal, code = example.partition("::\n")
    assert literal, "the Example: section has no code block (a line ending in ::)"
    namespace: dict[str, Any] = {}
    exec(compile(textwrap.dedent(code), "<BaseWorkflowManager example>", "exec"), namespace)
    return namespace["CustomWorkflowManager"]


@pytest.fixture
def example() -> type[BaseWorkflowManager]:
    return _example_class()


async def _reply(manager: BaseWorkflowManager, message: str) -> list[str]:
    return [
        event.content async for event in manager.process(message, "user", "session")
    ]


async def test_the_example_can_be_instantiated(example):
    with MockContext(google_api_key="test-key") as ctx:
        manager = example([AgentConfig(name="a", prompt="p")], settings=ctx.settings)

    assert isinstance(manager, BaseWorkflowManager)
    assert manager.backend_type == "custom"


async def test_a_turn_initializes_it_and_replies(example):
    with MockContext(google_api_key="test-key") as ctx:
        manager = example([AgentConfig(name="a", prompt="p")], settings=ctx.settings)

        assert await _reply(manager, "hello") == ["echo: hello"]
        assert manager.is_initialized

        await manager.cleanup()


async def test_cleanup_releases_it_and_the_next_turn_starts_it_again(example):
    with MockContext(google_api_key="test-key") as ctx:
        manager = example([AgentConfig(name="a", prompt="p")], settings=ctx.settings)
        await _reply(manager, "first")

        await manager.cleanup()

        assert not manager.is_initialized
        assert manager.catalog is None
        assert manager.services == {}
        assert await _reply(manager, "second") == ["echo: second"]
        assert manager.is_initialized
        assert manager.catalog is not None

        await manager.cleanup()


async def test_reinitialize_keeps_the_services_it_carries(example):
    with MockContext(google_api_key="test-key") as ctx:
        manager = example([AgentConfig(name="a", prompt="p")], settings=ctx.settings)
        await _reply(manager, "first")
        engine = manager.services[PERMISSION_ENGINE]

        await manager.reinitialize()

        assert manager.is_initialized
        assert manager.services[PERMISSION_ENGINE] is engine
        assert await _reply(manager, "again") == ["echo: again"]

        await manager.cleanup()
