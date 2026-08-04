"""Agent-graph validation and settings-scoped construction.

Before construction the graph was only implicitly checked: a missing sub-agent
was logged and dropped, duplicates/cycles/self-references were unchecked, and
the two-pass build (leaves, then coordinators) depended on declaration order —
a coordinator whose child was itself a coordinator lost that child. Callable
prompts were also evaluated outside the manager's settings context, so they
resolved against the global singleton.
"""

from __future__ import annotations

import pytest

from agentic_cli.workflow.config import (
    AgentConfig,
    AgentGraphError,
    validate_agent_graph,
)

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from tests.conftest import MockContext  # noqa: E402


class TestGraphValidation:
    def test_empty_graph_is_rejected(self):
        with pytest.raises(AgentGraphError, match="No agents configured"):
            validate_agent_graph([])

    def test_duplicate_names_are_rejected(self):
        configs = [
            AgentConfig(name="dup", prompt="a"),
            AgentConfig(name="dup", prompt="b"),
        ]
        with pytest.raises(AgentGraphError, match="Duplicate agent name.*dup"):
            validate_agent_graph(configs)

    def test_missing_sub_agent_is_rejected_not_dropped(self):
        configs = [AgentConfig(name="coord", prompt="p", sub_agents=["ghost"])]
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(configs)
        assert "coord -> ghost" in str(exc.value)

    def test_self_reference_is_rejected(self):
        configs = [AgentConfig(name="loop", prompt="p", sub_agents=["loop"])]
        with pytest.raises(AgentGraphError, match="themselves as sub_agents.*loop"):
            validate_agent_graph(configs)

    def test_cycle_is_rejected_and_named(self):
        configs = [
            AgentConfig(name="a", prompt="p", sub_agents=["b"]),
            AgentConfig(name="b", prompt="p", sub_agents=["c"]),
            AgentConfig(name="c", prompt="p", sub_agents=["a"]),
        ]
        with pytest.raises(AgentGraphError, match="Delegation cycle") as exc:
            validate_agent_graph(configs)
        assert "a" in str(exc.value) and "c" in str(exc.value)

    def test_shared_child_is_rejected_as_unsupported_topology(self):
        configs = [
            AgentConfig(name="p1", prompt="p", sub_agents=["shared"]),
            AgentConfig(name="p2", prompt="p", sub_agents=["shared"]),
            AgentConfig(name="shared", prompt="p"),
        ]
        with pytest.raises(AgentGraphError, match="more than one parent"):
            validate_agent_graph(configs)

    def test_build_order_puts_children_first(self):
        configs = [
            AgentConfig(name="top", prompt="p", sub_agents=["mid"]),
            AgentConfig(name="mid", prompt="p", sub_agents=["leaf"]),
            AgentConfig(name="leaf", prompt="p"),
        ]
        graph = validate_agent_graph(configs)
        order = list(graph.build_order)
        assert order.index("leaf") < order.index("mid") < order.index("top")
        assert graph.root_name == "top"


class TestConstructionOrderIndependence:
    """Declaration order must not change the built hierarchy."""

    def _nested_configs(self, order: str) -> list[AgentConfig]:
        top = AgentConfig(name="top", prompt="p", sub_agents=["mid"])
        mid = AgentConfig(name="mid", prompt="p", sub_agents=["leaf"])
        leaf = AgentConfig(name="leaf", prompt="p")
        return {"top-first": [top, mid, leaf], "leaf-first": [leaf, mid, top]}[order]

    @pytest.mark.parametrize("order", ["top-first", "leaf-first"])
    def test_two_level_hierarchy_is_built_completely(self, order: str):
        with MockContext(google_api_key="test-key") as ctx:
            mgr = GoogleADKWorkflowManager(
                agent_configs=self._nested_configs(order), settings=ctx.settings
            )
            root = mgr._create_agents()

        assert root.name == "top"
        assert [a.name for a in root.sub_agents] == ["mid"]
        # The nested coordinator must keep its own child.
        assert [a.name for a in root.sub_agents[0].sub_agents] == ["leaf"]


class TestPromptSettingsScope:
    """Prompt factories resolve against the manager's settings, not the global."""

    def test_zero_arg_factory_sees_manager_settings(self):
        seen: list[str] = []

        def _prompt() -> str:
            from agentic_cli.config import get_settings

            seen.append(get_settings().app_name)
            return "instruction"

        with MockContext(google_api_key="test-key", app_name="manager-app") as ctx:
            from agentic_cli.config import BaseSettings, set_settings

            # Global singleton points somewhere else entirely.
            set_settings(BaseSettings(app_name="global-app"))
            mgr = GoogleADKWorkflowManager(
                agent_configs=[AgentConfig(name="a", prompt=_prompt)],
                settings=ctx.settings,
            )
            mgr._create_agents()

        assert seen == ["manager-app"]

    def test_factory_taking_settings_is_passed_them(self):
        received: list[object] = []

        def _prompt(settings) -> str:
            received.append(settings)
            return f"app={settings.app_name}"

        with MockContext(google_api_key="test-key", app_name="manager-app") as ctx:
            mgr = GoogleADKWorkflowManager(
                agent_configs=[AgentConfig(name="a", prompt=_prompt)],
                settings=ctx.settings,
            )
            root = mgr._create_agents()

        assert received == [ctx.settings]
        assert root.instruction == "app=manager-app"

    def test_plain_string_prompt_is_unchanged(self):
        config = AgentConfig(name="a", prompt="literal")
        assert config.get_prompt() == "literal"
        assert config.get_prompt(settings=object()) == "literal"

    def test_zero_arg_factory_without_settings_still_works(self):
        config = AgentConfig(name="a", prompt=lambda: "made up")
        assert config.get_prompt() == "made up"


class TestPromptFactorySignatures:
    """Only the two documented shapes are accepted; the rest fail by name."""

    def _config(self, prompt):
        return AgentConfig(name="scribe", prompt=prompt)

    def test_zero_argument_factory(self):
        assert self._config(lambda: "made up").get_prompt() == "made up"

    def test_factory_with_only_defaulted_parameters_is_called_with_none(self):
        """``lambda prefix="default": ...`` binds with zero args — keep doing that."""
        config = self._config(lambda prefix="default": f"{prefix}!")
        assert config.get_prompt() == "default!"
        # Even when settings are available, the zero-arg form wins.
        assert config.get_prompt(settings=object()) == "default!"

    def test_single_required_parameter_receives_settings(self):
        sentinel = object()
        received: list[object] = []

        def _prompt(settings):
            received.append(settings)
            return "ok"

        assert self._config(_prompt).get_prompt(settings=sentinel) == "ok"
        assert received == [sentinel]

    def test_single_required_parameter_without_settings_is_an_error(self):
        with pytest.raises(AgentGraphError, match="scribe"):
            self._config(lambda settings: "x").get_prompt()

    def test_two_required_parameters_are_rejected(self):
        with pytest.raises(AgentGraphError, match="unsupported prompt factory"):
            self._config(lambda settings, extra: "x").get_prompt(settings=object())

    def test_var_args_factory_is_called_with_no_arguments(self):
        def _prompt(*args):
            return f"args={len(args)}"

        assert self._config(_prompt).get_prompt(settings=object()) == "args=0"

    def test_non_string_result_is_rejected(self):
        with pytest.raises(AgentGraphError, match="not a string"):
            self._config(lambda: 42).get_prompt()

    def test_async_factory_is_rejected(self):
        async def _prompt():
            return "nope"

        with pytest.raises(AgentGraphError, match="async prompt factories"):
            self._config(_prompt).get_prompt()

    def test_error_names_the_agent(self):
        config = AgentConfig(name="researcher", prompt=lambda a, b: "x")
        with pytest.raises(AgentGraphError) as exc:
            config.get_prompt(settings=object())
        assert "researcher" in str(exc.value)

    def test_plain_string_prompt_is_never_called(self):
        assert AgentConfig(name="a", prompt="literal").get_prompt() == "literal"


class TestSingleRoot:
    """A forest is rejected: only one root ever runs."""

    def test_two_roots_are_rejected_and_both_named(self):
        configs = [
            AgentConfig(name="alpha", prompt="p", sub_agents=["helper"]),
            AgentConfig(name="helper", prompt="p"),
            AgentConfig(name="beta", prompt="p"),  # never reachable
        ]
        with pytest.raises(AgentGraphError, match="2 roots") as exc:
            validate_agent_graph(configs)
        message = str(exc.value)
        assert "alpha" in message and "beta" in message

    def test_two_standalone_agents_are_rejected(self):
        configs = [
            AgentConfig(name="one", prompt="p"),
            AgentConfig(name="two", prompt="p"),
        ]
        with pytest.raises(AgentGraphError, match="roots"):
            validate_agent_graph(configs)

    def test_single_agent_is_the_root(self):
        graph = validate_agent_graph([AgentConfig(name="solo", prompt="p")])
        assert graph.root_name == "solo"

    def test_single_tree_is_accepted(self):
        configs = [
            AgentConfig(name="coord", prompt="p", sub_agents=["a", "b"]),
            AgentConfig(name="a", prompt="p"),
            AgentConfig(name="b", prompt="p"),
        ]
        assert validate_agent_graph(configs).root_name == "coord"

    def test_manager_rejects_a_forest_before_it_builds_anything(self):
        with MockContext(google_api_key="test-key") as ctx:
            mgr = GoogleADKWorkflowManager(
                agent_configs=[
                    AgentConfig(name="one", prompt="p"),
                    AgentConfig(name="two", prompt="p"),
                ],
                settings=ctx.settings,
            )
            with pytest.raises(AgentGraphError, match="roots"):
                mgr._create_agents()


class TestValidationHappensBeforeAllocation:
    """A static graph error must not cost discovery or service construction."""

    async def test_bad_graph_fails_before_discovery_and_services(self):
        from unittest.mock import AsyncMock, MagicMock

        with MockContext(google_api_key="test-key") as ctx:
            mgr = GoogleADKWorkflowManager(
                agent_configs=[
                    AgentConfig(name="coord", prompt="p", sub_agents=["ghost"]),
                ],
                settings=ctx.settings,
            )
            refresh = AsyncMock()
            mgr._model_registry = MagicMock(refresh=refresh)
            created: list[str] = []
            mgr._ensure_managers_initialized = lambda: created.append("services")
            mgr._make_session_service = lambda: created.append("session") or object()

            with pytest.raises(AgentGraphError, match="ghost"):
                await mgr.initialize_services()

            refresh.assert_not_awaited()
            assert created == [], "resources were allocated before validation"
            assert mgr.is_initialized is False
