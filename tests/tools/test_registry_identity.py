"""Canonical tool identity and registry-declared service requirements.

Two defects:

1. ``register_tool(name="public_name")`` stored the public name but handed the
   original callable to the backend, which derives the model-visible tool name
   from ``func.__name__``. Permission lookup (keyed on the backend's name) then
   missed the registry entry entirely.
2. Service detection read a central ``_TOOL_SERVICE_MAP`` keyed by tool name,
   so an extension could not ship a service-backed tool without editing the
   framework. Tools now declare ``requires=``.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Callable

import pytest

from agentic_cli.tools.registry import ToolCategory, ToolRegistry
from agentic_cli.workflow.permissions import EXEMPT
from agentic_cli.workflow.permissions.capabilities import Capability


class TestCanonicalName:
    def test_renamed_tool_exposes_the_registered_name(self):
        registry = ToolRegistry()

        @registry.register(name="public_name", capabilities=EXEMPT)
        def _internal_impl(query: str) -> dict:
            """Do a thing."""
            return {"success": True, "query": query}

        defn = registry.get("public_name")
        assert defn is not None
        # What the backend will call the tool == what the registry knows.
        assert defn.func.__name__ == "public_name"
        assert _internal_impl.__name__ == "public_name"

    def test_renamed_tool_still_runs_and_keeps_its_contract(self):
        registry = ToolRegistry()

        @registry.register(name="public_name", capabilities=EXEMPT)
        def _internal_impl(query: str, limit: int = 5) -> dict:
            """Do a thing."""
            return {"success": True, "query": query, "limit": limit}

        result = registry.get("public_name").func("hello")
        assert result == {"success": True, "query": "hello", "limit": 5}

    def test_signature_and_docstring_survive_the_rename(self):
        registry = ToolRegistry()

        @registry.register(name="public_name", capabilities=EXEMPT)
        def _internal_impl(query: str, limit: int = 5) -> dict:
            """First line of docs."""
            return {"success": True}

        func = registry.get("public_name").func
        assert list(inspect.signature(func).parameters) == ["query", "limit"]
        assert func.__doc__.startswith("First line of docs.")
        assert registry.get("public_name").description == "First line of docs."
        assert inspect.signature(func).parameters["limit"].default == 5

    def test_async_tool_stays_async(self):
        registry = ToolRegistry()

        @registry.register(name="public_async", capabilities=EXEMPT)
        async def _internal_async(x: int) -> dict:
            """Async thing."""
            return {"success": True, "x": x}

        func = registry.get("public_async").func
        assert inspect.iscoroutinefunction(func)
        assert registry.get("public_async").is_async is True
        assert asyncio.run(func(3)) == {"success": True, "x": 3}

    def test_unrenamed_tool_is_not_wrapped(self):
        """No rename, no wrapper — the registered callable is the original."""
        registry = ToolRegistry()

        def plain_tool() -> dict:
            """Plain."""
            return {"success": True}

        returned = registry.register(plain_tool, capabilities=EXEMPT)
        assert returned is plain_tool
        assert registry.get("plain_tool").func is plain_tool

    def test_permission_lookup_finds_the_renamed_tool(self):
        """The plugin looks the tool up by the backend-visible name."""
        registry = ToolRegistry()

        @registry.register(
            name="public_name",
            capabilities=[Capability("fs.read", target_arg="path")],
        )
        def _internal_impl(path: str) -> dict:
            """Read."""
            return {"success": True}

        backend_visible_name = registry.get("public_name").func.__name__
        defn = registry.get(backend_visible_name)
        assert defn is not None, "permission lookup would fail closed on a real tool"
        assert defn.capabilities[0].name == "fs.read"


class TestIdentityIsObjectIdentity:
    """The identity map is keyed by ``is``, not by hash/eq or by name.

    Identity is owned *per registry*, so these use ``registry.identify()``;
    the module-level ``identify_tool()`` answers for the default registry only
    (see :class:`TestIdentityIsPerRegistry`).
    """

    def test_registered_callable_resolves(self):
        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def a_tool() -> dict:
            """A tool."""
            return {"success": True}

        assert registry.identify(a_tool) is registry.get("a_tool")

    def test_equality_colliding_object_does_not_resolve(self):
        """``WeakKeyDictionary`` semantics let this forge identity; ``is`` does not."""
        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def a_tool() -> dict:
            """A tool."""
            return {"success": True}

        genuine = registry.get("a_tool").func

        class _Collider:
            __name__ = "a_tool"

            def __hash__(self):
                return hash(genuine)

            def __eq__(self, other):
                return other is genuine

            def __call__(self):  # pragma: no cover
                return {"success": True}

        collider = _Collider()
        assert collider == genuine and hash(collider) == hash(genuine)
        assert registry.identify(collider) is None

    def test_same_name_different_object_does_not_resolve(self):
        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def a_tool() -> dict:
            """A tool."""
            return {"success": True}

        def a_tool_impostor() -> dict:  # noqa: D401
            """Impostor."""
            return {"success": True}

        a_tool_impostor.__name__ = "a_tool"
        assert registry.identify(a_tool_impostor) is None

    def test_binding_does_not_keep_the_object_alive(self):
        """Entries are weak, so per-manager service-bound tools are collectable."""
        import gc
        import weakref

        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def a_tool() -> dict:
            """A tool."""
            return {"success": True}

        definition = registry.get("a_tool")

        def _variant() -> dict:
            """Variant."""
            return {"success": True}

        registry.bind_identity(_variant, definition)
        assert registry.identify(_variant) is definition

        ref = weakref.ref(_variant)
        del _variant
        gc.collect()
        assert ref() is None

    def test_non_weakrefable_object_is_not_bound(self):
        """An unbindable object fails closed rather than being keyed by id alone.

        Binding it by id would hand its capabilities to whatever object next
        lands on that address.
        """
        import weakref

        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def a_tool() -> dict:
            """A tool."""
            return {"success": True}

        class _NoWeakref:
            __slots__ = ()  # no __weakref__ slot

            def __call__(self):  # pragma: no cover
                return {"success": True}

        obj = _NoWeakref()
        with pytest.raises(TypeError):
            weakref.ref(obj)

        registry.bind_identity(obj, registry.get("a_tool"))
        assert registry.identify(obj) is None


class TestDuplicateNamePolicy:
    """Registering a name twice is an error, not a silent takeover.

    The second registration replaced ``_tools[name]`` and left the first
    definition's identity bindings in place, so the same tool name resolved to
    two different definitions depending on which callable you asked about — and
    the framework's service map, keyed by name, would hand an agent the
    *framework's* implementation for a name an application had taken over.
    """

    def test_conflicting_registration_raises(self):
        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def a_tool() -> dict:
            """First."""
            return {"success": True}

        with pytest.raises(ValueError, match="already registered"):

            @registry.register(
                name="a_tool",
                capabilities=[Capability("fs.read", target_arg="path")],
            )
            def _second(path: str) -> dict:
                """Second, and it wants more."""
                return {"success": True}

    def test_conflicting_requires_raises(self):
        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def a_tool() -> dict:
            """First."""
            return {"success": True}

        with pytest.raises(ValueError, match="already registered"):

            @registry.register(
                name="a_tool", capabilities=EXEMPT, requires="kb_manager"
            )
            def _second() -> dict:
                """Second."""
                return {"success": True}

    def test_identical_metadata_is_not_enough_to_alias(self):
        """Same capabilities, different docstring and signature — still a clash.

        Metadata equality says nothing about what the model sees or what the
        callable does, so it cannot be grounds for silently sharing a name.
        Sharing must be declared (see :class:`TestDeclaredVariants`).
        """
        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT, category=ToolCategory.PLANNING)
        def a_tool(content: str) -> dict:
            """Save the thing."""
            return {"success": True}

        with pytest.raises(ValueError, match="already registered"):

            @registry.register(
                name="a_tool", capabilities=EXEMPT, category=ToolCategory.PLANNING
            )
            def _other_backend(content: str, extra: int = 0) -> dict:
                """Save the thing, differently, with another argument."""
                return {"success": True}


class TestDeclaredVariants:
    """Backend-native implementations share metadata only when they say so."""

    def _declared(self):
        from agentic_cli.tools.registry import declare_tool

        registry = ToolRegistry()
        declare_tool(
            "a_tool",
            description="Do the thing.",
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
            registry=registry,
        )
        return registry

    def test_a_declaration_has_no_implementation(self):
        registry = self._declared()
        definition = registry.get("a_tool")

        assert definition.func is None
        assert definition.variants == ()

    def test_variants_bind_to_the_declaration(self):
        registry = self._declared()
        definition = registry.get("a_tool")

        @registry.register(
            variant_of="a_tool", capabilities=EXEMPT, category=ToolCategory.PLANNING
        )
        def a_tool(content: str, ctx: object) -> dict:
            """Backend A."""
            return {"success": True}

        @registry.register(
            variant_of="a_tool", capabilities=EXEMPT, category=ToolCategory.PLANNING
        )
        def _backend_b(content: str, state: dict) -> dict:
            """Backend B, other signature entirely."""
            return {"success": True}

        assert registry.identify(a_tool) is definition
        assert registry.identify(_backend_b) is definition
        assert len(registry.get("a_tool").variants) == 2
        assert registry.get("a_tool").func is None

    def test_a_variant_may_not_change_the_capabilities(self):
        registry = self._declared()

        with pytest.raises(ValueError, match="different declaration"):

            @registry.register(
                variant_of="a_tool",
                capabilities=[Capability("fs.read", target_arg="path")],
                category=ToolCategory.PLANNING,
            )
            def _greedy(path: str) -> dict:
                """Backend that wants more."""
                return {"success": True}

    def test_a_variant_may_not_change_requires(self):
        registry = self._declared()

        with pytest.raises(ValueError, match="different declaration"):

            @registry.register(
                variant_of="a_tool",
                capabilities=EXEMPT,
                category=ToolCategory.PLANNING,
                requires="kb_manager",
            )
            def _needy() -> dict:
                """Backend that wants a service."""
                return {"success": True}

    def test_variant_of_an_unknown_tool_raises(self):
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="not declared"):

            @registry.register(variant_of="nope", capabilities=EXEMPT)
            def _orphan() -> dict:
                """Nothing to be a variant of."""
                return {"success": True}

    def test_a_declaration_cannot_be_registered_over(self):
        registry = self._declared()

        with pytest.raises(ValueError, match="already registered"):

            @registry.register(name="a_tool", capabilities=EXEMPT)
            def _takeover() -> dict:
                """Takeover."""
                return {"success": True}

    def test_colliding_with_a_builtin_service_tool_raises(self):
        from agentic_cli.tools.knowledge_tools import kb_search  # noqa: F401
        from agentic_cli.tools.registry import get_registry, register_tool

        assert get_registry().get("kb_search") is not None

        with pytest.raises(ValueError, match="kb_search"):

            @register_tool(name="kb_search", capabilities=EXEMPT)
            def _my_kb_search(query: str) -> dict:
                """An application's own search."""
                return {"success": True, "mine": True}

    def test_the_builtin_survives_a_rejected_collision(self):
        """The registry is unchanged, so the framework's tool still runs."""
        from agentic_cli.tools.knowledge_tools import kb_search
        from agentic_cli.tools.registry import get_registry, identify_tool, register_tool

        before = get_registry().get("kb_search")

        with pytest.raises(ValueError):

            @register_tool(name="kb_search", capabilities=EXEMPT)
            def _my_kb_search(query: str) -> dict:
                """Impostor."""
                return {"success": True, "mine": True}

        assert get_registry().get("kb_search") is before
        assert identify_tool(kb_search) is before

    def test_replace_retires_the_old_identities(self):
        from agentic_cli.tools.registry import identify_tool  # noqa: F401

        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def a_tool() -> dict:
            """First."""
            return {"success": True}

        first = a_tool

        @registry.register(name="a_tool", capabilities=EXEMPT, replace=True)
        def _second() -> dict:
            """Second."""
            return {"success": True}

        assert registry.identify(first) is None, "a retired tool kept its capabilities"
        assert registry.identify(_second) is registry.get("a_tool")

    def test_replace_keeps_the_new_definition_reachable(self):
        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def a_tool() -> dict:
            """First."""
            return {"success": True}

        @registry.register(
            name="a_tool",
            capabilities=[Capability("fs.read", target_arg="path")],
            replace=True,
        )
        def _second(path: str) -> dict:
            """Second."""
            return {"success": True}

        definition = registry.get("a_tool")
        assert definition.capabilities[0].name == "fs.read"


class TestServiceSubstitutionIsExact:
    """A service variant replaces the *definition it implements*, not a name.

    ``_build_tools`` looked up ``service_map[definition.name]``. If an
    application deliberately replaced a built-in service tool, the framework
    still substituted its own closure — so a different implementation ran than
    the one the agent was configured with.
    """

    @staticmethod
    def _manager(tools):
        from unittest.mock import MagicMock

        from agentic_cli.workflow.config import AgentConfig

        settings = MagicMock()
        settings.app_name = "test-app"
        config = AgentConfig(
            name="a", prompt="p", tools=list(tools), include_state_tools=False
        )
        return _stub_manager_cls()(agent_configs=[config], settings=settings), config

    def test_replaced_builtin_runs_the_replacement(self):
        """The proof: the assembled tool is the application's implementation."""
        from agentic_cli.tools.registry import ToolRegistry as _Registry  # noqa: F401
        from agentic_cli.tools.registry import get_registry, register_tool

        from agentic_cli.tools.knowledge_tools import kb_search as _builtin  # noqa: F401

        registry = get_registry()
        original = registry.get("kb_search")
        try:

            @register_tool(name="kb_search", capabilities=EXEMPT, replace=True)
            def _app_kb_search(query: str) -> dict:
                """The application's own search."""
                return {"success": True, "implementation": "application"}

            mgr, config = self._manager([_app_kb_search])
            service_map = self._framework_kb_variant()

            built = mgr._build_tools(config, service_map=service_map)

            assert len(built) == 1
            assert built[0]("q")["implementation"] == "application", (
                "the framework's service variant replaced the application's tool"
            )
        finally:
            # Restore the registry for the rest of the session.
            registry._tools["kb_search"] = original
            registry.bind_identity(original.func, original)

    @staticmethod
    def _framework_kb_variant() -> dict:
        from unittest.mock import MagicMock

        from agentic_cli.tools.factories import make_kb_tools

        return {t.__name__: t for t in make_kb_tools(MagicMock())}

    def test_genuine_builtin_is_still_substituted(self):
        from agentic_cli.tools.knowledge_tools import kb_search

        mgr, config = self._manager([kb_search])
        service_map = self._framework_kb_variant()

        built = mgr._build_tools(config, service_map=service_map)

        assert built == [service_map["kb_search"]]

    def test_unbound_variant_is_not_substituted(self):
        """A map entry that is not the registered tool proves nothing."""
        from agentic_cli.tools.knowledge_tools import kb_search

        mgr, config = self._manager([kb_search])
        impostor = lambda query: {"success": True}  # noqa: E731
        impostor.__name__ = "kb_search"

        built = mgr._build_tools(config, service_map={"kb_search": impostor})

        assert built == [kb_search]


class TestVariantContractIsComplete:
    """The declared-variant mechanism, exercised the way callers use it."""

    def _declared(self, description: str = "Do the thing."):
        from agentic_cli.tools.registry import declare_tool

        registry = ToolRegistry()
        declare_tool(
            "a_tool",
            description=description,
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
            registry=registry,
        )
        return registry

    def test_direct_register_returns_the_canonical_callable(self):
        registry = self._declared()

        def _backend_impl(content: str) -> dict:
            """Backend implementation with its own name."""
            return {"success": True}

        returned = registry.register(
            _backend_impl,
            variant_of="a_tool",
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
        )

        assert returned is not None
        assert callable(returned)
        assert returned.__name__ == "a_tool"

    def test_module_level_register_tool_returns_the_canonical_callable(self):
        from agentic_cli.tools.registry import declare_tool, register_tool

        declare_tool(
            "_variant_probe_tool",
            description="Probe.",
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
        )

        def _probe_impl(content: str) -> dict:
            """Probe implementation."""
            return {"success": True}

        returned = register_tool(
            _probe_impl,
            variant_of="_variant_probe_tool",
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
        )

        assert returned is not None and returned.__name__ == "_variant_probe_tool"

    def test_assembly_never_yields_none_for_a_variant(self):
        """A config listing the variant's *original* callable must still work."""
        from unittest.mock import MagicMock

        from agentic_cli.tools.registry import declare_tool, register_tool
        from agentic_cli.workflow.config import AgentConfig

        declare_tool(
            "_assembly_probe_tool",
            description="Probe.",
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
        )

        def _assembly_impl(content: str) -> dict:
            """Backend implementation under a private name."""
            return {"success": True}

        register_tool(
            _assembly_impl,
            variant_of="_assembly_probe_tool",
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
        )

        settings = MagicMock()
        settings.app_name = "test-app"
        config = AgentConfig(
            name="a", prompt="p", tools=[_assembly_impl], include_state_tools=False
        )
        mgr = _stub_manager_cls()(agent_configs=[config], settings=settings)

        built = mgr._build_tools(config, service_map={})

        assert built and built[0] is not None, "assembly produced a None tool"
        assert built[0].__name__ == "_assembly_probe_tool"

    def test_redeclaring_with_a_different_description_raises(self):
        from agentic_cli.tools.registry import declare_tool

        registry = self._declared()

        with pytest.raises(ValueError, match="different declaration"):
            declare_tool(
                "a_tool",
                description="Something else entirely.",
                capabilities=EXEMPT,
                category=ToolCategory.PLANNING,
                registry=registry,
            )

    def test_redeclaring_the_same_contract_is_idempotent(self):
        from agentic_cli.tools.registry import declare_tool

        registry = self._declared()
        first = registry.get("a_tool")

        again = declare_tool(
            "a_tool",
            description="Do the thing.",
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
            registry=registry,
        )
        assert again is first

    def test_variant_order_is_deterministic(self):
        """Ordered by module and qualified name, not by registration order."""
        registry = self._declared()

        def _zeta(content: str) -> dict:
            """Z."""
            return {"success": True}

        def _alpha(content: str) -> dict:
            """A."""
            return {"success": True}

        for impl in (_zeta, _alpha):
            registry.register(
                impl,
                variant_of="a_tool",
                capabilities=EXEMPT,
                category=ToolCategory.PLANNING,
            )

        keys = [
            (v.__module__, getattr(v, "__wrapped__", v).__qualname__)
            for v in registry.get("a_tool").variants
        ]
        assert keys == sorted(keys)

    def test_ambiguity_error_distinguishes_variants_by_module(self):
        pytest.importorskip("langgraph")

        from agentic_cli.tools.adk import state_tools as _adk  # noqa: F401
        from agentic_cli.tools.langgraph import state_tools as _lg  # noqa: F401
        from agentic_cli.tools.tool_resolver import resolve_tool

        with pytest.raises(ValueError, match="ambiguous") as exc:
            resolve_tool("save_plan")

        message = str(exc.value)
        assert "agentic_cli.tools.adk.state_tools" in message
        assert "agentic_cli.tools.langgraph.state_tools" in message


class TestReplacedStateToolIsNotInjected:
    """``replace=True`` on a state tool must retire it from auto-injection."""

    def test_retired_variants_are_not_auto_injected(self):
        pytest.importorskip("google.adk")
        from unittest.mock import MagicMock

        from agentic_cli.tools.registry import get_registry, register_tool
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
        from agentic_cli.workflow.config import AgentConfig

        from agentic_cli.tools.adk import state_tools as _adk  # noqa: F401

        registry = get_registry()
        original = registry.get("save_plan")
        assert original is not None
        try:

            @register_tool(name="save_plan", capabilities=EXEMPT, replace=True)
            def _app_save_plan(content: str) -> dict:
                """The application's own plan tool."""
                return {"success": True, "implementation": "application"}

            settings = MagicMock()
            settings.app_name = "test-app"
            config = AgentConfig(
                name="a", prompt="p", tools=[_app_save_plan], include_state_tools=True
            )
            mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
            mgr._settings = settings
            mgr._services = {}

            built = mgr._build_tools(config, service_map={})

            names = [getattr(t, "__name__", "") for t in built]
            assert names.count("save_plan") == 1, (
                f"a retired variant was injected alongside the replacement: {names}"
            )
            assert built[names.index("save_plan")]("x")["implementation"] == (
                "application"
            )
        finally:
            registry._tools["save_plan"] = original
            for variant in original.variants:
                registry.bind_identity(variant, original)


class TestIdentityIsPerRegistry:
    """Each registry owns its bindings; the framework trusts only its own."""

    def test_default_registry_does_not_see_another_registrys_binding(self):
        from agentic_cli.tools.registry import identify_tool

        other = ToolRegistry()

        @other.register(capabilities=EXEMPT)
        def foreign() -> dict:
            """Registered elsewhere."""
            return {"success": True}

        assert other.identify(foreign) is other.get("foreign")
        assert identify_tool(foreign) is None

    def test_default_registry_sees_its_own_binding(self):
        from agentic_cli.tools.registry import (
            get_registry,
            identify_tool,
            register_tool,
        )

        @register_tool(capabilities=EXEMPT)
        def _default_registry_probe_tool() -> dict:
            """Registered in the framework's registry."""
            return {"success": True}

        assert identify_tool(_default_registry_probe_tool) is get_registry().get(
            "_default_registry_probe_tool"
        )


class TestIdentityDoesNotPinDeadObjects:
    """A binding must not keep an otherwise-dead registry graph alive.

    The identity map used to be a module-level dict holding the
    ``ToolDefinition`` *strongly*, and a definition holds its callable — so the
    weak reference to that callable could never fire. Every local
    ``ToolRegistry`` (one per test, per short-lived tool set) leaked its
    definitions and closures for the life of the process.
    """

    def test_local_registry_graph_is_collectable(self):
        import gc
        import weakref

        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def a_tool() -> dict:
            """A tool."""
            return {"success": True}

        definition = registry.get("a_tool")
        refs = {
            "registry": weakref.ref(registry),
            "definition": weakref.ref(definition),
            "callable": weakref.ref(definition.func),
        }

        del registry, definition, a_tool
        gc.collect()

        alive = [name for name, ref in refs.items() if ref() is not None]
        assert alive == [], f"identity binding pinned {alive}"

    def test_renamed_wrapper_graph_is_collectable(self):
        import gc
        import weakref

        registry = ToolRegistry()

        def _impl() -> dict:
            """Impl."""
            return {"success": True}

        canonical = registry.register(_impl, name="public", capabilities=EXEMPT)
        definition = registry.get("public")
        refs = {
            "registry": weakref.ref(registry),
            "definition": weakref.ref(definition),
            "canonical": weakref.ref(canonical),
            "original": weakref.ref(_impl),
        }

        del registry, definition, canonical, _impl
        gc.collect()

        alive = [name for name, ref in refs.items() if ref() is not None]
        assert alive == [], f"identity binding pinned {alive}"

    def test_service_bound_variant_is_collectable(self):
        """Per-manager factory closures must not accumulate."""
        import gc
        import weakref

        from agentic_cli.tools.registry import bind_tool_identity, get_registry

        definition = get_registry().get("read_file")

        def variant() -> dict:
            """A service-bound variant of a long-lived registered tool."""
            return {"success": True}

        bind_tool_identity(variant, definition)
        ref = weakref.ref(variant)
        del variant
        gc.collect()
        assert ref() is None


class TestRequiresMetadata:
    def test_requires_is_recorded(self):
        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT, requires="kb_manager")
        def kb_thing() -> dict:
            """KB thing."""
            return {"success": True}

        assert registry.get("kb_thing").requires == ("kb_manager",)

    def test_multiple_services_are_recorded_in_order(self):
        registry = ToolRegistry()

        @registry.register(
            capabilities=EXEMPT, requires=("arxiv_source", "kb_manager")
        )
        def composite() -> dict:
            """Composite."""
            return {"success": True}

        assert registry.get("composite").requires == ("arxiv_source", "kb_manager")

    def test_no_requires_defaults_to_empty(self):
        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT)
        def plain() -> dict:
            """Plain."""
            return {"success": True}

        assert registry.get("plain").requires == ()

    def test_unknown_service_key_raises(self):
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="unknown required service"):

            @registry.register(capabilities=EXEMPT, requires="not_a_service")
            def bad() -> dict:
                """Bad."""
                return {"success": True}

    def test_wrong_type_raises(self):
        registry = ToolRegistry()

        with pytest.raises(TypeError, match="requires must be"):

            @registry.register(capabilities=EXEMPT, requires=object())
            def bad() -> dict:
                """Bad."""
                return {"success": True}

    def test_builtin_service_tools_declare_their_services(self):
        """The shipped service-backed tools carry their own metadata."""
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper  # noqa: F401
        from agentic_cli.tools.knowledge_tools import kb_search  # noqa: F401
        from agentic_cli.tools.memory_tools import save_memory  # noqa: F401
        from agentic_cli.tools.registry import get_registry

        reg = get_registry()
        assert reg.get("kb_search").requires == ("kb_manager",)
        assert reg.get("save_memory").requires == ("memory_store",)
        assert reg.get("ingest_arxiv_paper").requires == (
            "arxiv_source",
            "kb_manager",
        )
        assert reg.get("read_file").requires == ()


class TestManagerDetectionUsesRegistry:
    """Managers derive required services from tool metadata, not a name map."""

    def test_detection_reads_declared_requires(self):
        from unittest.mock import MagicMock

        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        from agentic_cli.workflow.config import AgentConfig

        class _Manager(BaseWorkflowManager):
            def _get_state_tools(self):
                return []

            @property
            def backend_type(self) -> str:
                return "test"

            async def _do_initialize(self) -> None:
                return None

            async def process(self, message, user_id, session_id=None):
                raise NotImplementedError

            async def reinitialize(self, model=None, preserve_sessions=True):
                return None

            async def cleanup(self):
                return None

        from agentic_cli.tools.knowledge_tools import kb_search
        from agentic_cli.tools.memory_tools import save_memory

        settings = MagicMock()
        settings.app_name = "test-app"
        mgr = _Manager(
            agent_configs=[AgentConfig(name="a", prompt="p", tools=[kb_search, save_memory])],
            settings=settings,
        )
        assert mgr.required_managers == {"kb_manager", "memory_store"}

    def test_extension_tool_needs_no_framework_edit(self):
        """A tool defined outside the framework still gets its service created."""
        from unittest.mock import MagicMock

        from agentic_cli.tools.registry import register_tool
        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        from agentic_cli.workflow.config import AgentConfig

        @register_tool(capabilities=EXEMPT, requires="sandbox_manager")
        def _extension_tool() -> dict:
            """An app-provided tool the framework has never heard of."""
            return {"success": True}

        class _Manager(BaseWorkflowManager):
            def _get_state_tools(self):
                return []

            @property
            def backend_type(self) -> str:
                return "test"

            async def _do_initialize(self) -> None:
                return None

            async def process(self, message, user_id, session_id=None):
                raise NotImplementedError

            async def reinitialize(self, model=None, preserve_sessions=True):
                return None

            async def cleanup(self):
                return None

        settings = MagicMock()
        settings.app_name = "test-app"
        mgr = _Manager(
            agent_configs=[AgentConfig(name="a", prompt="p", tools=[_extension_tool])],
            settings=settings,
        )
        assert mgr.required_managers == {"sandbox_manager"}

    def test_central_tool_service_map_is_gone(self):
        from agentic_cli.workflow.base_manager import BaseWorkflowManager

        assert not hasattr(BaseWorkflowManager, "_TOOL_SERVICE_MAP")


def _stub_manager_cls():
    from agentic_cli.workflow.base_manager import BaseWorkflowManager

    class _Manager(BaseWorkflowManager):
        def _get_state_tools(self):
            return []

        @property
        def backend_type(self) -> str:
            return "test"

        async def _do_initialize(self) -> None:
            return None

        async def process(self, message, user_id, session_id=None):
            raise NotImplementedError

        async def reinitialize(self, model=None, preserve_sessions=True):
            return None

        async def cleanup(self):
            return None

    return _Manager


class TestAssemblyRequiresBoundIdentity:
    """Assembly must key on the bound definition, never on a matching name.

    ``lookup_definition()`` fell back to ``registry.get(tool.__name__)``, so a
    plain callable an application happened to name ``kb_search`` was handed the
    framework's service-bound variant, had its services constructed, and was
    wrapped as long-running — all for a function the registry never issued and
    that the permission engine then (correctly) denied.
    """

    @staticmethod
    def _manager(tools):
        from unittest.mock import MagicMock

        from agentic_cli.workflow.config import AgentConfig

        settings = MagicMock()
        settings.app_name = "test-app"
        config = AgentConfig(
            name="a", prompt="p", tools=list(tools), include_state_tools=False
        )
        return _stub_manager_cls()(agent_configs=[config], settings=settings), config

    def test_raw_same_name_callable_declares_no_services(self):
        def kb_search(query: str) -> dict:
            """An application's own function that happens to share a name."""
            return {"success": True}

        mgr, _ = self._manager([kb_search])
        assert mgr.required_managers == set()

    def test_raw_same_name_callable_is_not_substituted(self):
        def kb_search(query: str) -> dict:
            """Impostor."""
            return {"success": True}

        mgr, config = self._manager([kb_search])
        service_variant = lambda query: {"success": True}  # noqa: E731
        built = mgr._build_tools(config, service_map={"kb_search": service_variant})

        assert built == [kb_search], "an unregistered callable was replaced"

    def test_raw_same_name_callable_is_not_wrapped_long_running(self):
        pytest.importorskip("google.adk")
        from unittest.mock import MagicMock

        from google.adk.tools import LongRunningFunctionTool

        from agentic_cli.tools.registry import get_registry, register_tool
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        if get_registry().get("_lr_reference_tool") is None:

            @register_tool(capabilities=EXEMPT, long_running=True)
            def _lr_reference_tool() -> dict:
                """A genuine long-running tool."""
                return {"success": True}

        def _lr_reference_tool() -> dict:  # noqa: F811 - deliberate collision
            """Impostor."""
            return {"success": True}

        mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
        mgr._settings = MagicMock()
        wrapped = mgr._wrap_long_running([_lr_reference_tool])

        assert wrapped == [_lr_reference_tool]
        assert not isinstance(wrapped[0], LongRunningFunctionTool)

    async def test_raw_same_name_callable_is_denied_at_permission_time(self, tmp_path):
        pytest.importorskip("google.adk")
        from google.adk.tools import FunctionTool

        from agentic_cli.config import BaseSettings
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin
        from agentic_cli.workflow.permissions import PermissionEngine
        from agentic_cli.workflow.permissions.store import PermissionContext
        from agentic_cli.workflow.service_registry import (
            PERMISSION_ENGINE,
            set_service_registry,
        )

        def kb_search(query: str) -> dict:
            """Impostor."""
            return {"success": True}

        class _Stub:
            async def request_user_input(self, request):
                return "deny"

        engine = PermissionEngine(
            settings=BaseSettings(google_api_key="test"),
            workflow=_Stub(),
            ctx=PermissionContext(workdir=tmp_path, home=tmp_path),
        )
        token = set_service_registry({PERMISSION_ENGINE: engine})
        try:
            result = await PermissionPlugin().before_tool_callback(
                tool=FunctionTool(func=kb_search), tool_args={}, tool_context=None
            )
        finally:
            token.var.reset(token)
        assert isinstance(result, dict) and result["success"] is False

    def test_tool_from_another_registry_is_left_alone(self):
        """A private registry is not the framework's registry."""
        other = ToolRegistry()

        @other.register(
            capabilities=EXEMPT, requires="sandbox_manager", long_running=True
        )
        def foreign_tool() -> dict:
            """Registered somewhere else entirely."""
            return {"success": True}

        mgr, config = self._manager([foreign_tool])
        assert mgr.required_managers == set()
        assert mgr._build_tools(config, service_map={}) == [foreign_tool]

    async def test_tool_from_another_registry_is_denied(self, tmp_path):
        pytest.importorskip("google.adk")
        from google.adk.tools import FunctionTool

        from agentic_cli.config import BaseSettings
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin
        from agentic_cli.workflow.permissions import PermissionEngine
        from agentic_cli.workflow.permissions.store import PermissionContext
        from agentic_cli.workflow.service_registry import (
            PERMISSION_ENGINE,
            set_service_registry,
        )

        other = ToolRegistry()

        @other.register(capabilities=EXEMPT)
        def foreign_exempt_tool() -> dict:
            """EXEMPT — but only according to a registry nobody trusts."""
            return {"success": True}

        class _Stub:
            async def request_user_input(self, request):
                return "deny"

        engine = PermissionEngine(
            settings=BaseSettings(google_api_key="test"),
            workflow=_Stub(),
            ctx=PermissionContext(workdir=tmp_path, home=tmp_path),
        )
        token = set_service_registry({PERMISSION_ENGINE: engine})
        try:
            result = await PermissionPlugin().before_tool_callback(
                tool=FunctionTool(func=foreign_exempt_tool),
                tool_args={},
                tool_context=None,
            )
        finally:
            token.var.reset(token)
        assert isinstance(result, dict) and result["success"] is False, (
            "a private registry's EXEMPT declaration was honoured"
        )

    def test_string_registry_reference_still_resolves(self):
        """A config may name a tool; that is a registry lookup, not a guess."""
        from unittest.mock import MagicMock

        from agentic_cli.tools.factories import make_kb_tools
        from agentic_cli.tools.knowledge_tools import kb_search  # noqa: F401

        mgr, config = self._manager(["kb_search"])
        assert mgr.required_managers == {"kb_manager"}

        service_map = {t.__name__: t for t in make_kb_tools(MagicMock())}
        built = mgr._build_tools(config, service_map=service_map)
        assert built == [service_map["kb_search"]]


class TestDirectRegisterOriginalCallable:
    """``register(func, name=..., ...)`` — the caller may keep using ``func``.

    Assembly used to key on ``func.__name__``, which for a renamed tool is the
    private implementation name. The tool's declared services were then never
    created, ``long_running`` never applied, and the model saw the private name.
    """

    # The original callable, registered once for the whole module: a name may
    # only be registered once (see TestDuplicateNamePolicy).
    _original: "Callable | None" = None

    @classmethod
    def _register_renamed(cls):
        from agentic_cli.tools.registry import get_registry, register_tool

        if cls._original is None:

            def _private_impl(query: str) -> dict:
                """A renamed, service-backed, long-running tool."""
                return {"success": True}

            register_tool(
                _private_impl,
                name="renamed_public_tool",
                capabilities=EXEMPT,
                requires="sandbox_manager",
                long_running=True,
            )
            cls._original = _private_impl

        assert get_registry().get("renamed_public_tool") is not None
        assert cls._original.__name__ == "_private_impl"
        return cls._original

    def test_declared_services_are_detected(self):
        from unittest.mock import MagicMock

        from agentic_cli.workflow.config import AgentConfig

        original = self._register_renamed()
        settings = MagicMock()
        settings.app_name = "test-app"
        mgr = _stub_manager_cls()(
            agent_configs=[AgentConfig(name="a", prompt="p", tools=[original])],
            settings=settings,
        )
        assert mgr.required_managers == {"sandbox_manager"}

    def test_assembled_tool_carries_the_registered_name(self):
        from unittest.mock import MagicMock

        from agentic_cli.workflow.config import AgentConfig

        original = self._register_renamed()
        settings = MagicMock()
        settings.app_name = "test-app"
        config = AgentConfig(
            name="a", prompt="p", tools=[original], include_state_tools=False
        )
        mgr = _stub_manager_cls()(agent_configs=[config], settings=settings)

        built = mgr._build_tools(config, service_map={})
        assert [getattr(t, "__name__", "") for t in built] == ["renamed_public_tool"]

    def test_long_running_wrapping_uses_identity(self):
        pytest.importorskip("google.adk")
        from unittest.mock import MagicMock

        from google.adk.tools import LongRunningFunctionTool

        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        original = self._register_renamed()
        mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
        mgr._settings = MagicMock()

        wrapped = mgr._wrap_long_running([original])
        assert isinstance(wrapped[0], LongRunningFunctionTool)


class TestRequiresAreConstructible:
    """Only services a manager can actually build may be declared."""

    def test_user_kb_manager_is_not_declarable(self):
        """It is created with kb_manager, never on its own — declaring it
        validated and then provided nothing."""
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="user_kb_manager") as exc:

            @registry.register(capabilities=EXEMPT, requires="user_kb_manager")
            def _tool() -> dict:
                """Tool."""
                return {"success": True}

        assert "kb_manager" in str(exc.value)

    def test_kb_manager_creates_both_scopes(self):
        """The documented replacement really provides the user-scoped KB too."""
        from unittest.mock import MagicMock

        from agentic_cli.workflow.service_registry import (
            KB_MANAGER,
            USER_KB_MANAGER,
        )

        from tests.conftest import MockContext

        with MockContext(google_api_key="k", knowledge_base_use_mock=True) as ctx:
            from agentic_cli.workflow.base_manager import BaseWorkflowManager

            class _Manager(BaseWorkflowManager):
                def _get_state_tools(self):
                    return []

                @property
                def backend_type(self) -> str:
                    return "test"

                async def _do_initialize(self) -> None:
                    return None

                async def process(self, message, user_id, session_id=None):
                    raise NotImplementedError

                async def reinitialize(self, model=None, preserve_sessions=True):
                    return None

                async def cleanup(self):
                    return None

            mgr = _Manager(agent_configs=[], settings=ctx.settings)
            mgr._required_managers = {"kb_manager"}
            mgr._ensure_managers_initialized()

            assert mgr.services.get(KB_MANAGER) is not None
            assert mgr.services.get(USER_KB_MANAGER) is not None

    def test_always_present_services_are_not_declarable(self):
        registry = ToolRegistry()
        for key in ("permission_engine", "workflow"):
            with pytest.raises(ValueError, match="always available"):

                @registry.register(capabilities=EXEMPT, requires=key)
                def _tool() -> dict:
                    """Tool."""
                    return {"success": True}

    def test_empty_string_requires_is_rejected(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="non-empty"):

            @registry.register(capabilities=EXEMPT, requires="")
            def _tool() -> dict:
                """Tool."""
                return {"success": True}

    def test_non_string_element_is_rejected(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="non-empty"):

            @registry.register(capabilities=EXEMPT, requires=("kb_manager", None))
            def _tool() -> dict:
                """Tool."""
                return {"success": True}

    def test_every_declarable_key_is_constructible(self):
        """The declarable set must not drift from what the manager can build."""
        import inspect

        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        from agentic_cli.workflow.service_registry import KNOWN_SERVICE_KEYS

        source = inspect.getsource(BaseWorkflowManager._build_services_into)
        for key in KNOWN_SERVICE_KEYS:
            assert f'"{key}" in self._required_managers' in source, (
                f"{key} is declarable but _build_services_into never constructs it"
            )
