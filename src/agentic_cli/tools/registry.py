"""Tool registry for standardized tool management.

Provides:
- ToolDefinition: Metadata-rich tool definition
- ToolRegistry: Registry for tool discovery and management
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import functools
import inspect
import weakref

from agentic_cli.workflow.permissions.capabilities import (
    Capability,
    CapabilitiesSpec,
    EXEMPT,
)
from agentic_cli.workflow.permissions.capabilities import _CapabilityExempt
from agentic_cli.workflow.service_registry import (
    KNOWN_SERVICE_KEYS,
    SERVICE_KEY_HINTS,
)


class ToolCategory(Enum):
    """Categories for organizing tools.

    Categories are organized by primary function:
    - READ: Read-only file operations (safe)
    - WRITE: File modification operations (require caution)
    - NETWORK: Network and web operations
    - EXECUTION: Shell and code execution
    - PLANNING: Task and workflow management
    - MEMORY: State and context management
    - KNOWLEDGE: External knowledge access
    - INTERACTION: Human-in-the-loop operations
    """

    # File operations (split by safety)
    READ = "read"  # read_file, grep, glob, diff
    WRITE = "write"  # write_file, edit_file

    # Network operations
    NETWORK = "network"  # web_search, web_fetch, api calls

    # Execution (potentially dangerous)
    EXECUTION = "execution"  # shell, python executor

    # Planning and task management
    PLANNING = "planning"  # save_plan, get_plan

    # Memory and state
    MEMORY = "memory"  # remember, recall, search_memory

    # External knowledge
    KNOWLEDGE = "knowledge"  # arxiv, knowledge_base

    # Human interaction
    INTERACTION = "interaction"  # ask_clarification

    OTHER = "other"


@dataclass
class ToolDefinition:
    """Metadata-rich tool definition.

    ``name`` is the tool's canonical identity: it is what the model calls, what
    permission rules match, and what service detection keys on. ``func`` is
    guaranteed to expose it as ``__name__`` (see :meth:`ToolRegistry.register`)
    so backends that derive a tool name from the callable agree with it.

    Attributes:
        name: Tool name (defaults to function name)
        description: Human-readable description
        category: Tool category for organization (READ, WRITE, NETWORK, etc.)
        capabilities: Capability declarations for the permission engine
        requires: Service keys the tool needs at runtime (see
            ``service_registry.KNOWN_SERVICE_KEYS``); managers create exactly
            these, lazily.
        is_async: Whether the tool is async
        func: The backend-neutral implementation, or **None** for a tool that
            only exists as backend-native variants (see ``variants``). A bare
            name that resolves to such a tool is an error, not a guess.
        variants: Backend-native implementations declared with
            ``register(..., variant_of=name)``. They share this tool's identity
            and permission metadata, and may differ in signature and docstring
            — that is what makes them native.
    """

    name: str
    description: str
    func: Callable[..., Any] | None
    capabilities: CapabilitiesSpec
    category: ToolCategory = ToolCategory.OTHER
    requires: tuple[str, ...] = ()
    is_async: bool = False
    long_running: bool = False  # tool starts a background job; see tools/jobs/
    variants: tuple[Callable[..., Any], ...] = ()

    def __post_init__(self):
        """Infer is_async from function."""
        if inspect.iscoroutinefunction(self.func):
            self.is_async = True


def bind_tool_identity(obj: Any, definition: "ToolDefinition") -> None:
    """Record, in the default registry, that ``obj`` *is* ``definition``'s tool.

    Called by the registry on registration, and by the framework whenever it
    hands a backend something other than the registered callable for the same
    tool: a service-bound factory variant, the canonical-name wrapper, or a
    backend-native tool object the framework itself constructed (ADK skill
    tools). Binding is the only way to acquire capabilities — an object the
    framework never issued stays unbound and is gated as unregistered.

    Bindings made by some *other* :class:`ToolRegistry` are deliberately
    invisible here: the framework trusts the registry it owns, not one an
    application happens to construct.
    """
    _default_registry.bind_identity(obj, definition)


def identify_tool(obj: Any) -> "ToolDefinition | None":
    """Resolve what ``obj`` is, per the default registry, by object identity.

    Strictly this object: no name lookup, no equality, no attribute traversal.
    Callers that legitimately need to look *inside* a backend wrapper must
    unwrap it themselves, and only for wrapper types they trust (see
    ``workflow/adk/permission_plugin.py``).

    Returns:
        The bound ``ToolDefinition``, or None when this object was never issued
        by the default registry — callers must treat None as "not a tool".
    """
    return _default_registry.identify(obj)


def _validate_capabilities(caps: Any, tool_name: str) -> CapabilitiesSpec:
    """Validate and return a capabilities value for a tool registration.

    - ``EXEMPT`` (or any ``_CapabilityExempt`` instance) passes through unchanged.
    - A non-empty ``list`` of ``Capability`` instances is returned as-is.
    - An empty list raises ``ValueError`` (use ``EXEMPT`` to opt out explicitly).
    - Any other type raises ``TypeError``.
    """
    if isinstance(caps, _CapabilityExempt):
        return caps
    if isinstance(caps, list):
        if not caps:
            raise ValueError(
                f"Tool {tool_name!r}: capabilities=[] is not allowed. "
                "Use capabilities=EXEMPT to opt out explicitly."
            )
        for item in caps:
            if not isinstance(item, Capability):
                raise TypeError(
                    f"Tool {tool_name!r}: capabilities list items must be "
                    f"Capability instances, got {type(item)!r}."
                )
        return caps
    raise TypeError(
        f"Tool {tool_name!r}: capabilities must be EXEMPT or a list of Capability "
        f"instances, got {type(caps)!r}."
    )


def _validate_requires(requires: Any, tool_name: str) -> tuple[str, ...]:
    """Validate declared service keys against the constructible service keys.

    A key the manager cannot construct would be a silent no-op: nothing would
    be created and the tool would fail at call time with a missing service. The
    declarable set is therefore exactly what
    ``_ensure_managers_initialized`` knows how to build.
    """
    if requires is None:
        return ()
    if isinstance(requires, str):
        requires = (requires,)
    if not isinstance(requires, (list, tuple, set, frozenset)):
        raise TypeError(
            f"Tool {tool_name!r}: requires must be a string or a sequence of "
            f"service keys, got {type(requires)!r}."
        )
    keys = tuple(dict.fromkeys(requires))  # de-duplicate, keep order
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                f"Tool {tool_name!r}: every requires entry must be a non-empty "
                f"service-key string, got {key!r}."
            )
    unknown = [k for k in keys if k not in KNOWN_SERVICE_KEYS]
    if unknown:
        hints = " ".join(
            SERVICE_KEY_HINTS[k] for k in sorted(unknown) if k in SERVICE_KEY_HINTS
        )
        raise ValueError(
            f"Tool {tool_name!r}: unknown required service(s): "
            f"{', '.join(sorted(unknown))}. Known services: "
            f"{', '.join(sorted(KNOWN_SERVICE_KEYS))}."
            + (f" {hints}" if hints else "")
        )
    return keys


def _variant_sort_key(variant: Callable[..., Any]) -> tuple[str, str]:
    """Order variants by where they are defined, not by import order."""
    target = getattr(variant, "__wrapped__", variant)
    return (
        getattr(target, "__module__", "") or "",
        getattr(target, "__qualname__", "") or "",
    )


def _ordered_variants(
    variants: "tuple[Callable[..., Any], ...]",
) -> "tuple[Callable[..., Any], ...]":
    """Deterministic variant order: module-qualified, import-order independent."""
    return tuple(sorted(variants, key=_variant_sort_key))


def _same_declaration(
    existing: "ToolDefinition",
    capabilities: CapabilitiesSpec,
    requires: tuple[str, ...],
    category: "ToolCategory",
    long_running: bool,
) -> bool:
    """Whether a re-registration describes the *same tool*, differently implemented.

    Everything the framework acts on — what it may do, what it needs, whether
    it is long-running — must match. Only the callable may differ, which is the
    legitimate case of one tool with two backend-native implementations.
    """
    return (
        existing.capabilities == capabilities
        and existing.requires == requires
        and existing.category == category
        and existing.long_running == long_running
    )


def _with_canonical_name(func: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Return a callable whose ``__name__`` is the registered tool name.

    Backends derive the model-visible tool name from ``func.__name__`` (ADK
    does), while permission lookup, service detection and tool assembly key on
    the registry name. When ``register_tool(name=...)`` renames a tool those two
    diverge, and the permission engine then looks up a name that isn't
    registered. Wrapping keeps a single identity.

    The wrapper preserves the signature (via ``__wrapped__``), docstring,
    annotations and async-ness, so schema generation and the
    ``{"success": bool}`` return contract are unaffected.
    """
    if getattr(func, "__name__", None) == name:
        return func

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

    else:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper


class ToolRegistry:
    """Registry for managing and discovering tools.

    Provides:
    - Tool registration with metadata
    - Tool lookup by name or category
    - Tool list generation for agents
    - **Identity**: which exact objects *are* the tools it issued

    Identity is owned per registry, not globally. Permission gating and tool
    assembly ask the framework's default registry (``get_registry()``), so a
    tool registered into some other ``ToolRegistry`` is not one of *its* tools:
    it is left as-is during assembly and denied at permission time. Keeping the
    map on the instance is also what lets a short-lived registry — with its
    definitions and their closures — be garbage collected.
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        # id(obj) -> (weak ref to obj, the definition it implements).
        #
        # Not keyed by the object itself: a ``WeakKeyDictionary`` resolves by
        # ``hash``/``__eq__``, which any object can define to collide with a
        # registered callable. Keyed by ``id`` instead, with every hit
        # confirmed by ``is`` against the weak reference, so a recycled address
        # cannot inherit a dead object's capabilities. The weak reference also
        # drops the entry when the object dies, so per-manager service-bound
        # variants do not accumulate.
        self._identity: dict[int, tuple[weakref.ref, ToolDefinition]] = {}
        # Callables that *were* one of this registry's tools until a
        # replace=True took the name over. Weak, and distinct from "never
        # registered" — see _retire_identities.
        self._retired: "weakref.WeakSet[Any]" = weakref.WeakSet()

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        category: ToolCategory = ToolCategory.OTHER,
        capabilities: CapabilitiesSpec,
        requires: str | tuple[str, ...] | list[str] | None = None,
        long_running: bool = False,
        replace: bool = False,
        variant_of: str | None = None,
    ) -> Callable[..., Any]:
        """Register a tool function.

        Can be used as a decorator:
            @registry.register(category=ToolCategory.READ, capabilities=[Capability(...)])
            def my_tool(query: str) -> dict:
                ...

        Or called directly:
            registry.register(my_tool, category=ToolCategory.READ, capabilities=EXEMPT)

        ``requires=`` declares the service keys the tool needs (e.g.
        ``requires="kb_manager"``); workflow managers read that metadata to
        create exactly those services, lazily. Unknown keys raise.

        ``long_running=True`` marks a tool that starts a background job (it should
        return a ``job_id`` and delegate to ``JobManager``); see ``tools/jobs/``.

        A name means **one tool**. Registering it again raises ``ValueError``
        rather than taking it over: the two definitions would disagree about
        what the tool is, and consumers keyed on the name (permission rules,
        the service-tool map) would follow whichever they happened to ask.
        Matching capabilities are *not* grounds for sharing a name either —
        they say nothing about what the model sees or what the callable does.

        Sharing is declared, never inferred:

        - ``variant_of="<name>"`` registers a **backend-native variant** of an
          already-declared tool (see :func:`declare_tool`). It binds to that
          tool's identity and permission metadata, and may differ in signature
          and docstring, which is the whole point. Its declaration
          (capabilities/requires/category/long_running) must match exactly.
        - ``replace=True`` takes the name over deliberately; the previous
          definition's identities are retired, so its callables resolve to
          nothing and are denied rather than inheriting the replacement's
          capabilities.

        Returns:
            The callable to bind at the definition site. When ``name`` renames
            the tool this is a thin wrapper carrying the canonical
            ``__name__``, so every consumer sees one identity.

        Raises:
            ValueError: If the name is already registered and ``replace`` is
                False (or the capabilities/requires declarations are invalid).
        """

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = variant_of or name or f.__name__
            tool_desc = description or (f.__doc__ or "").split("\n")[0].strip()
            validated_caps = _validate_capabilities(capabilities, tool_name)
            validated_requires = _validate_requires(requires, tool_name)
            canonical = _with_canonical_name(f, tool_name)

            existing = self._tools.get(tool_name)

            if variant_of is not None:
                if existing is None:
                    raise ValueError(
                        f"Tool {variant_of!r} is not declared, so "
                        f"{f.__qualname__!r} cannot be a variant of it. Declare "
                        "the tool first with declare_tool()."
                    )
                if not _same_declaration(
                    existing,
                    validated_caps,
                    validated_requires,
                    category,
                    long_running,
                ):
                    raise ValueError(
                        f"Variant of {variant_of!r} declares a different "
                        "declaration than the tool it implements: capabilities, "
                        "requires, category and long_running must match exactly "
                        "(a variant shares the tool's permission contract)."
                    )
                existing.variants = _ordered_variants(
                    existing.variants + (canonical,)
                )
                self.bind_identity(canonical, existing)
                if canonical is not f:
                    self.bind_identity(f, existing)
                return canonical

            if existing is not None:
                if not replace:
                    raise ValueError(
                        f"Tool {tool_name!r} is already registered (by "
                        f"{getattr(existing.func, '__qualname__', existing.func)!r}). "
                        "Pick a different name, pass variant_of= for a "
                        "backend-native implementation of the same tool, or "
                        "replace=True to take it over deliberately."
                    )
                self._retire_identities(existing)

            definition = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                func=canonical,
                capabilities=validated_caps,
                category=category,
                requires=validated_requires,
                long_running=long_running,
            )

            self._tools[tool_name] = definition
            # Bind identity for every callable that legitimately *is* this tool:
            # the canonical (possibly renamed) wrapper handed to backends, and
            # the original, which a caller of ``register(func, name=...)`` may
            # keep using.
            self.bind_identity(canonical, definition)
            if canonical is not f:
                self.bind_identity(f, definition)
            return canonical

        if func is not None:
            return decorator(func)
        return decorator

    def bind_identity(self, obj: Any, definition: "ToolDefinition") -> None:
        """Record that ``obj`` *is* the tool ``definition`` describes.

        Objects that cannot be weak-referenced (rare; some C callables) are
        skipped rather than bound by id alone, since a recycled id would
        otherwise hand a later object someone else's capabilities. They resolve
        to no definition, which fails closed.
        """
        key = id(obj)
        identity = self._identity

        def _drop(ref: "weakref.ref") -> None:
            # Only clear our own entry: by the time this runs the id may
            # already have been re-used and re-bound by a live object.
            entry = identity.get(key)
            if entry is not None and entry[0] is ref:
                del identity[key]

        try:
            ref = weakref.ref(obj, _drop)
        except TypeError:  # not weak-referenceable
            return
        identity[key] = (ref, definition)

    def declare(
        self,
        name: str,
        *,
        description: str,
        capabilities: CapabilitiesSpec,
        category: ToolCategory = ToolCategory.OTHER,
        requires: str | tuple[str, ...] | list[str] | None = None,
        long_running: bool = False,
    ) -> "ToolDefinition":
        """Declare a tool that exists only as backend-native variants.

        The declaration owns the name and the permission contract; each backend
        then registers its own implementation with
        ``register(..., variant_of=name)``. Neither backend can win the name by
        importing first, and a bare-name reference resolves deterministically —
        to nothing, because there is no backend-neutral implementation to give.

        Idempotent: re-declaring the same contract returns the existing
        definition (module import order must not matter), while a conflicting
        re-declaration raises.

        Returns:
            The declared ``ToolDefinition`` (``func`` is None).

        Raises:
            ValueError: If the name already has a different declaration.
        """
        validated_caps = _validate_capabilities(capabilities, name)
        validated_requires = _validate_requires(requires, name)

        existing = self._tools.get(name)
        if existing is not None:
            if (
                existing.func is not None
                or existing.description != description
                or not _same_declaration(
                    existing, validated_caps, validated_requires, category, long_running
                )
            ):
                raise ValueError(
                    f"Tool {name!r} is already registered with a different "
                    "declaration; declare_tool() cannot take it over."
                )
            return existing

        definition = ToolDefinition(
            name=name,
            description=description,
            func=None,
            capabilities=validated_caps,
            category=category,
            requires=validated_requires,
            long_running=long_running,
        )
        self._tools[name] = definition
        return definition

    def _retire_identities(self, definition: "ToolDefinition") -> None:
        """Unbind every object that used to *be* ``definition``'s tool.

        A replaced tool must not keep its capabilities through a callable the
        application still holds: those bindings now describe a tool this
        registry no longer has. Retired objects resolve to nothing, which is
        denied at permission time and left alone during assembly.

        They are also remembered (weakly) as *retired*, which is different from
        "never registered": a backend's state-tool variant that has been
        replaced must not be auto-injected, while an application's own
        unregistered callable of the same name is nobody's business but its
        author's.
        """
        stale = [
            key for key, (_, bound) in self._identity.items() if bound is definition
        ]
        for key in stale:
            ref, _ = self._identity.pop(key)
            obj = ref()
            if obj is not None:
                try:
                    self._retired.add(obj)
                except TypeError:  # pragma: no cover - unhashable
                    pass
        for variant in definition.variants:
            try:
                self._retired.add(variant)
            except TypeError:  # pragma: no cover - unhashable
                pass

    def is_retired(self, obj: Any) -> bool:
        """Whether ``obj`` implemented a tool this registry has since replaced."""
        try:
            return obj in self._retired
        except TypeError:  # pragma: no cover - unhashable
            return False

    def canonical_for(self, obj: Any) -> Any:
        """The canonical-named callable for a tool object this registry issued.

        ``register(func, name=...)`` and ``register(func, variant_of=...)``
        both hand back a wrapper carrying the registered name while the caller
        may keep the original. Assembly must give the backend the wrapper, so
        the model-visible name is the tool's identity — and for a declared tool
        the canonical form is *that variant's* wrapper, since the declaration
        itself has no implementation to substitute.

        Returns ``obj`` unchanged when it is already canonical, or when this
        registry did not issue it.
        """
        definition = self.identify(obj)
        if definition is None:
            return obj
        if getattr(obj, "__name__", None) == definition.name:
            return obj
        for candidate in (definition.func, *definition.variants):
            if candidate is not None and getattr(candidate, "__wrapped__", None) is obj:
                return candidate
        return definition.func if definition.func is not None else obj

    def identify(self, obj: Any) -> "ToolDefinition | None":
        """The definition ``obj`` was bound to *in this registry*, or None."""
        if obj is None:
            return None
        entry = self._identity.get(id(obj))
        if entry is None:
            return None
        ref, definition = entry
        if ref() is not obj:  # id recycled after the bound object died
            return None
        return definition

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        """List tools by category."""
        return [t for t in self._tools.values() if t.category == category]

    def get_functions(self) -> list[Callable[..., Any]]:
        """Get all tool functions (for passing to agents).

        Declared-only tools are skipped: they have no backend-neutral
        implementation to hand out (see :meth:`declare`).
        """
        return [t.func for t in self._tools.values() if t.func is not None]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Global registry instance
_default_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the default tool registry."""
    return _default_registry


def declare_tool(
    name: str,
    *,
    description: str,
    capabilities: CapabilitiesSpec,
    category: ToolCategory = ToolCategory.OTHER,
    requires: str | tuple[str, ...] | list[str] | None = None,
    long_running: bool = False,
    registry: "ToolRegistry | None" = None,
) -> "ToolDefinition":
    """Declare a tool implemented only by backend-native variants.

    See :meth:`ToolRegistry.declare`. Defaults to the framework registry.
    """
    # `registry or ...` would fall through for an *empty* registry: the class
    # defines __len__, so a registry with no tools yet is falsy.
    target = _default_registry if registry is None else registry
    return target.declare(
        name,
        description=description,
        capabilities=capabilities,
        category=category,
        requires=requires,
        long_running=long_running,
    )


def register_tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    category: ToolCategory = ToolCategory.OTHER,
    capabilities: CapabilitiesSpec,
    requires: str | tuple[str, ...] | list[str] | None = None,
    long_running: bool = False,
    replace: bool = False,
    variant_of: str | None = None,
) -> Callable[..., Any]:
    """Register a tool with the default registry.

    ``capabilities`` is a required keyword argument. Pass ``EXEMPT`` to opt out
    of the permission engine, or a list of ``Capability`` instances to declare
    the resources this tool accesses.

    ``requires`` declares which **framework-provided** services the tool needs
    (``"kb_manager"``, ``"memory_store"``, … — the full set is
    ``service_registry.KNOWN_SERVICE_KEYS``), so managers create exactly those,
    lazily. A downstream tool may request any of them without editing the
    framework; anything else raises, because nothing would construct it. There
    is no mechanism for registering new service *types*.

    ``long_running=True`` marks a tool that starts a background job (see
    ``tools/jobs/``).

    A name may be registered once; a collision raises ``ValueError`` unless
    ``variant_of=`` (a backend-native implementation of a declared tool) or
    ``replace=True`` is passed deliberately (see ``ToolRegistry.register``).
    """

    def _outer(f: Callable[..., Any]) -> Callable[..., Any]:
        return _default_registry.register(
            f,
            name=name,
            description=description,
            category=category,
            capabilities=capabilities,
            requires=requires,
            long_running=long_running,
            replace=replace,
            variant_of=variant_of,
        )

    if func is not None:
        return _outer(func)
    return _outer
