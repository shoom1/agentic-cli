"""The services a workflow manager builds for its tools: one table, and the set one manager holds.

A tool declares the services it needs (``register_tool(requires=...)``). A
manager builds the ones its agents' tools declare, publishes them for the
tools (``service_registry.get_service``), hands each tool that needs one a
version bound to it (``tool_map``) and closes them when it releases. Each
service is one :class:`ServiceSpec` in :data:`SERVICES`: its key, how it is
built and closed, the tools it gives, and whether a tool may declare it.
Adding a service is adding a row: ``service_registry.KNOWN_SERVICE_KEYS``
(what a tool may declare) and the closers are read from the table.

:class:`ServiceSet` is what one manager holds: the services it publishes,
which of them a model switch carries (``BaseWorkflowManager._keeping``), and
which of them it holds but does not own. A manager can lend its services to
another (:class:`ServiceLoan`), which owns them once they are given: each
service's references to its owner then move to the new one (``rebind``).

The builders and tool makers import what they build when they run: tools
import ``service_registry`` (and so this module) when they register.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.workflow.base_manager import BaseWorkflowManager

logger = Loggers.workflow()

# ---- The registry keys ----

ARXIV_SOURCE = "arxiv_source"
JOB_MANAGER = "job_manager"
KNOWLEDGE_BASE = "knowledge_base"
LLM_SUMMARIZER = "llm_summarizer"
MEMORY_STORE = "memory_store"
PERMISSION_ENGINE = "permission_engine"
SANDBOX_MANAGER = "sandbox_manager"
WORKFLOW = "workflow"

# The embedding service both knowledge base scopes share. Built with the
# knowledge base, never on its own: no tool can declare it.
EMBEDDING_SERVICE = "_embedding_service"

#: How a service is built: called with the manager and the services
#: available so far (those built before it in this build, then the
#: published ones); None means it is not built (it cannot open).
Builder = Callable[["BaseWorkflowManager", Mapping[str, Any]], Any]
#: The tools a service gives: called with the service, every service present
#: (the manager's own ``WORKFLOW`` included) and the manager.
ToolMaker = Callable[[Any, Mapping[str, Any], "BaseWorkflowManager"], Iterable[Callable]]
#: How a service's references to its owner move to a new owner: called with
#: the service, the manager that gave it and the one that owns it now.
Rebinder = Callable[[Any, "BaseWorkflowManager", "BaseWorkflowManager"], None]


@dataclass(frozen=True)
class ServiceSpec:
    """One service: how it is built, closed and turned into tools, and who may ask for it.

    Attributes:
        key: Its registry key.
        build: How it is built (:data:`Builder`).
        close: The synchronous method that releases what it holds (a
            sandbox's ``cleanup``, a job manager's ``close``); None for a
            service that holds nothing to release.
        tools: The service-bound tools it gives (:data:`ToolMaker`), which
            take the place of the module-level tools of the same identity.
        declarable: Whether a tool may name it in ``register_tool(requires=)``.
        always: Built for every manager, whatever the tools declare.
        built_with: The key of the service it is built with (and only with).
        rebind: How its references to its owner move to a new owner
            (:data:`Rebinder`), for a service built holding the manager.
    """

    key: str
    build: Builder
    close: str | None = None
    tools: ToolMaker | None = None
    declarable: bool = False
    always: bool = False
    built_with: str | None = None
    rebind: Rebinder | None = None


# ---- Builders: each imports what it builds, when it runs ----


def _the_manager(manager: "BaseWorkflowManager", available: Mapping[str, Any]) -> Any:
    return manager


def _embedding_service(manager: "BaseWorkflowManager", available: Mapping[str, Any]) -> Any:
    # One service for both knowledge base scopes, so the model loads once.
    from agentic_cli.workflow.memory_services import build_embedding_service

    return build_embedding_service(manager.settings)


def _memory_store(manager: "BaseWorkflowManager", available: Mapping[str, Any]) -> Any:
    from agentic_cli.workflow.memory_services import open_memory_store

    return open_memory_store(manager.settings)  # None: it cannot open, and the memory tools say so


def _knowledge_base(manager: "BaseWorkflowManager", available: Mapping[str, Any]) -> Any:
    from agentic_cli.workflow.memory_services import open_knowledge_base

    return open_knowledge_base(
        manager.settings,
        summarizer=manager,
        embedding_service=available.get(EMBEDDING_SERVICE),
    )


def _sandbox_manager(manager: "BaseWorkflowManager", available: Mapping[str, Any]) -> Any:
    from agentic_cli.tools.sandbox.manager import SandboxManager

    return SandboxManager(manager.settings)


def _job_manager(manager: "BaseWorkflowManager", available: Mapping[str, Any]) -> Any:
    from pathlib import Path

    from agentic_cli.tools.jobs import JobManager

    settings = manager.settings
    # User-scoped so long jobs persist across projects and CLI restarts.
    return JobManager(
        settings,
        base_dir=Path.home() / f".{settings.app_name}" / "jobs",
        max_concurrent=getattr(settings, "max_concurrent_jobs", 4),
    )


def _arxiv_source(manager: "BaseWorkflowManager", available: Mapping[str, Any]) -> Any:
    from agentic_cli.tools.arxiv_source import ArxivSearchSource

    return ArxivSearchSource()


def _permission_engine(manager: "BaseWorkflowManager", available: Mapping[str, Any]) -> Any:
    from pathlib import Path

    from agentic_cli.tools.exec_policy import uncontained_exec_capabilities
    from agentic_cli.workflow.permissions import PermissionContext, PermissionEngine

    settings = manager.settings
    ctx = PermissionContext(workdir=Path.cwd(), home=Path.home(), app_name=settings.app_name)
    # Code that would run on the host without a sandbox is approved per
    # call, and so is a memory write once the conversation has read outside
    # content; the engine asks which at each check.
    return PermissionEngine(
        settings=settings,
        workflow=manager,
        ctx=ctx,
        uncontained=uncontained_exec_capabilities,
        per_call=manager._per_call_capabilities,
    )


# ---- Rebinders: what a service holds of its owner, moved to a new one ----


def _rebind_permission_engine(engine: Any, old: Any, new: Any) -> None:
    # Its prompts reach the new owner's user; its per-call say is the new owner's.
    engine.set_workflow(new, per_call=new._per_call_capabilities)


def _rebind_knowledge_base(kb: Any, old: Any, new: Any) -> None:
    # A scope that summarized through the old owner summarizes through the new
    # one (none does with knowledge_base_summarize off).
    for scope in kb.scopes:
        manager = kb.manager(scope)
        if manager.summarizer is old:
            manager.summarizer = new


# ---- Tool makers ----


def _memory_tools(store: Any, services: Mapping[str, Any], manager: Any) -> Iterable[Callable]:
    from agentic_cli.tools.factories import make_memory_tools

    return make_memory_tools(store)


def _kb_tools(kb: Any, services: Mapping[str, Any], manager: Any) -> Iterable[Callable]:
    from agentic_cli.tools.factories import make_kb_tools

    return make_kb_tools(kb)


def _webfetch_tool(summarizer: Any, services: Mapping[str, Any], manager: Any) -> Iterable[Callable]:
    from agentic_cli.tools.factories import make_webfetch_tool

    return [make_webfetch_tool(summarizer)]


def _sandbox_tool(sandbox: Any, services: Mapping[str, Any], manager: Any) -> Iterable[Callable]:
    from agentic_cli.tools.factories import make_sandbox_tool

    return [make_sandbox_tool(sandbox, manager)]


def _arxiv_tools(source: Any, services: Mapping[str, Any], manager: Any) -> Iterable[Callable]:
    from agentic_cli.tools.factories import make_arxiv_tools, make_ingest_arxiv_tool

    tools = list(make_arxiv_tools(source))
    knowledge_base = services.get(KNOWLEDGE_BASE)
    if knowledge_base is not None:  # ingest_arxiv_paper composes both services
        tools.append(make_ingest_arxiv_tool(source, knowledge_base))
    return tools


def _interaction_tools(workflow: Any, services: Mapping[str, Any], manager: Any) -> Iterable[Callable]:
    from agentic_cli.tools.factories import make_interaction_tools

    return make_interaction_tools(workflow)


#: Every service a manager can build, in the order it builds them (and
#: closes them).
SERVICES: tuple[ServiceSpec, ...] = (
    ServiceSpec(EMBEDDING_SERVICE, _embedding_service, built_with=KNOWLEDGE_BASE),
    ServiceSpec(MEMORY_STORE, _memory_store, tools=_memory_tools, declarable=True),
    ServiceSpec(
        # A database connection per scope.
        KNOWLEDGE_BASE,
        _knowledge_base,
        close="close",
        tools=_kb_tools,
        declarable=True,
        rebind=_rebind_knowledge_base,
    ),
    ServiceSpec(LLM_SUMMARIZER, _the_manager, tools=_webfetch_tool, declarable=True),
    ServiceSpec(
        SANDBOX_MANAGER, _sandbox_manager, close="cleanup", tools=_sandbox_tool, declarable=True
    ),
    ServiceSpec(JOB_MANAGER, _job_manager, close="close", declarable=True),
    ServiceSpec(ARXIV_SOURCE, _arxiv_source, tools=_arxiv_tools, declarable=True),
    # Every agent may need it.
    ServiceSpec(
        PERMISSION_ENGINE, _permission_engine, always=True, rebind=_rebind_permission_engine
    ),
    # The interaction tools: the manager is always there for them.
    ServiceSpec(WORKFLOW, _the_manager, tools=_interaction_tools, always=True),
)

#: The services that hold something to release, and the synchronous method
#: that releases it, in the table's order.
SERVICE_CLOSERS: tuple[tuple[str, str], ...] = tuple(
    (spec.key, spec.close) for spec in SERVICES if spec.close is not None
)


def close_services(
    services: Mapping[str, Any], closers: Iterable[tuple[str, str]] = SERVICE_CLOSERS
) -> None:
    """Release what ``services`` hold, one closer at a time. Never raises.

    Each closer is isolated: one raising is logged and the rest still run.
    Synchronous, and the closers block (a container removed, a kernel shut
    down, a lock waited for): run it on a worker thread.
    """
    for key, method in closers:
        service = services.get(key)
        if service is None:
            continue
        try:
            getattr(service, method)()
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail
            logger.warning("resource_close_failed", resource=key, error=str(exc))


def rebind_services(services: Mapping[str, Any], old: Any, new: Any) -> None:
    """Move what each of ``services`` holds of its owner ``old`` to ``new``.

    Each service by its row's ``rebind``. Never raises: a rebind that fails
    is logged (``service_rebind_failed``) and the rest still move.
    """
    for key, service in services.items():
        spec = _SPECS.get(key)
        if spec is None or spec.rebind is None:
            continue
        try:
            spec.rebind(service, old, new)
        except Exception as exc:  # noqa: BLE001 - a hand-over must not stop halfway
            logger.warning(
                "service_rebind_failed", service=key, error=f"{type(exc).__name__}: {exc}"
            )


class ServiceLoan:
    """Services one manager lends another, until it gives them over.

    ``BaseWorkflowManager.lend_services`` makes one, of the services the
    lender owns: the table's, but the manager itself (each manager is its
    own ``WORKFLOW`` and summarizer). A manager that initializes with it
    (``initialize_services(services=loan)``) borrows them: it publishes them
    for its tools, builds only what they lack, and no release of it closes
    them. The lender's ``give_services(loan)`` makes the borrower their
    owner. Until then the lender owns them, uses them, and closes them when
    it releases.

    Attributes:
        lender: The manager that lent them.
        services: The services lent, by key.
        borrower: The manager that last initialized with them, or None.
        given: Whether the lender gave them over.
    """

    def __init__(self, lender: "BaseWorkflowManager", services: Mapping[str, Any]) -> None:
        self.lender = lender
        self.services: Mapping[str, Any] = MappingProxyType(dict(services))
        self.borrower: "BaseWorkflowManager | None" = None
        self.given = False


class ServiceSet:
    """The services one manager holds.

    ``services`` is the dict the manager publishes for its tools (the
    registry a turn's ``get_service`` reads); ``carried`` names those a
    model switch keeps through its release (``BaseWorkflowManager._keeping``);
    ``not_owned`` holds those published here that another manager owns
    (borrowed and not yet given, or given away and not yet released): no
    release here closes them, and one that does not carry them drops them.
    """

    def __init__(self, services: dict[str, Any] | None = None) -> None:
        self.services: dict[str, Any] = {} if services is None else services
        self.carried: frozenset[str] = frozenset()
        self.not_owned: dict[str, Any] = {}

    def reset(self, services: dict[str, Any]) -> None:
        """Hold ``services`` from now on, all of them owned here."""
        self.services = services
        self.not_owned = {}

    def owns(self, key: str, service: Any) -> bool:
        """Whether ``service`` is published here under ``key`` and owned here."""
        return self.services.get(key) is service and self.not_owned.get(key) is not service

    def build_into(
        self, manager: "BaseWorkflowManager", built: dict[str, Any], existing: frozenset[str]
    ) -> None:
        """Construct the services ``manager``'s tools declared into ``built``.

        Pure construction, in the table's order: it never writes to
        ``services``. A service is built when a tool declared it, when it is
        built for every manager, or with the one it is built with, unless
        ``existing`` names it. Each builder sees the services built before it
        here, then the published ones (reading those is safe: nothing
        publishes while services are built), so a service carried over a
        model switch is reused rather than built again.
        """
        required = manager.required_managers

        def wanted(spec: ServiceSpec) -> bool:
            if spec.key in existing:
                return False
            if spec.always or spec.key in required:
                return True
            return spec.built_with is not None and wanted(_SPECS[spec.built_with])

        available = ChainMap(built, self.services)
        for spec in SERVICES:
            if not wanted(spec):
                continue
            service = spec.build(manager, available)
            if service is not None:
                built[spec.key] = service

    def publish(self, built: Mapping[str, Any]) -> None:
        """Adopt constructed services without displacing anything already live."""
        for key, service in built.items():
            self.services.setdefault(key, service)

    def release(self) -> dict[str, Any]:
        """Detach the services a release closes, and return them for closing.

        Every service owned here but those ``carried``: they stay published
        and open. One owned elsewhere is never returned: carried, it stays
        published; otherwise it is dropped.
        """
        carried = {key: s for key, s in self.services.items() if key in self.carried}
        detached = {
            key: s
            for key, s in self.services.items()
            if key not in self.carried and self.owns(key, s)
        }
        self.not_owned = {key: s for key, s in self.not_owned.items() if carried.get(key) is s}
        self.services = carried
        return detached

    def lendable(self, manager: "BaseWorkflowManager") -> dict[str, Any]:
        """What ``manager``, which holds this set, may lend: the table's
        services it owns, but itself."""
        return {
            key: s
            for key, s in self.services.items()
            if key in _SPECS and s is not manager and self.owns(key, s)
        }

    def borrow(self, services: Mapping[str, Any]) -> None:
        """Publish another manager's ``services``, as its: a key published here already keeps its own."""
        for key, service in services.items():
            if key not in self.services:
                self.services[key] = service
                self.not_owned[key] = service

    def borrowed(self, services: Mapping[str, Any]) -> dict[str, Any]:
        """Those of ``services`` published here as another manager's."""
        return {
            key: s
            for key, s in services.items()
            if self.services.get(key) is s and self.not_owned.get(key) is s
        }

    def give(self, services: Mapping[str, Any]) -> None:
        """Own ``services`` no more: they stay published until the release, which drops them."""
        self.not_owned.update(services)

    def adopt(self, services: Mapping[str, Any]) -> None:
        """Own ``services``, borrowed until now."""
        for key, service in services.items():
            if self.not_owned.get(key) is service:
                del self.not_owned[key]

    def tool_map(self, manager: "BaseWorkflowManager") -> dict[str, Callable]:
        """The service-bound tools of the services present, by name.

        The interaction tools are always there: their service is the manager.
        The factories bind each closure to the definition of the exact
        module-level tool it re-binds (``factories._issued``), so a variant
        carries identity only while the framework still owns that tool.
        """
        present = {**self.services, WORKFLOW: manager}
        tools: dict[str, Callable] = {}
        for spec in SERVICES:
            service = present.get(spec.key)
            if spec.tools is None or service is None:
                continue
            for tool in spec.tools(service, present, manager):
                tools[tool.__name__] = tool
        return tools


_SPECS: dict[str, ServiceSpec] = {spec.key: spec for spec in SERVICES}
