"""The services a workflow manager builds for its tools: one table, and the set one manager holds.

A tool declares the services it needs (``register_tool(requires=...)``). A
manager builds the ones its agents' tools declare, publishes them for the
tools (``service_registry.get_service``), hands each tool that needs one a
version bound to it (``tool_map``) and closes them when it releases. Each
service is one :class:`ServiceSpec` in :data:`SERVICES`: its key, how it is
built and closed, the tools it gives, and whether a tool may declare it.
Adding a service is adding a row: ``service_registry.KNOWN_SERVICE_KEYS``
(what a tool may declare) and the closers are read from the table.

:class:`ServiceSet` is what one manager holds: the services it publishes, and
which of them a model switch carries (``BaseWorkflowManager._keeping``).

The builders and tool makers import what they build when they run: tools
import ``service_registry`` (and so this module) when they register.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
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
    """

    key: str
    build: Builder
    close: str | None = None
    tools: ToolMaker | None = None
    declarable: bool = False
    always: bool = False
    built_with: str | None = None


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
        KNOWLEDGE_BASE, _knowledge_base, close="close", tools=_kb_tools, declarable=True
    ),
    ServiceSpec(LLM_SUMMARIZER, _the_manager, tools=_webfetch_tool, declarable=True),
    ServiceSpec(
        SANDBOX_MANAGER, _sandbox_manager, close="cleanup", tools=_sandbox_tool, declarable=True
    ),
    ServiceSpec(JOB_MANAGER, _job_manager, close="close", declarable=True),
    ServiceSpec(ARXIV_SOURCE, _arxiv_source, tools=_arxiv_tools, declarable=True),
    # Every agent may need it.
    ServiceSpec(PERMISSION_ENGINE, _permission_engine, always=True),
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


class ServiceSet:
    """The services one manager holds.

    ``services`` is the dict the manager publishes for its tools (the
    registry a turn's ``get_service`` reads); ``carried`` names those a
    model switch keeps through its release (``BaseWorkflowManager._keeping``).
    """

    def __init__(self, services: dict[str, Any] | None = None) -> None:
        self.services: dict[str, Any] = {} if services is None else services
        self.carried: frozenset[str] = frozenset()

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

        Every service but those ``carried``: they stay published and open.
        """
        carried = {key: s for key, s in self.services.items() if key in self.carried}
        detached = {key: s for key, s in self.services.items() if key not in self.carried}
        self.services = carried
        return detached

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
