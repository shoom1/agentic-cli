"""Workflow settings mixin.

Provides settings for agentic workflow configuration, independent of UI.
These settings control model selection, orchestration, retry behavior,
HITL (human-in-the-loop), memory management, API keys, and tool configuration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from agentic_cli.logging import Loggers
from agentic_cli.models import (
    ModelEntry,
    ModelSpecError,
    ProviderSpec,
    RoleMap,
    parse_ref,
    validate_provider_name,
)
from agentic_cli.models.entries import check_entries
from agentic_cli.settings_sources import RULES_BY_SOURCE, SettingsSourcesMixin
from agentic_cli.workflow.context import MIN_BUDGET

logger = Loggers.config()

#: The global thinking effort, applied to every agent without its own.
_ThinkingEffort = Literal["none", "low", "medium", "high"]
#: The settings an app's settings class ships entries in, merged with the
#: sources' entries rather than replaced by them (``_merge_shipped_entries``).
_SHIPPED_ENTRY_FIELDS = ("models", "model_providers")


class PermissionRuleConfig(BaseModel):
    """A single permission rule — serialised to settings.json as JSON.

    The one schema of a rule: ``settings.permissions`` holds these, and the
    permission engine checks each entry of a project file's ``permissions``
    section with it.
    """

    capability: str
    target: str


class PermissionsConfig(BaseModel):
    """The ``permissions`` setting: the allow and deny rules the permission
    engine enforces (``workflow/permissions/store.py::settings_rules``)."""

    allow: list[PermissionRuleConfig] = []
    deny: list[PermissionRuleConfig] = []


class OrchestratorType(str, Enum):
    """Types of workflow orchestrators available."""

    ADK = "adk"
    LANGGRAPH = "langgraph"
    DEEPAGENTS = "deepagents"


class WorkflowSettingsMixin(SettingsSourcesMixin):
    """Settings for agentic workflow configuration.

    Mixin class that provides workflow-specific settings including:
    - Model selection: the default model, roles, the models list
      (``models``) and the configured providers (``model_providers``), read
      through ``workflow.model_services``
    - Provider API keys
    - Orchestrator and retry settings
    - Tool configuration (web search/fetch, KB, executor)
    - HITL and persistence settings

    Should be composed with BaseSettings via multiple inheritance, before
    pydantic-settings' class: it brings the settings' sources and their trust
    filters (``SettingsSourcesMixin``) with it.

    Note: This is a mixin, not a BaseSettings subclass, to avoid
    MRO issues when composed with other settings classes.
    """

    # Model configuration
    default_model: str | None = Field(
        default=None,
        title="Model",
        description=(
            "The default model, as a provider:model reference or a model id "
            "(when unset, the newest default-tier model in models, else of the "
            "first provider with a key)"
        ),
        json_schema_extra={"ui_order": 10},
    )
    thinking_effort: _ThinkingEffort = Field(
        default="medium",
        title="Thinking Effort",
        description="Controls depth of reasoning for models that support it",
        json_schema_extra={"ui_order": 20},
    )
    # Models and providers (agentic_cli.models). Provisional in 0.7.0.
    #
    # model_roles, model_providers and models are declared in Annotated form,
    # with no class attribute: an app's settings class overrides their
    # defaults, and pydantic warns when a field shadows an attribute of a
    # parent class.
    #
    # model_providers is user-only (not in PROJECT_SETTABLE_KEYS): it holds
    # base URLs, so a project file that could set it could send the user's key
    # to its own server. An app ships its providers by overriding this default
    # in its settings class; the user file's and the environment's are merged
    # into it field by field (_merge_shipped_entries). No save writes it
    # (settings_persistence.HAND_EDITED_KEYS), nor does the /settings dialog
    # show it.
    model_providers: Annotated[
        dict[str, ProviderSpec],
        Field(
            default_factory=dict,
            title="Model Providers",
            description=(
                "OpenAI-compatible hosts (OpenRouter, Together, Fireworks, a local vLLM) by "
                "name: kind, base_url, the environment variable holding the key "
                "(api_key_env). User settings only."
            ),
        ),
    ]
    # model_roles and models only choose among the providers the app and the
    # user defined (an entry's numbers are hints), so a project may set them.
    model_roles: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            title="Model Roles",
            description=(
                "The model for each role other than default (fast, strong, or an app's "
                "own), as a provider:model reference or a model id"
            ),
            json_schema_extra={"ui_order": 11},
        ),
    ]
    # An app ships its list by overriding this default in its settings class;
    # the user's, the project's and the environment's entries are merged into
    # it field by field (_merge_shipped_entries). No save writes it
    # (settings_persistence.HAND_EDITED_KEYS), nor does the /settings dialog
    # show it.
    models: Annotated[
        dict[str, ModelEntry],
        Field(
            default_factory=dict,
            title="Models",
            description=(
                "The models the model pickers offer, by provider:model reference, each "
                "with optional details: context_window, max_output, tools, image_input, "
                "and for a model on a model_providers provider thinking, effort_levels and "
                "reasoning_text. An app ships its list; user and project settings add to it."
            ),
        ),
    ]

    # When a conversation is summarized, on every backend (workflow/context.py):
    # at context_summarize_at of the model's input budget, or at
    # context_max_tokens if that comes first. They replace 0.6's
    # context_window_* settings; a settings file still holding one loads.
    context_summarize_at: float = Field(
        default=0.85,
        ge=0.5,
        le=0.95,
        title="Summarize At",
        description=(
            "How full the model's input budget gets before the earlier conversation "
            "is summarized, as a fraction (0.5 to 0.95)"
        ),
        json_schema_extra={"ui_order": 25},
    )
    context_max_tokens: int = Field(
        default=200_000,
        ge=0,
        title="Context Ceiling",
        description=(
            "Summarize the earlier conversation once a prompt reaches this many tokens, "
            "even on a model with a larger window (at least 16000; 0 means no ceiling)"
        ),
        json_schema_extra={"ui_order": 26},
    )
    # The net beneath every tool's own limits, on every backend
    # (workflow/tool_output.py).
    tool_result_max_chars: int = Field(
        default=50_000,
        ge=2_000,
        title="Tool Result Limit",
        description=(
            "A tool result longer than this many characters reaches the model cut to "
            "its beginning and end; the full text is saved in ./.{app_name}/tool-output "
            "for the agent to read in parts (at least 2000)"
        ),
        json_schema_extra={"ui_order": 27},
    )

    # API Keys (common across all domains, never saved to JSON).
    #
    # Each accepts BOTH the provider's environment variable name and its Python
    # field name (``AliasChoices``): the bare env alias made
    # ``BaseSettings(google_api_key=...)`` bind nothing at all — the value was
    # dropped by ``extra="ignore"`` and the field kept its default. The env name
    # is listed first, so a real environment variable still wins within a source.
    # Values are kept out of ``repr()`` and out of every persisted file (see
    # ``settings_persistence.SECRET_FIELDS``).
    google_api_key: str | None = Field(
        default=None,
        description="Google API key for Gemini models",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "google_api_key"),
        repr=False,
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude models",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key"),
        repr=False,
    )
    # The OpenAI key is never exported to the process environment (see
    # export_api_keys_to_env): the model adapters pass every key to its client
    # explicitly. A configured provider's key (OpenRouter's among them) has no
    # setting: it is read where its api_key_env says (model_services).
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for GPT models",
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
        repr=False,
    )
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily API key for web search",
        validation_alias=AliasChoices("TAVILY_API_KEY", "tavily_api_key"),
        repr=False,
    )
    brave_api_key: str | None = Field(
        default=None,
        description="Brave Search API key for web search",
        validation_alias=AliasChoices("BRAVE_API_KEY", "brave_api_key"),
        repr=False,
    )

    # Web search configuration
    search_backend: Literal["tavily", "brave"] | None = Field(
        default=None,
        title="Search Backend",
        description="Web search provider to use (tavily or brave)",
        json_schema_extra={"ui_order": 55},
    )

    # Web fetch configuration
    webfetch_blocked_domains: list[str] = Field(
        default_factory=list,
        title="WebFetch Blocked Domains",
        description="Domains to block from fetching (supports wildcards like *.example.com)",
        json_schema_extra={"ui_order": 57},
    )
    webfetch_cache_ttl_seconds: int = Field(
        default=900,
        title="WebFetch Cache TTL",
        description="Cache TTL in seconds for fetched pages (default: 15 minutes)",
        json_schema_extra={"ui_order": 58},
    )
    webfetch_max_content_bytes: int = Field(
        default=102400,
        ge=1,
        le=10485760,
        title="WebFetch Max Content",
        description=(
            "How much of a page's text the summarizer sees, in bytes of UTF-8 "
            "(default: 100KB): the page converted whole (an HTML page's main "
            "text, a PDF's text), then cut at this limit"
        ),
        json_schema_extra={"ui_order": 59},
    )
    webfetch_max_pdf_bytes: int = Field(
        default=5242880,
        ge=1,
        le=104857600,
        title="WebFetch Max PDF Size",
        description="Maximum PDF size in bytes (default: 5MB). Separate from HTML limit because PDFs are larger but extracted text is compact.",
        json_schema_extra={"ui_order": 60},
    )
    webfetch_max_download_bytes: int = Field(
        default=5242880,
        ge=1,
        le=104857600,
        title="WebFetch Max Download",
        description=(
            "Largest HTML, text, Markdown, JSON or XML page read and saved, in "
            "bytes (default: 5MB; PDFs use webfetch_max_pdf_bytes instead). "
            "The summarizer sees the first webfetch_max_content_bytes of its text."
        ),
        json_schema_extra={"ui_order": 61},
    )
    webfetch_saved_max_age_days: int = Field(
        default=7,
        ge=0,
        title="Saved Pages Max Age",
        description="Saved pages in ./.{app_name}/fetched are deleted after this many days",
        json_schema_extra={"ui_order": 62},
    )
    webfetch_saved_max_mb: int = Field(
        default=200,
        ge=0,
        title="Saved Pages Max Size",
        description="The oldest saved pages are deleted when ./.{app_name}/fetched grows past this many megabytes",
        json_schema_extra={"ui_order": 63},
    )
    arxiv_max_pdf_bytes: int = Field(
        default=52428800,
        ge=1,
        le=524288000,
        title="ArXiv Max PDF Size",
        description=(
            "Largest paper PDF ingest_arxiv_paper downloads, in bytes (default: "
            "50MB); a larger one is not ingested. web_fetch keeps "
            "webfetch_max_pdf_bytes."
        ),
        json_schema_extra={"ui_order": 64},
    )

    # Knowledge Base configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        title="Embedding Model",
        description="Sentence transformer model for embeddings",
        json_schema_extra={"ui_order": 150},
    )
    embedding_batch_size: int = Field(
        default=32,
        title="Embedding Batch Size",
        description="Batch size for embedding generation",
        json_schema_extra={"ui_order": 151},
    )
    embedding_device: str = Field(
        default="auto",
        title="Embedding Device",
        description=(
            "Device for sentence-transformers: 'auto' picks cuda → "
            "mps (Apple Silicon only) → cpu. Force 'cpu' on Intel Macs "
            "with discrete AMD GPUs (MPS command-buffer failures). "
            "Valid values: auto, cpu, mps, cuda."
        ),
        json_schema_extra={"ui_order": 152},
    )
    knowledge_base_use_mock: bool = Field(
        default=False,
        title="Use Mock Knowledge Base",
        description="Use mock knowledge base (no ML dependencies required)",
        json_schema_extra={"ui_order": 152},
    )
    knowledge_base_summarize: bool = Field(
        default=True,
        title="Summarize Knowledge-Base Documents",
        description=(
            "Give knowledge bases the workflow's summarizer, so ingesting a "
            "document writes an LLM summary and sidecar (one summarizer call "
            "per document). Off: store the first ~500 characters instead."
        ),
        json_schema_extra={"ui_order": 153},
    )
    auto_extract_session_facts: bool = Field(
        default=False,
        title="Propose Memories",
        description="At the end of a conversation, propose memories for review in /memory inbox",
        json_schema_extra={"ui_order": 154},
    )
    memory_index_enabled: bool = Field(
        default=True,
        title="Memory Index",
        description="Send the main agent an index of saved memories at the start of each conversation",
        json_schema_extra={"ui_order": 155},
    )
    memory_index_max_bytes: int = Field(
        default=25_000,
        ge=1_000,
        le=100_000,
        title="Memory Index Size",
        description=(
            "The memory index's size cap in bytes (1,000 to 100,000; at most 200 lines, "
            "and a tenth of the main agent's input budget)"
        ),
        json_schema_extra={"ui_order": 156},
    )

    # User identity (needed by workflow.process())
    default_user: str = Field(
        default="default_user",
        title="Default User",
        description="Default user identifier for sessions",
        json_schema_extra={"ui_order": 60},
    )

    # Orchestrator selection
    orchestrator: OrchestratorType = Field(
        default=OrchestratorType.ADK,
        title="Orchestrator",
        description=(
            "Workflow orchestrator backend: adk (the default), deepagents "
            "(pip install 'agentic-cli[deepagents]') or langgraph (deprecated, "
            "removed in 0.8.0)"
        ),
        json_schema_extra={"ui_order": 100},  # Advanced setting
    )

    # Retry configuration
    retry_max_attempts: int = Field(
        default=3,
        title="Max Retry Attempts",
        description="Maximum retry attempts for transient errors",
        json_schema_extra={"ui_order": 110},
    )
    retry_initial_delay: float = Field(
        default=2.0,
        title="Retry Initial Delay",
        description="Initial delay in seconds before first retry",
        json_schema_extra={"ui_order": 111},
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        title="Retry Backoff Factor",
        description="Multiplier for exponential backoff between retries",
        json_schema_extra={"ui_order": 112},
    )
    anthropic_request_timeout: float = Field(
        default=900.0,
        title="Anthropic Request Timeout",
        description=(
            "Overall timeout (seconds) for direct-API Claude requests. A "
            "non-default value lets high-thinking (large max_tokens) requests "
            "run without the SDK's streaming-required guard."
        ),
        json_schema_extra={"ui_order": 113},
    )

    # Python executor
    python_executor_timeout: int = Field(
        default=30,
        title="Python Executor Timeout",
        description="Default timeout for Python execution (seconds)",
        json_schema_extra={"ui_order": 120},
    )
    python_executor_max_memory_mb: int = Field(
        default=512,
        title="Python Executor Memory Limit",
        description="Maximum memory for Python executor subprocess (MB, Unix only)",
        json_schema_extra={"ui_order": 121},
    )

    # Sandbox executor (stateful Jupyter-backed execution)
    stateful_executor_backend: Literal["none", "local", "docker"] = Field(
        default="none",
        title="Stateful Executor Backend",
        description=(
            "Backend for the stateful sandbox_execute tool. 'none' disables it; "
            "'docker' runs each session in an isolated container (no network, "
            "read-only root filesystem, capped memory/CPU/PIDs; recommended); "
            "'local' runs a Jupyter kernel on the host as you, with your files, "
            "network and privileges: it is NOT isolated and restricts nothing. "
            "Future: 'modal', 'runpod'."
        ),
        json_schema_extra={"ui_order": 121},
    )
    sandbox_timeout: int = Field(
        default=120,
        title="Sandbox Timeout",
        description="Default timeout for sandbox execution (seconds)",
        json_schema_extra={"ui_order": 123},
    )
    sandbox_max_sessions: int = Field(
        default=5,
        title="Sandbox Max Sessions",
        description="Maximum concurrent sandbox sessions",
        json_schema_extra={"ui_order": 124},
    )
    sandbox_image: str = Field(
        default="quay.io/jupyter/scipy-notebook:python-3.12",
        title="Sandbox Image",
        description="Container image for the jupyter_docker backend. Must contain ipykernel/jupyter_client. Pin to a digest in production.",
        json_schema_extra={"ui_order": 126},
    )
    sandbox_memory_mb: int = Field(
        default=2048,
        title="Sandbox Memory (MB)",
        description="Per-container memory cap for the docker backend; also disables swap.",
        json_schema_extra={"ui_order": 127},
    )
    sandbox_cpus: float = Field(
        default=2.0,
        title="Sandbox CPUs",
        description="Per-container CPU cap for the docker backend.",
        json_schema_extra={"ui_order": 128},
    )
    sandbox_pids_limit: int = Field(
        default=256,
        title="Sandbox PID Limit",
        description="Per-container process/thread cap for the docker backend.",
        json_schema_extra={"ui_order": 129},
    )
    sandbox_network: str = Field(
        default="none",
        title="Sandbox Network",
        description="Docker network mode for the docker backend. v1 supports 'none' only.",
        json_schema_extra={"ui_order": 130},
    )
    sandbox_container_user: str = Field(
        default="",
        title="Sandbox Container User",
        description="uid:gid to run the container as; empty uses the image default.",
        json_schema_extra={"ui_order": 131},
    )
    sandbox_data_mounts: list[str] = Field(
        default_factory=list,
        title="Sandbox Data Mounts",
        description="Read-only data staged into the container as 'host_path:mount_name' (mounted read-only under /data/).",
        json_schema_extra={"ui_order": 132},
    )
    sandbox_start_timeout: int = Field(
        default=180,
        title="Sandbox Start Timeout",
        description="Seconds to wait for container start + image pull + kernel readiness (docker backend).",
        json_schema_extra={"ui_order": 133},
    )
    sandbox_outputs_dir: str = Field(
        default="",
        title="Sandbox Outputs Dir",
        description="Shared host dir mounted at /workspace/outputs for FINAL deliverables (default: <workspace_dir>/artifacts).",
        json_schema_extra={"ui_order": 134},
    )

    @field_validator("default_model")
    @classmethod
    def _validate_default_model(cls, v: str | None) -> str | None:
        """A model, never a role: ``default`` is the role it fills."""
        if v is not None:
            try:
                parse_ref(v)
            except ModelSpecError as exc:
                raise ValueError(f"default_model: {exc}") from None
        return v

    @field_validator("context_max_tokens")
    @classmethod
    def _validate_context_max_tokens(cls, v: int) -> int:
        """0 (no ceiling), or a ceiling an agent can work under."""
        if 0 < v < MIN_BUDGET:
            raise ValueError(
                f"context_max_tokens must be 0 (no ceiling) or at least {MIN_BUDGET:,} "
                f"(got {v:,})"
            )
        return v

    @field_validator("model_providers")
    @classmethod
    def _validate_provider_names(cls, v: dict[str, ProviderSpec]) -> dict[str, ProviderSpec]:
        """A provider's name is the provider in its references: no built-in's name."""
        for name in v:
            validate_provider_name(name)
        return v

    @field_validator("model_roles")
    @classmethod
    def _validate_model_roles(cls, v: dict[str, str]) -> dict[str, str]:
        """Each role names a model, never another role, and default is default_model."""
        RoleMap(default_model=None, model_roles=v)
        return v

    @field_validator("models")
    @classmethod
    def _validate_models(cls, v: dict[str, ModelEntry]) -> dict[str, ModelEntry]:
        """Each key a provider:model reference; no thinking fields on a built-in's model."""
        return check_entries(v)

    @model_validator(mode="before")
    @classmethod
    def _merge_shipped_entries(cls, data: Any, info: ValidationInfo) -> Any:
        """The app's shipped entries, then the sources' over them, field by field.

        pydantic-settings merges a dict setting across the user file, the
        project file and the environment key by key, but a value from any of
        them replaces the class default, where an app ships its entries: so
        the merge is done here, on the raw values, before validation (a
        partial override of an entry validates). The merged value is always
        put in the input, so the app's own entries are validated too.

        An assignment (``update_setting``) sets the value it is given: pydantic
        validates only the field assigned there, and takes the rest of what
        this returns as the object's other fields, unvalidated, so nothing is
        changed.
        """
        if info.field_name is not None or not isinstance(data, dict):
            return data
        merged = dict(data)
        for name in _SHIPPED_ENTRY_FIELDS:
            shipped = cls.model_fields[name].get_default(call_default_factory=True) or {}
            given = data.get(name, {})
            if (shipped or name in data) and isinstance(given, Mapping):
                merged[name] = _merged_entries(shipped, given)
        # The rules: the app's, then every source's, lowest priority first
        # (each filed apart by the sources hook: _source_sections), each rule
        # once.
        if "permissions" in data:
            shipped = cls.model_fields["permissions"].get_default(call_default_factory=True)
            sections = _source_sections(data["permissions"])
            merged["permissions"] = _union_rules(
                sections if shipped is None else [shipped, *sections]
            )
        return merged

    @field_validator("sandbox_network")
    @classmethod
    def _validate_sandbox_network(cls, v: str) -> str:
        """Fail closed: the docker backend's no-egress isolation depends on
        --network none, so reject any other value rather than silently
        weakening it. (v1 supports 'none' only.)"""
        if v != "none":
            raise ValueError(
                f"sandbox_network must be 'none' (got {v!r}). The docker sandbox's "
                "network-isolation guarantee depends on it; other modes are not "
                "supported in v1."
            )
        return v

    # OS-level sandboxing
    os_sandbox_enabled: bool = Field(
        default=True,
        title="OS Sandbox Enabled",
        description=(
            "Run execute_python's code inside an OS-level sandbox (sandbox-exec "
            "on macOS, bwrap on Linux). Turned off, the code runs on your "
            "machine with your privileges: the executor's module and syntax "
            "checks are not a security boundary, so each call is approved on "
            "its own."
        ),
        json_schema_extra={"ui_order": 134},
    )
    os_sandbox_strict: bool = Field(
        default=True,
        title="OS Sandbox Strict",
        description=(
            "With the OS sandbox enabled, refuse to run code when no sandbox "
            "backend is available or the sandbox cannot be set up. Turned off, "
            "the code then runs without a sandbox, with your privileges, and "
            "each call is approved on its own."
        ),
        json_schema_extra={"ui_order": 137},
    )
    os_sandbox_writable_paths: list[str] = Field(
        default_factory=list,
        title="OS Sandbox Writable Paths",
        description="Additional paths the sandboxed process can write to (working directory is always writable)",
        json_schema_extra={"ui_order": 135},
    )
    os_sandbox_allow_network: bool = Field(
        default=False,
        title="OS Sandbox Allow Network",
        description="Allow network access from sandboxed processes",
        json_schema_extra={"ui_order": 136},
    )

    # Permissions. The permission engine enforces these rules (store.py,
    # settings_rules): the union of the app's class default, the user file,
    # the environment and the constructor (each source's rules filed apart by
    # SettingsSourcesMixin's sources hook, joined by _merge_shipped_entries).
    # No save writes it (HAND_EDITED_KEYS).
    # Annotated form, with no class attribute, so an app overriding the
    # default gets no pydantic warning. The project file's section sets no
    # setting (PROJECT_PRESERVED_KEYS): the engine reads its deny rules itself.
    permissions: Annotated[
        PermissionsConfig,
        Field(
            default_factory=PermissionsConfig,
            title="Permissions",
            description=(
                "Allow and deny rules the permission engine enforces for tool "
                "capabilities. A project settings file may add deny rules only."
            ),
            json_schema_extra={"ui_order": 138},
        ),
    ]
    permissions_enabled: bool = Field(
        default=True,
        title="Permissions Enabled",
        description="Master switch; when False, all tool calls are allowed.",
        json_schema_extra={"ui_order": 139},
    )
    max_concurrent_jobs: int = Field(
        default=4,
        ge=1,
        title="Max Concurrent Jobs",
        description="Maximum long-running jobs running at once; excess are queued.",
        json_schema_extra={"ui_order": 140},
    )
    job_auto_resume: bool = Field(
        default=False,
        title="Auto-resume Finished Jobs",
        description=(
            "When True, a finished long-running job that opted in "
            "(resume_on_complete) automatically resumes the agent with its "
            "result at the next turn boundary of its conversation, or when "
            "that conversation is opened. When False, /jobs resume does it."
        ),
        json_schema_extra={"ui_order": 141},
    )

    # Session persistence — durable conversations across restarts.
    # Drives every backend: ADK uses DatabaseSessionService, LangGraph and Deep
    # Agents a persistent checkpointer. "memory" = ephemeral. Each backend
    # keeps its own sessions; none reads another's.
    session_store: Literal["memory", "sqlite", "postgres"] = Field(
        default="sqlite",
        title="Session Store",
        description=(
            "Where conversations are persisted: sqlite (default, a single file), "
            "postgres (shared/multi-instance via Postgres URI), or memory (ephemeral). "
            "ADK keeps them in {workspace}/sessions/sessions.db and Deep Agents in "
            "{workspace}/sessions/checkpoints.db; sessions do not move between "
            "backends, so after an orchestrator switch the conversation goes on "
            "from empty under its id (its history stays with the other backend)."
        ),
        json_schema_extra={"ui_order": 145},
    )

    # Skills (Agent Skills / SKILL.md folders)
    skills_dirs: list[str] = Field(
        default_factory=list,
        title="Skills Directories",
        description=(
            "Absolute directories searched for named skills "
            "(Agent Skills / SKILL.md folders)"
        ),
        json_schema_extra={"ui_order": 141},
    )
    # NOTE: ``skill_scripts_enabled`` was removed. Turning it on exposed ADK's
    # ``run_skill_script`` while the supported manager path supplies no code
    # executor, so every call answered ``NO_CODE_EXECUTOR``. Script execution is
    # now enabled by passing a code executor to ``make_skill_toolset`` — the
    # thing that actually makes it work — instead of by a switch that cannot.

    # Where the session store lives (session_store picks which one).
    postgres_uri: str | None = Field(
        default=None,
        title="PostgreSQL URI",
        description=(
            "PostgreSQL connection URI for session_store=postgres: ADK's sessions "
            "and the Deep Agents and LangGraph checkpoints; sessions do not move "
            "between backends"
        ),
        json_schema_extra={"ui_order": 146},
    )
    sqlite_uri: str | None = Field(
        default=None,
        title="SQLite URI",
        description=(
            "SQLite file for session_store=sqlite, as a sqlite:/// URI or a path "
            "(default: ADK's and Deep Agents' own files under {workspace}/sessions/). "
            "ADK and Deep Agents may share one file: their tables do not collide, "
            "and sessions do not move between backends"
        ),
        json_schema_extra={"ui_order": 147},
    )
    store_type: Literal["memory", "postgres"] | None = Field(
        default="memory",
        title="Store Type",
        description="LangGraph only: store type for long-term memory (memory or postgres)",
        json_schema_extra={"ui_order": 148},
    )

    def session_db_url(self) -> str | None:
        """Async SQLAlchemy URL for the session store, or None when ephemeral.

        Shared by both backends so ADK's DatabaseSessionService and LangGraph's
        checkpointer persist to the same place. SQLite is the zero-config
        default (``{workspace}/sessions/sessions.db``); Postgres via uri.
        """
        store = getattr(self, "session_store", "sqlite")
        if store == "memory":
            return None
        if store == "postgres":
            uri = self.postgres_uri
            if not uri:
                raise ValueError("session_store='postgres' requires postgres_uri")
            return uri.replace("postgresql://", "postgresql+asyncpg://", 1)
        # sqlite (default)
        if self.sqlite_uri:
            # One reading of sqlite_uri for every backend: a sqlite:/// or
            # sqlite+aiosqlite:/// URI, or a file path.
            from agentic_cli.workflow.checkpointers import sqlite_file

            return f"sqlite+aiosqlite:///{sqlite_file(self.sqlite_uri)}"
        path = self.sessions_dir / "sessions.db"
        return f"sqlite+aiosqlite:///{path}"

    # LLM debugging settings
    raw_llm_logging: bool = Field(
        default=False,
        title="Raw LLM Logging",
        description="Enable logging of raw LLM request/response traffic for debugging",
        json_schema_extra={"ui_order": 160},
    )
    prompt_caching_enabled: bool = Field(
        default=True,
        title="Prompt Caching",
        description=(
            "Cache prompts on Claude (every backend) and Gemini (explicit caches on "
            "ADK only, billed for storage); reduces cost and latency"
        ),
        json_schema_extra={"ui_order": 170},
    )

    # === Model setters ===

    def set_model(self, model: str) -> None:
        """Set the default model: a ``provider:model`` reference or a model id.

        Only the reference is checked here. Whether its provider has a key and
        whether the model exists are checked when the workflow resolves its
        models (``/settings`` reinitializes it, and rolls the change back if
        that fails).

        Raises:
            ValueError: If ``model`` is malformed or names a role.
        """
        parse_ref(model)
        object.__setattr__(self, "default_model", model)

    def set_model_role(self, role: str, model: str) -> None:
        """Map ``role`` (``fast``, ``strong`` or an app's own) to a model in ``model_roles``.

        Only the names are checked here, as for ``set_model``. The mapping is
        replaced with a new dict, never changed in place, so a caller that
        kept the old one (``/settings``, to roll back) still holds it.

        Raises:
            ValueError: If ``role`` is malformed or ``default`` (that role is
                ``default_model``), or ``model`` is malformed or names a
                role; the message names ``model_roles[role]``.
        """
        roles = {**self.model_roles, role: model}
        RoleMap(default_model=None, model_roles=roles)
        object.__setattr__(self, "model_roles", roles)

    def set_thinking_effort(self, effort: str) -> None:
        """Set the global thinking effort.

        Raises:
            ValueError: If ``effort`` is not none, low, medium or high.
        """
        levels = get_args(_ThinkingEffort)
        if effort not in levels:
            raise ValueError(
                f"Invalid thinking effort '{effort}'. Valid levels: {', '.join(levels)}"
            )
        object.__setattr__(self, "thinking_effort", effort)

    def export_api_keys_to_env(self) -> None:
        """Export configured API keys to provider environment variables.

        Provider SDKs used by the orchestrators (ADK's AnthropicLlm/Gemini,
        LangChain clients) read credentials from process env vars. The export
        OVERWRITES the env from this settings instance: the key fields bind
        only via their env alias (real env vars are the highest-priority
        source), so settings and env diverge only when the process env
        changed after this instance loaded — e.g. an earlier manager's
        export, or a key loaded from a class-specific env_file — and then
        this instance's configured value must win. (The previous
        set-if-absent guard let the first exporting manager pin credentials
        for every later one.) A key unset in settings leaves the environment
        untouched. Credentials are process-global; SettingsContext does not
        isolate them.

        Only the Google and Anthropic keys are exported. ``openai_api_key``
        never is: the model adapters pass every key to its client explicitly,
        so nothing needs it in the environment.
        """
        import os

        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key


def _source_sections(value: Any) -> list[Any]:
    """A ``permissions`` value as each source gave it, lowest priority first.

    The sources hook files each source's value under its rank
    (``settings_sources.RULES_BY_SOURCE``, 0 the highest priority), so
    pydantic-settings' merge, which replaces the lists inside a dict, keeps
    every one. A value given outside the hook (a command-line source that
    pydantic-settings adds above them, a class with its own sources, a value
    validated directly) is taken as it is, after the filed ones.
    """
    if not isinstance(value, Mapping) or RULES_BY_SOURCE not in value:
        return [value]
    filed = value[RULES_BY_SOURCE]
    if not isinstance(filed, Mapping):
        return [filed]  # for validation to refuse
    ranks = sorted(filed, key=lambda rank: int(rank) if str(rank).isdigit() else -1, reverse=True)
    rest = {key: given for key, given in value.items() if key != RULES_BY_SOURCE}
    return [filed[rank] for rank in ranks] + ([rest] if rest else [])


def _union_rules(sections: Iterable[Any]) -> Any:
    """``permissions`` values as one: ``allow`` and ``deny`` each
    concatenated in order, a rule already there (the same capability and
    target) left out.

    A value that is not a section (an object whose ``allow`` and ``deny`` are
    lists) is returned as it is, for validation to refuse: one source's bad
    value is not hidden behind another's.
    """
    merged: dict[str, list[Any]] = {"allow": [], "deny": []}
    seen: dict[str, set[tuple[str, str]]] = {"allow": set(), "deny": set()}
    for section in sections:
        if isinstance(section, PermissionsConfig):
            section = {"allow": section.allow, "deny": section.deny}
        if not isinstance(section, Mapping):
            return section
        for effect, rules in merged.items():
            entries = section.get(effect)
            if entries is None:
                continue
            if not isinstance(entries, (list, tuple)):
                return section
            for entry in entries:
                key = _rule_key(entry)
                if key is not None:
                    if key in seen[effect]:
                        continue
                    seen[effect].add(key)
                rules.append(entry)
    return merged


def _rule_key(entry: Any) -> tuple[str, str] | None:
    """A rule's capability and target, or None for an entry that is no rule."""
    if isinstance(entry, PermissionRuleConfig):
        return entry.capability, entry.target
    if isinstance(entry, Mapping):
        capability, target = entry.get("capability"), entry.get("target")
        if isinstance(capability, str) and isinstance(target, str):
            return capability, target
    return None


def _merged_entries(shipped: Mapping[str, Any], given: Mapping[str, Any]) -> dict[str, Any]:
    """The app's entries, then the sources' value, field by field.

    Keys keep the shipped order, then the sources' new keys in theirs. An
    entry that is neither a model nor a mapping is passed on as it is, for
    validation to refuse.
    """
    merged = {key: _entry_fields(value) for key, value in shipped.items()}
    for key, value in given.items():
        fields, earlier = _entry_fields(value), merged.get(key)
        if isinstance(earlier, dict) and isinstance(fields, dict):
            fields = {**earlier, **fields}
        merged[key] = fields
    return merged


def _entry_fields(value: Any) -> Any:
    """An entry's fields as given: a model's set fields, a mapping's items."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_unset=True)
    if isinstance(value, Mapping):
        return dict(value)
    return value
