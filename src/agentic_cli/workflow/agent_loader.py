"""Load agent definitions from a unified YAML config.

This is the framework's own declarative format (distinct from a native ADK
``root_agent.yaml`` — see ``adk_config_bridge`` for reusing those). It produces
``AgentConfig`` objects, so the result feeds the normal factory/manager path.

Example YAML::

    agents:
      - name: coordinator
        model: gemini-2.5-pro
        model_settings:
          temperature: 0.2
          thinking: {mode: high}
        instruction: |
          You are the coordinator...
        tools: [kb_search, my_pkg.tools.custom_tool]
        sub_agents: [researcher]
      - name: researcher
        instruction_file: prompts/researcher.md
        tools: [web_search]

A top-level bare list of agents (without the ``agents:`` key) is also accepted.
Tool entries stay as strings here and are resolved to callables when the
workflow manager is constructed (see ``tools.tool_resolver``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.mcp import MCPServerConfig
from agentic_cli.workflow.model_settings import ModelSettings


class AgentSpec(BaseModel):
    """YAML schema for a single agent (validated, then mapped to AgentConfig)."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    name: str
    # Accept either ``prompt:`` or ``instruction:`` for the system prompt.
    prompt: str | None = Field(
        default=None,
        validation_alias=AliasChoices("prompt", "instruction"),
    )
    instruction_file: str | None = None
    description: str = ""
    model: str | None = None
    model_settings: ModelSettings | None = None
    tools: list[str] = Field(default_factory=list)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    sub_agents: list[str] = Field(default_factory=list)
    include_state_tools: bool = True


class AgentsFile(BaseModel):
    """Top-level YAML schema: a list of agents under ``agents:``."""

    model_config = ConfigDict(extra="forbid")

    agents: list[AgentSpec] = Field(min_length=1)


def _resolve_prompt(spec: AgentSpec, base_dir: Path) -> str:
    """Resolve an agent's prompt from inline text or an instruction file."""
    if spec.instruction_file:
        path = Path(spec.instruction_file)
        if not path.is_absolute():
            path = base_dir / path
        return path.read_text(encoding="utf-8")
    if spec.prompt is not None:
        return spec.prompt
    raise ValueError(
        f"Agent {spec.name!r}: must set 'prompt'/'instruction' or 'instruction_file'."
    )


def load_agents_from_yaml(path: str | Path) -> list[AgentConfig]:
    """Load and validate agent configs from a unified YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        List of ``AgentConfig`` (tool refs left as strings for later resolution).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is invalid or a prompt cannot be resolved.
        pydantic.ValidationError: If the schema does not validate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Agent config file not found: {path}")

    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        raise ValueError(f"Agent config file is empty: {path}")
    # Allow a top-level bare list of agents.
    if isinstance(data, list):
        data = {"agents": data}

    spec_file = AgentsFile.model_validate(data)
    base_dir = path.parent

    configs: list[AgentConfig] = []
    for spec in spec_file.agents:
        configs.append(
            AgentConfig(
                name=spec.name,
                prompt=_resolve_prompt(spec, base_dir),
                tools=list(spec.tools),
                sub_agents=list(spec.sub_agents),
                description=spec.description,
                model=spec.model,
                model_settings=spec.model_settings,
                mcp_servers=list(spec.mcp_servers),
                skills=list(spec.skills),
                include_state_tools=spec.include_state_tools,
            )
        )
    return configs


def create_workflow_manager_from_yaml(
    path: str | Path,
    settings: Any,
    **kwargs: Any,
):
    """Convenience: load agents from YAML and build a workflow manager.

    Equivalent to ``create_workflow_manager_from_settings(load_agents_from_yaml(path), settings)``.
    """
    from agentic_cli.workflow.factory import create_workflow_manager_from_settings

    configs = load_agents_from_yaml(path)
    return create_workflow_manager_from_settings(configs, settings, **kwargs)
