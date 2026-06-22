"""Reuse an existing native ADK agent config (``root_agent.yaml``).

Two strategies:

- **native** (:func:`load_adk_agent_native`): hand the YAML straight to ADK's
  ``from_config`` and run the resulting agent tree as-is. Full ADK fidelity
  (planners, callbacks, code executors, ``model_code``/LiteLlm), but framework
  service-tool / state-tool auto-injection does not apply, and it only works on
  the ADK backend.

- **translate** (:func:`translate_adk_yaml`): best-effort conversion of the ADK
  YAML into the framework's ``AgentConfig`` list, so the normal manager path
  (service tools, state tools, permissions, plus the new model_settings/mcp/
  skills features) applies. ADK-only constructs are dropped with a warning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentic_cli.logging import Loggers
from agentic_cli.workflow.config import AgentConfig

logger = Loggers.workflow()

# LlmAgentConfig fields with no framework equivalent — dropped during translate.
_DROPPED_FIELDS = (
    "planner",
    "code_executor",
    "model_code",
    "static_instruction",
    "input_schema",
    "output_schema",
    "output_key",
    "before_model_callbacks",
    "after_model_callbacks",
    "before_tool_callbacks",
    "after_tool_callbacks",
    "before_agent_callbacks",
    "after_agent_callbacks",
)


def load_adk_agent_native(config_path: str | Path) -> Any:
    """Build an agent tree natively via ADK ``from_config`` (full fidelity)."""
    from google.adk.agents.config_agent_utils import from_config

    return from_config(str(config_path))


def translate_adk_yaml(config_path: str | Path) -> list[AgentConfig]:
    """Translate a native ADK YAML config into framework ``AgentConfig`` objects.

    Best-effort: only ``LlmAgent`` nodes are translated; other agent classes and
    ADK-only fields are skipped/dropped with a warning. Sub-agents referenced by
    ``config_path`` are translated recursively; ``code`` references are skipped.
    """
    path = Path(config_path)
    out: list[AgentConfig] = []
    _translate_node(_read_yaml(path), path.parent, out)
    return out


def _read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"ADK config {path} did not parse to a mapping.")
    return data


def _translate_node(
    data: dict, base_dir: Path, out: list[AgentConfig]
) -> AgentConfig | None:
    """Translate one ADK agent node (appending children first, then itself)."""
    name = data.get("name")
    agent_class = data.get("agent_class") or "LlmAgent"
    if agent_class not in ("LlmAgent", "Agent"):
        logger.warning(
            "adk_translate_unsupported_agent_class",
            name=name,
            agent_class=agent_class,
        )
        return None

    sub_names: list[str] = []
    for ref in data.get("sub_agents") or []:
        if not isinstance(ref, dict):
            continue
        if ref.get("config_path"):
            sub_path = base_dir / ref["config_path"]
            child = _translate_node(_read_yaml(sub_path), sub_path.parent, out)
            if child is not None:
                sub_names.append(child.name)
        elif ref.get("code"):
            logger.warning("adk_translate_code_subagent_skipped", code=ref["code"])

    tools: list[str] = []
    for tool in data.get("tools") or []:
        tool_name = tool.get("name") if isinstance(tool, dict) else tool
        if isinstance(tool, dict) and tool.get("args"):
            logger.warning("adk_translate_tool_args_dropped", tool=tool_name)
        if tool_name:
            tools.append(tool_name)

    for field in _DROPPED_FIELDS:
        if data.get(field):
            logger.warning("adk_translate_field_dropped", name=name, field=field)

    config = AgentConfig(
        name=name,
        prompt=data.get("instruction", "") or "",
        description=data.get("description", "") or "",
        model=data.get("model"),
        model_settings=_translate_model_settings(data.get("generate_content_config")),
        tools=tools,
        sub_agents=sub_names,
    )
    out.append(config)
    return config


def _translate_model_settings(gcc: dict | None):
    """Map an ADK ``generate_content_config`` dict to ``ModelSettings``."""
    if not gcc:
        return None
    from agentic_cli.workflow.model_settings import ModelSettings, ThinkingSettings

    kwargs: dict[str, Any] = {}
    for src, dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("max_output_tokens", "max_tokens"),
    ):
        if gcc.get(src) is not None:
            kwargs[dst] = gcc[src]
    if gcc.get("stop_sequences"):
        kwargs["stop_sequences"] = gcc["stop_sequences"]

    tc = gcc.get("thinking_config")
    if isinstance(tc, dict):
        if tc.get("thinking_budget") is not None:
            kwargs["thinking"] = ThinkingSettings(
                mode="budget", budget_tokens=tc["thinking_budget"]
            )
        elif tc.get("thinking_level"):
            level = str(tc["thinking_level"]).lower().split(".")[-1]
            mode = level if level in ("low", "medium", "high") else "high"
            kwargs["thinking"] = ThinkingSettings(mode=mode)

    return ModelSettings(**kwargs) if kwargs else None
