"""The tool ``agents.yaml`` names: ``word_count``.

An agent's tool in YAML is a registered name, so this module registers it with
``@register_tool`` and ``app.py`` imports it before the agents are built. The
permission gate denies a tool the registry did not issue, so a function that
is only importable is not enough.
"""

from agentic_cli.tools import ToolCategory, register_tool
from agentic_cli.workflow.permissions import EXEMPT


@register_tool(category=ToolCategory.OTHER, capabilities=EXEMPT)
def word_count(text: str) -> dict:
    """Count the words in a text (the pieces between whitespace)."""
    return {"success": True, "words": len(text.split())}
