"""Research Demo - A demo CLI application showcasing P0/P1 features.

This demo showcases:
- Memory: Stores research context and learnings
- Planning: Breaks down research into task graphs
- File ops: Saves findings, compares drafts
- Shell: Runs safe commands (ls, cat, etc.)
- HITL: Requires approval for destructive ops, checkpoints for review

Usage:
    conda run -n agenticcli python -m examples.research_demo
"""

from examples.research_demo.app import ResearchDemoApp
from examples.research_demo.settings import ResearchDemoSettings, get_settings

__all__ = [
    "ResearchDemoApp",
    "ResearchDemoSettings",
    "get_settings",
]
