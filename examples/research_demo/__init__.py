"""Research Demo - A multi-agent research assistant.

This demo showcases:
- Memory: Stores research context and learnings
- Planning & task tracking via backend state tools
- Knowledge base: Search, ingest, and read documents
- Web search & content fetching (incl. PDF)
- Academic research (arXiv)
- File operations (read, write, search)
- Human-in-the-loop (approvals)

Usage:
    conda run -n agenticcli python -m examples.research_demo
"""

from examples.research_demo.app import ResearchDemoApp
from examples.research_demo.settings import ResearchDemoSettings

__all__ = [
    "ResearchDemoApp",
    "ResearchDemoSettings",
]
