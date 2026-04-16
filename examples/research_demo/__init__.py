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
    research-demo          # console script (after pip install -e .)
    python -m research_demo
"""

from .app import ResearchDemoApp
from .settings import ResearchDemoSettings

__all__ = [
    "ResearchDemoApp",
    "ResearchDemoSettings",
]
