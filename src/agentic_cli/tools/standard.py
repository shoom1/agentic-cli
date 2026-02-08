"""Backward-compatible re-exports from domain-specific tool modules.

The tools that were originally in this file have been split into:
- knowledge_tools.py: search_knowledge_base, ingest_to_knowledge_base
- arxiv_tools.py: search_arxiv, fetch_arxiv_paper, analyze_arxiv_paper
- execution_tools.py: execute_python
- interaction_tools.py: ask_clarification
"""

# Knowledge base tools
from agentic_cli.tools.knowledge_tools import (  # noqa: F401
    _get_knowledge_base_manager,
    search_knowledge_base,
    ingest_to_knowledge_base,
)

# ArXiv tools
from agentic_cli.tools.arxiv_tools import (  # noqa: F401
    _arxiv_source,
    _get_arxiv_source,
    _clean_arxiv_id,
    search_arxiv,
    fetch_arxiv_paper,
    analyze_arxiv_paper,
)

# Execution tools
from agentic_cli.tools.execution_tools import execute_python  # noqa: F401

# Interaction tools
from agentic_cli.tools.interaction_tools import ask_clarification  # noqa: F401
