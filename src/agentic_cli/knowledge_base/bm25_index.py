"""BM25 keyword index for hybrid retrieval.

Uses bm25s if available, falls back to rank_bm25, or MockBM25Index.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Every backend's index file name (each keeps its own), so a knowledge base
# can remove an index that another backend saved.
INDEX_FILES = ("bm25s_sidecar.json", "bm25_rank.json", "bm25_index.json")


def create_bm25_index(use_mock: bool = False):
    """Create the best available BM25 index.

    The backends import their library only when first searched, so the
    library itself is imported here: importing the backend class proves
    nothing about whether the library is installed.

    Args:
        use_mock: Force use of mock implementation.

    Returns:
        A BM25sIndex, a RankBM25Index or a MockBM25Index.
    """
    if use_mock:
        from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index
        return MockBM25Index()

    # Try bm25s first (fast, C-backed)
    try:
        import bm25s  # noqa: F401
    except ImportError:
        pass
    else:
        from agentic_cli.knowledge_base._bm25_backends import BM25sIndex
        return BM25sIndex()

    # Try rank_bm25 (pure Python)
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        pass
    else:
        from agentic_cli.knowledge_base._bm25_backends import RankBM25Index
        return RankBM25Index()

    # Fallback to mock
    logger.info("No BM25 library available, using mock BM25 index")
    from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index
    return MockBM25Index()
