"""BM25 keyword index for hybrid retrieval.

Uses bm25s if available, falls back to rank_bm25, or MockBM25Index.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_bm25_index(use_mock: bool = False):
    """Create the best available BM25 index.

    Args:
        use_mock: Force use of mock implementation.

    Returns:
        A BM25 index instance (MockBM25Index).
    """
    if use_mock:
        from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index
        return MockBM25Index()

    # Try bm25s first (fast, C-backed)
    try:
        from agentic_cli.knowledge_base._bm25_backends import BM25sIndex
        return BM25sIndex()
    except ImportError:
        pass

    # Try rank_bm25 (pure Python)
    try:
        from agentic_cli.knowledge_base._bm25_backends import RankBM25Index
        return RankBM25Index()
    except ImportError:
        pass

    # Fallback to mock
    logger.info("No BM25 library available, using mock BM25 index")
    from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index
    return MockBM25Index()
