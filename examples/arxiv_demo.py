#!/usr/bin/env python
"""Standalone demo for the arXiv search source.

This demo tests the arXiv search functionality:
1. Basic paper search
2. Category filtering
3. Date range filtering
4. Sort options
5. Rate limiting and caching
6. Paper metadata extraction

Usage:
    conda run -n agenticcli python examples/arxiv_demo.py

    # Search for specific topic:
    conda run -n agenticcli python examples/arxiv_demo.py "transformer architecture"

    # With category filter:
    conda run -n agenticcli python examples/arxiv_demo.py --category cs.AI
"""

import sys
import time

from agentic_cli.knowledge_base.sources import SearchSourceResult
from agentic_cli.tools.arxiv_source import ArxivSearchSource


# =============================================================================
# Demo Functions
# =============================================================================


def demo_source_info():
    """Demo arXiv source properties."""
    print("\n" + "=" * 60)
    print("arXiv Search Source Info")
    print("=" * 60)

    source = ArxivSearchSource()

    print(f"  Name: {source.name}")
    print(f"  Description: {source.description}")
    print(f"  Rate limit: {source.rate_limit} seconds between requests")
    print(f"  Requires API key: {source.requires_api_key or 'No'}")
    print(f"  Is available: {source.is_available()}")
    print()


def demo_search_result_format():
    """Demo the SearchSourceResult data structure."""
    print("\n" + "=" * 60)
    print("SearchSourceResult Data Structure")
    print("=" * 60)

    # Create sample result
    sample = SearchSourceResult(
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        snippet="We propose a new simple network architecture, the Transformer...",
        source_name="arxiv",
        metadata={
            "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
            "published": "2017-06-12",
            "categories": ["cs.CL", "cs.LG"],
            "arxiv_id": "1706.03762",
        },
    )

    print("  SearchSourceResult fields:")
    print(f"    title:       {sample.title}")
    print(f"    url:         {sample.url}")
    print(f"    snippet:     {sample.snippet[:50]}...")
    print(f"    source_name: {sample.source_name}")
    print(f"    metadata:")
    for key, value in sample.metadata.items():
        print(f"      {key}: {value}")
    print()


def demo_basic_search(query: str = "large language models"):
    """Demo basic arXiv search."""
    print("\n" + "=" * 60)
    print("Basic arXiv Search Demo")
    print("=" * 60)

    source = ArxivSearchSource()

    print(f"  Query: '{query}'")
    print(f"  Max results: 5")
    print("  Searching...")
    print()

    start_time = time.time()
    results = source.search(query, max_results=5)
    elapsed = time.time() - start_time

    print(f"  Found {len(results)} results in {elapsed:.2f}s")
    print()

    for i, result in enumerate(results, 1):
        print(f"  [{i}] {result.title[:70]}{'...' if len(result.title) > 70 else ''}")
        print(f"      Authors: {', '.join(result.metadata.get('authors', [])[:3])}")
        if len(result.metadata.get('authors', [])) > 3:
            print(f"               ...and {len(result.metadata['authors']) - 3} more")
        print(f"      Published: {result.metadata.get('published', 'N/A')}")
        print(f"      Categories: {', '.join(result.metadata.get('categories', []))}")
        print(f"      URL: {result.url}")
        print()

    return results


def demo_category_filter(category: str = "cs.AI"):
    """Demo arXiv search with category filter."""
    print("\n" + "=" * 60)
    print("Category Filter Demo")
    print("=" * 60)

    source = ArxivSearchSource()

    print(f"  Query: 'neural network'")
    print(f"  Category filter: {category}")
    print("  Searching...")
    print()

    results = source.search(
        "neural network",
        max_results=3,
        categories=[category],
    )

    print(f"  Found {len(results)} results in category {category}")
    for result in results:
        cats = result.metadata.get('categories', [])
        print(f"    - {result.title[:50]}... [{', '.join(cats)}]")
    print()


def demo_sort_options():
    """Demo different sort options."""
    print("\n" + "=" * 60)
    print("Sort Options Demo")
    print("=" * 60)

    source = ArxivSearchSource()
    query = "machine learning"

    # Test different sort options
    sort_options = [
        ("relevance", "descending"),
        ("submittedDate", "descending"),
        ("lastUpdatedDate", "descending"),
    ]

    for sort_by, sort_order in sort_options:
        print(f"  Sort by: {sort_by} ({sort_order})")

        # Note: Need to wait for rate limit between requests
        results = source.search(
            query,
            max_results=2,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        for result in results:
            date = result.metadata.get('published', 'N/A')
            print(f"    - [{date}] {result.title[:45]}...")
        print()


def demo_caching():
    """Demo caching behavior."""
    print("\n" + "=" * 60)
    print("Caching Demo")
    print("=" * 60)

    source = ArxivSearchSource(cache_ttl_seconds=300)  # 5 minute cache
    query = "deep learning"

    # First request (should hit API)
    print("  First request (hits API):")
    start = time.time()
    results1 = source.search(query, max_results=3)
    elapsed1 = time.time() - start
    print(f"    Results: {len(results1)}, Time: {elapsed1:.3f}s")

    # Second request (should be cached - instant)
    print("\n  Second request (from cache):")
    start = time.time()
    results2 = source.search(query, max_results=3)
    elapsed2 = time.time() - start
    print(f"    Results: {len(results2)}, Time: {elapsed2:.3f}s")

    # Verify cache hit (should be much faster)
    if elapsed2 < elapsed1 / 10:
        print("\n  Cache is working! Second request was much faster.")
    else:
        print("\n  Cache may not be working as expected.")

    # Clear cache
    source.clear_cache()
    print("  Cache cleared.")
    print()


def demo_rate_limiting():
    """Demo rate limiting behavior."""
    print("\n" + "=" * 60)
    print("Rate Limiting Demo")
    print("=" * 60)

    # Create source without caching to test rate limiting
    source = ArxivSearchSource(cache_ttl_seconds=0)

    print(f"  Rate limit: {source.rate_limit}s between requests")
    print("  Making 2 different queries back-to-back...")
    print()

    queries = ["quantum computing", "natural language processing"]

    for query in queries:
        start = time.time()
        results = source.search(query, max_results=1)
        elapsed = time.time() - start
        print(f"  Query: '{query}'")
        print(f"    Results: {len(results)}, Time: {elapsed:.2f}s")
        if results:
            print(f"    First result: {results[0].title[:50]}...")
        print()


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#  arXiv Search Demo")
    print("#" * 60)

    # Parse args
    args = sys.argv[1:]
    query = "large language models"
    category = None

    for i, arg in enumerate(args):
        if arg == "--category" and i + 1 < len(args):
            category = args[i + 1]
        elif not arg.startswith("--"):
            query = arg

    # Run demos
    demo_source_info()
    demo_search_result_format()
    demo_basic_search(query)

    if category:
        demo_category_filter(category)
    else:
        demo_category_filter()

    demo_sort_options()
    demo_caching()
    demo_rate_limiting()

    print("\n" + "#" * 60)
    print("#  Demo Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
