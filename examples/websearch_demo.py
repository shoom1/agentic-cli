#!/usr/bin/env python
"""Standalone demo for the web search tool.

This demo tests the web search tool with pluggable backends:
1. Backend configuration (Tavily, Brave)
2. Search query execution
3. Result parsing and formatting
4. Error handling for missing API keys

Usage:
    conda run -n agenticcli python examples/websearch_demo.py

    # With Tavily backend (requires TAVILY_API_KEY):
    TAVILY_API_KEY=your_key conda run -n agenticcli python examples/websearch_demo.py --backend tavily

    # With Brave backend (requires BRAVE_API_KEY):
    BRAVE_API_KEY=your_key conda run -n agenticcli python examples/websearch_demo.py --backend brave
"""

import asyncio
import sys

from agentic_cli.tools.search import (
    web_search,
    SearchResult,
    TavilyBackend,
    BraveBackend,
    SEARCH_BACKENDS,
)
from agentic_cli.config import get_settings


# =============================================================================
# Demo Functions
# =============================================================================


def demo_backend_info():
    """Demo available backends and their configuration."""
    print("\n" + "=" * 60)
    print("Available Search Backends")
    print("=" * 60)

    print(f"  Registered backends: {list(SEARCH_BACKENDS.keys())}")
    print()

    for name, backend_class in SEARCH_BACKENDS.items():
        print(f"  {name}:")
        print(f"    Class: {backend_class.__name__}")
        if hasattr(backend_class, 'BASE_URL'):
            print(f"    API URL: {backend_class.BASE_URL}")
        print()


def demo_configuration_check():
    """Demo checking current configuration."""
    print("\n" + "=" * 60)
    print("Configuration Check")
    print("=" * 60)

    settings = get_settings()

    # Check configured backend
    backend = getattr(settings, 'search_backend', None)
    print(f"  Configured backend: {backend or '(not set)'}")

    # Check API keys (masked)
    tavily_key = getattr(settings, 'tavily_api_key', None)
    brave_key = getattr(settings, 'brave_api_key', None)

    print(f"  Tavily API key: {'***' + tavily_key[-4:] if tavily_key else '(not set)'}")
    print(f"  Brave API key: {'***' + brave_key[-4:] if brave_key else '(not set)'}")
    print()

    # Determine what can be tested
    available_backends = []
    if tavily_key:
        available_backends.append("tavily")
    if brave_key:
        available_backends.append("brave")

    if available_backends:
        print(f"  Available for testing: {available_backends}")
    else:
        print("  No API keys configured. Set TAVILY_API_KEY or BRAVE_API_KEY to test.")
    print()

    return available_backends


def demo_search_result_format():
    """Demo the SearchResult data structure."""
    print("\n" + "=" * 60)
    print("SearchResult Data Structure")
    print("=" * 60)

    # Create sample result
    sample = SearchResult(
        title="Example Search Result",
        url="https://example.com/article",
        snippet="This is a sample snippet showing what search results look like...",
        score=0.95,
    )

    print("  SearchResult fields:")
    print(f"    title:   {sample.title}")
    print(f"    url:     {sample.url}")
    print(f"    snippet: {sample.snippet[:50]}...")
    print(f"    score:   {sample.score}")
    print()


async def demo_mock_search():
    """Demo search with mock data (no API key needed)."""
    print("\n" + "=" * 60)
    print("Mock Search Demo (No API needed)")
    print("=" * 60)

    # Without an API key, web_search returns an error
    result = await web_search("Python programming", max_results=3)

    print("  Query: 'Python programming'")
    print(f"  Success: {result['success']}")

    if result['success']:
        print(f"  Results: {len(result['results'])}")
        for r in result['results']:
            print(f"    - {r['title']}")
    else:
        print(f"  Error: {result['error']}")
        print("\n  (This is expected without API keys configured)")
    print()


async def demo_live_search(backend: str, query: str):
    """Demo live search with configured backend."""
    print("\n" + "=" * 60)
    print(f"Live Search Demo ({backend.title()} Backend)")
    print("=" * 60)

    print(f"  Query: '{query}'")
    print(f"  Max results: 5")
    print()

    # Temporarily set backend in settings
    settings = get_settings()
    original_backend = getattr(settings, 'search_backend', None)
    settings.search_backend = backend

    try:
        result = await web_search(query, max_results=5)

        print(f"  Success: {result['success']}")

        if result['success']:
            print(f"  Results found: {len(result['results'])}")
            print()

            for i, r in enumerate(result['results'], 1):
                print(f"  Result {i}:")
                print(f"    Title: {r['title'][:60]}{'...' if len(r['title']) > 60 else ''}")
                print(f"    URL: {r['url'][:70]}{'...' if len(r['url']) > 70 else ''}")
                snippet = r['snippet'][:100] + "..." if len(r['snippet']) > 100 else r['snippet']
                print(f"    Snippet: {snippet}")
                if r.get('score'):
                    print(f"    Score: {r['score']:.3f}")
                print()
        else:
            print(f"  Error: {result['error']}")
            print()

    finally:
        # Restore original backend
        settings.search_backend = original_backend


async def demo_error_handling():
    """Demo error handling scenarios."""
    print("\n" + "=" * 60)
    print("Error Handling Demo")
    print("=" * 60)

    settings = get_settings()

    # Test 1: No backend configured
    original_backend = getattr(settings, 'search_backend', None)
    settings.search_backend = None

    result = await web_search("test query")
    print("  Scenario: No backend configured")
    print(f"    Success: {result['success']}")
    print(f"    Error: {result['error'][:60]}...")
    print()

    # Test 2: Unknown backend
    settings.search_backend = "unknown_backend"

    result = await web_search("test query")
    print("  Scenario: Unknown backend")
    print(f"    Success: {result['success']}")
    print(f"    Error: {result['error'][:60]}...")
    print()

    # Restore
    settings.search_backend = original_backend


async def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#  Web Search Tool Demo")
    print("#" * 60)

    # Parse args
    args = sys.argv[1:]
    backend_arg = None
    query = "artificial intelligence news"

    for i, arg in enumerate(args):
        if arg == "--backend" and i + 1 < len(args):
            backend_arg = args[i + 1]
        elif not arg.startswith("--"):
            query = arg

    # Run demos
    demo_backend_info()
    available_backends = demo_configuration_check()
    demo_search_result_format()
    await demo_mock_search()
    await demo_error_handling()

    # Run live search if backend available
    if backend_arg and backend_arg in available_backends:
        await demo_live_search(backend_arg, query)
    elif available_backends:
        # Use first available backend
        await demo_live_search(available_backends[0], query)
    else:
        print("\n" + "-" * 60)
        print("  Skipping live search demo (no API keys available)")
        print("  To run live search, set one of:")
        print("    export TAVILY_API_KEY=your_key")
        print("    export BRAVE_API_KEY=your_key")
        print("-" * 60)

    print("\n" + "#" * 60)
    print("#  Demo Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
