#!/usr/bin/env python
"""Standalone demo for the webfetch tool.

This demo tests the webfetch tool end-to-end:
1. URL validation (SSRF protection)
2. Robots.txt compliance
3. Content fetching with caching
4. HTML to Markdown conversion
5. LLM summarization

Usage:
    conda run -n agenticcli python examples/webfetch_demo.py

    # With a specific URL:
    conda run -n agenticcli python examples/webfetch_demo.py https://example.com

    # With real LLM summarization (requires GOOGLE_API_KEY):
    conda run -n agenticcli python examples/webfetch_demo.py --real-llm
"""

import asyncio
import sys

from agentic_cli.config import get_settings
from agentic_cli.tools.webfetch import (
    URLValidator,
    RobotsTxtChecker,
    HTMLToMarkdown,
    ContentFetcher,
    FAST_MODEL_MAP,
    build_summarize_prompt,
)
from agentic_cli.tools.webfetch_tool import web_fetch
from agentic_cli.workflow.context import set_context_llm_summarizer


# =============================================================================
# LLM Summarizer Implementations
# =============================================================================


class GeminiSummarizer:
    """Summarizer using Google's Gemini API directly.

    This is a standalone implementation for demo purposes.
    In production, workflow managers provide their own.
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            import google.generativeai as genai

            settings = get_settings()
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY not set")

            genai.configure(api_key=settings.google_api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client

    async def summarize(self, content: str, prompt: str) -> str:
        """Summarize content using Gemini."""
        client = self._get_client()
        full_prompt = build_summarize_prompt(content, prompt)

        # Use sync API in async context
        response = await asyncio.to_thread(client.generate_content, full_prompt)
        return response.text


class MockSummarizer:
    """Mock summarizer for testing without API keys."""

    async def summarize(self, content: str, prompt: str) -> str:
        """Return a mock summary with content preview."""
        preview = content[:500].strip()
        if len(content) > 500:
            preview += "..."

        return f"[Mock Summary]\nPrompt: {prompt}\n\nContent preview:\n{preview}"


# =============================================================================
# Demo Functions
# =============================================================================


async def demo_url_validation():
    """Demo URL validation and SSRF protection."""
    print("\n" + "=" * 60)
    print("URL Validation Demo")
    print("=" * 60)

    validator = URLValidator(blocked_domains=["blocked.example.com"])

    test_urls = [
        ("https://example.com/page", "Valid public URL"),
        ("http://localhost/api", "Localhost (blocked)"),
        ("http://127.0.0.1/admin", "Loopback IP (blocked)"),
        ("http://192.168.1.1/router", "Private IP (blocked)"),
        ("ftp://files.example.com/file", "FTP scheme (blocked)"),
        ("https://blocked.example.com/page", "Blocked domain"),
    ]

    for url, description in test_urls:
        result = validator.validate(url)
        status = "PASS" if result.valid else "BLOCKED"
        print(f"  [{status:7}] {description}")
        print(f"           URL: {url}")
        if not result.valid:
            print(f"           Reason: {result.error}")
        print()


async def demo_html_conversion():
    """Demo HTML to Markdown conversion."""
    print("\n" + "=" * 60)
    print("HTML to Markdown Conversion Demo")
    print("=" * 60)

    converter = HTMLToMarkdown()

    html = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Welcome to the Demo</h1>
        <p>This is a <strong>test page</strong> with some content.</p>
        <ul>
            <li>Item one</li>
            <li>Item two</li>
            <li>Item three</li>
        </ul>
        <p>Visit <a href="https://example.com">our website</a> for more.</p>
        <script>alert('This should be removed');</script>
    </body>
    </html>
    """

    markdown = converter.convert(html, "text/html")

    print("  Input HTML (excerpt):")
    print("    <h1>Welcome to the Demo</h1>")
    print("    <p>This is a <strong>test page</strong>...</p>")
    print()
    print("  Output Markdown:")
    for line in markdown.split("\n")[:10]:
        if line.strip():
            print(f"    {line}")
    print()


async def demo_content_fetching(url: str = "https://example.com"):
    """Demo content fetching with caching."""
    print("\n" + "=" * 60)
    print("Content Fetching Demo")
    print("=" * 60)

    settings = get_settings()
    validator = URLValidator(blocked_domains=settings.webfetch_blocked_domains)
    robots_checker = RobotsTxtChecker()

    fetcher = ContentFetcher(
        validator=validator,
        robots_checker=robots_checker,
        cache_ttl_seconds=settings.webfetch_cache_ttl_seconds,
        max_content_bytes=settings.webfetch_max_content_bytes,
    )

    print(f"  Fetching: {url}")
    print()

    # First fetch
    result1 = await fetcher.fetch(url)

    if result1.success:
        print(f"  First fetch:")
        print(f"    Success: {result1.success}")
        print(f"    From cache: {result1.from_cache}")
        print(f"    Content type: {result1.content_type}")
        print(f"    Content length: {len(result1.content or '')} bytes")
        print(f"    Truncated: {result1.truncated}")
        print()

        # Second fetch (should be cached)
        result2 = await fetcher.fetch(url)
        print(f"  Second fetch:")
        print(f"    From cache: {result2.from_cache}")
        print()
    elif result1.redirect:
        print(f"  Redirect detected:")
        print(f"    To: {result1.redirect.to_url}")
        print(f"    Host: {result1.redirect.to_host}")
        print()
    else:
        print(f"  Fetch failed: {result1.error}")
        print()


async def demo_full_webfetch(url: str, use_real_llm: bool = False):
    """Demo the full web_fetch tool."""
    print("\n" + "=" * 60)
    print("Full web_fetch Tool Demo")
    print("=" * 60)

    # Set up summarizer
    if use_real_llm:
        try:
            summarizer = GeminiSummarizer()
            summarizer._get_client()  # Test connection
            print("  Using: Gemini API (real LLM)")
        except Exception as e:
            print(f"  Warning: {e}")
            print("  Falling back to mock summarizer")
            summarizer = MockSummarizer()
    else:
        summarizer = MockSummarizer()
        print("  Using: Mock summarizer (no API key needed)")

    print(f"  URL: {url}")
    print()

    # Set summarizer in context
    set_context_llm_summarizer(summarizer)

    try:
        # Reset the module-level fetcher for clean state
        import agentic_cli.tools.webfetch_tool as webfetch_module
        webfetch_module._fetcher = None

        result = await web_fetch(
            url=url,
            prompt="What is this page about? Summarize the main content.",
            timeout=30,
        )

        print("  Result:")
        print(f"    Success: {result.get('success')}")

        if result.get("success"):
            print(f"    Cached: {result.get('cached')}")
            print(f"    Truncated: {result.get('truncated')}")
            print()
            print("  Summary:")
            summary = result.get("summary", "")
            for line in summary.split("\n")[:15]:
                print(f"    {line}")
            if summary.count("\n") > 15:
                print("    ...")
        elif result.get("redirect"):
            print(f"    Redirect: {result.get('redirect_url')}")
            print(f"    Host: {result.get('redirect_host')}")
        else:
            print(f"    Error: {result.get('error')}")

        print()

    finally:
        set_context_llm_summarizer(None)


async def demo_model_mapping():
    """Demo the FAST_MODEL_MAP for summarization."""
    print("\n" + "=" * 60)
    print("Model Mapping Demo")
    print("=" * 60)

    print("  Main Model -> Fast Model for summarization:")
    print()

    for main_model, fast_model in FAST_MODEL_MAP.items():
        print(f"    {main_model}")
        print(f"      -> {fast_model}")
        print()


async def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#  WebFetch Tool Demo")
    print("#" * 60)

    # Parse command line args
    args = sys.argv[1:]
    use_real_llm = "--real-llm" in args
    urls = [arg for arg in args if not arg.startswith("--")]
    url = urls[0] if urls else "https://example.com"

    if use_real_llm:
        print("\nUsing real LLM for summarization (requires GOOGLE_API_KEY)")
    else:
        print("\nUsing mock summarizer (add --real-llm to use Gemini)")

    # Run demos
    await demo_url_validation()
    await demo_html_conversion()
    await demo_model_mapping()
    await demo_content_fetching(url)
    await demo_full_webfetch(url, use_real_llm)

    print("\n" + "#" * 60)
    print("#  Demo Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
