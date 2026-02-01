"""Tests for webfetch tool."""

import ipaddress

import pytest

from agentic_cli.config import BaseSettings


class TestWebFetchSettings:
    """Tests for webfetch settings fields."""

    def test_webfetch_model_default_none(self):
        """Test webfetch_model defaults to None."""
        settings = BaseSettings()
        assert settings.webfetch_model is None

    def test_webfetch_blocked_domains_default_empty(self):
        """Test webfetch_blocked_domains defaults to empty list."""
        settings = BaseSettings()
        assert settings.webfetch_blocked_domains == []

    def test_webfetch_cache_ttl_default(self):
        """Test webfetch_cache_ttl_seconds defaults to 900."""
        settings = BaseSettings()
        assert settings.webfetch_cache_ttl_seconds == 900

    def test_webfetch_max_content_bytes_default(self):
        """Test webfetch_max_content_bytes defaults to 102400."""
        settings = BaseSettings()
        assert settings.webfetch_max_content_bytes == 102400


class TestURLValidator:
    """Tests for URL validation and SSRF protection."""

    @pytest.fixture
    def validator(self):
        from agentic_cli.tools.webfetch.validator import URLValidator
        return URLValidator(blocked_domains=[])

    def test_valid_https_url(self, validator):
        """Test valid HTTPS URL passes validation."""
        result = validator.validate("https://example.com/page")
        assert result.valid is True
        assert result.error is None

    def test_valid_http_url(self, validator):
        """Test valid HTTP URL passes validation."""
        result = validator.validate("http://example.com/page")
        assert result.valid is True

    def test_invalid_scheme_ftp(self, validator):
        """Test FTP scheme is rejected."""
        result = validator.validate("ftp://example.com/file")
        assert result.valid is False
        assert "scheme" in result.error.lower()

    def test_invalid_scheme_file(self, validator):
        """Test file:// scheme is rejected."""
        result = validator.validate("file:///etc/passwd")
        assert result.valid is False

    def test_localhost_blocked(self, validator):
        """Test localhost is blocked."""
        result = validator.validate("http://localhost/api")
        assert result.valid is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_127_0_0_1_blocked(self, validator):
        """Test 127.0.0.1 is blocked."""
        result = validator.validate("http://127.0.0.1/api")
        assert result.valid is False

    def test_private_ip_10_x_blocked(self, validator):
        """Test 10.x.x.x range is blocked."""
        result = validator.validate("http://10.0.0.1/internal")
        assert result.valid is False

    def test_private_ip_172_16_blocked(self, validator):
        """Test 172.16.x.x range is blocked."""
        result = validator.validate("http://172.16.0.1/internal")
        assert result.valid is False

    def test_private_ip_192_168_blocked(self, validator):
        """Test 192.168.x.x range is blocked."""
        result = validator.validate("http://192.168.1.1/router")
        assert result.valid is False

    def test_link_local_blocked(self, validator):
        """Test 169.254.x.x (link-local) is blocked."""
        result = validator.validate("http://169.254.1.1/")
        assert result.valid is False

    def test_blocked_domain_exact_match(self):
        """Test exact domain blocking."""
        from agentic_cli.tools.webfetch.validator import URLValidator
        validator = URLValidator(blocked_domains=["blocked.com"])
        result = validator.validate("https://blocked.com/page")
        assert result.valid is False
        assert "blocked" in result.error.lower()

    def test_blocked_domain_wildcard(self):
        """Test wildcard domain blocking."""
        from agentic_cli.tools.webfetch.validator import URLValidator
        validator = URLValidator(blocked_domains=["*.blocked.com"])
        result = validator.validate("https://sub.blocked.com/page")
        assert result.valid is False

    def test_blocked_domain_wildcard_no_match_parent(self):
        """Test wildcard doesn't match parent domain."""
        from agentic_cli.tools.webfetch.validator import URLValidator
        validator = URLValidator(blocked_domains=["*.blocked.com"])
        result = validator.validate("https://blocked.com/page")
        assert result.valid is True  # *.blocked.com shouldn't match blocked.com

    def test_malformed_url(self, validator):
        """Test malformed URL is rejected."""
        result = validator.validate("not a url")
        assert result.valid is False


from unittest.mock import AsyncMock, patch


class TestRobotsTxtChecker:
    """Tests for robots.txt compliance."""

    @pytest.fixture
    def checker(self):
        from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
        return RobotsTxtChecker()

    @pytest.mark.asyncio
    async def test_allowed_when_no_robots_txt(self, checker):
        """Test URL is allowed when robots.txt doesn't exist."""
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value.status_code = 404
            result = await checker.can_fetch("https://example.com/page")
            assert result is True

    @pytest.mark.asyncio
    async def test_allowed_when_robots_txt_error(self, checker):
        """Test URL is allowed when robots.txt fetch fails."""
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Network error")
            result = await checker.can_fetch("https://example.com/page")
            assert result is True  # Permissive on error

    @pytest.mark.asyncio
    async def test_blocked_by_robots_txt(self, checker):
        """Test URL is blocked when robots.txt disallows it."""
        robots_content = """
User-agent: *
Disallow: /private/
"""
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = robots_content
            mock_get.return_value = mock_response
            result = await checker.can_fetch("https://example.com/private/secret")
            assert result is False

    @pytest.mark.asyncio
    async def test_allowed_by_robots_txt(self, checker):
        """Test URL is allowed when robots.txt permits it."""
        robots_content = """
User-agent: *
Disallow: /private/
"""
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = robots_content
            mock_get.return_value = mock_response
            result = await checker.can_fetch("https://example.com/public/page")
            assert result is True

    @pytest.mark.asyncio
    async def test_robots_txt_cached(self, checker):
        """Test robots.txt is cached per domain."""
        robots_content = "User-agent: *\nAllow: /"
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = robots_content
            mock_get.return_value = mock_response

            await checker.can_fetch("https://example.com/page1")
            await checker.can_fetch("https://example.com/page2")

            # Should only fetch robots.txt once
            assert mock_get.call_count == 1


import time
import httpx


class TestContentFetcher:
    """Tests for content fetching with caching and redirect handling."""

    @pytest.fixture
    def fetcher(self):
        from agentic_cli.tools.webfetch.fetcher import ContentFetcher
        from agentic_cli.tools.webfetch.validator import URLValidator
        from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
        return ContentFetcher(
            validator=URLValidator(blocked_domains=[]),
            robots_checker=RobotsTxtChecker(),
            cache_ttl_seconds=900,
            max_content_bytes=102400,
        )

    @pytest.mark.asyncio
    async def test_fetch_success(self, fetcher):
        """Test successful fetch."""
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = "<html><body>Content</body></html>"
            mock_response.headers = {"content-type": "text/html"}
            mock_response.history = []
            mock_response.url = "https://example.com/page"
            mock_get.return_value = mock_response

            with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock) as mock_robots:
                mock_robots.return_value = True
                result = await fetcher.fetch("https://example.com/page")

        assert result.success is True
        assert "Content" in result.content

    @pytest.mark.asyncio
    async def test_fetch_blocked_by_validator(self, fetcher):
        """Test fetch blocked by URL validator."""
        result = await fetcher.fetch("http://127.0.0.1/internal")
        assert result.success is False
        assert "blocked" in result.error.lower() or "private" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fetch_blocked_by_robots(self, fetcher):
        """Test fetch blocked by robots.txt."""
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock) as mock_robots:
            mock_robots.return_value = False
            with patch.object(fetcher._validator, "validate") as mock_validate:
                from agentic_cli.tools.webfetch.validator import ValidationResult
                mock_validate.return_value = ValidationResult(valid=True, resolved_ip="1.2.3.4")
                result = await fetcher.fetch("https://example.com/private/page")

        assert result.success is False
        assert "robots" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fetch_cross_host_redirect(self, fetcher):
        """Test cross-host redirect returns redirect info."""
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = "Content"
            mock_response.headers = {"content-type": "text/html"}
            mock_response.url = httpx.URL("https://other.com/page")

            redirect_response = AsyncMock()
            redirect_response.url = httpx.URL("https://example.com/page")
            mock_response.history = [redirect_response]

            mock_get.return_value = mock_response

            with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock) as mock_robots:
                mock_robots.return_value = True
                with patch.object(fetcher._validator, "validate") as mock_validate:
                    from agentic_cli.tools.webfetch.validator import ValidationResult
                    mock_validate.return_value = ValidationResult(valid=True, resolved_ip="1.2.3.4")
                    result = await fetcher.fetch("https://example.com/page")

        assert result.success is False
        assert result.redirect is not None
        assert result.redirect.to_host == "other.com"

    @pytest.mark.asyncio
    async def test_fetch_caching(self, fetcher):
        """Test responses are cached."""
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = "Cached content"
            mock_response.headers = {"content-type": "text/html"}
            mock_response.history = []
            mock_response.url = "https://example.com/page"
            mock_get.return_value = mock_response

            with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock) as mock_robots:
                mock_robots.return_value = True
                with patch.object(fetcher._validator, "validate") as mock_validate:
                    from agentic_cli.tools.webfetch.validator import ValidationResult
                    mock_validate.return_value = ValidationResult(valid=True, resolved_ip="1.2.3.4")

                    result1 = await fetcher.fetch("https://example.com/page")
                    result2 = await fetcher.fetch("https://example.com/page")

        assert result1.from_cache is False
        assert result2.from_cache is True
        assert mock_get.call_count == 1

    @pytest.mark.asyncio
    async def test_fetch_content_truncation(self, fetcher):
        """Test content is truncated when too large."""
        large_content = "x" * 200000

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = large_content
            mock_response.headers = {"content-type": "text/plain"}
            mock_response.history = []
            mock_response.url = "https://example.com/large"
            mock_get.return_value = mock_response

            with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock) as mock_robots:
                mock_robots.return_value = True
                with patch.object(fetcher._validator, "validate") as mock_validate:
                    from agentic_cli.tools.webfetch.validator import ValidationResult
                    mock_validate.return_value = ValidationResult(valid=True, resolved_ip="1.2.3.4")
                    result = await fetcher.fetch("https://example.com/large")

        assert result.success is True
        assert result.truncated is True
        assert len(result.content) <= fetcher._max_content_bytes + 100


class TestHTMLToMarkdown:
    """Tests for HTML to markdown conversion."""

    @pytest.fixture
    def converter(self):
        from agentic_cli.tools.webfetch.converter import HTMLToMarkdown
        return HTMLToMarkdown()

    def test_convert_simple_html(self, converter):
        """Test converting simple HTML."""
        html = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>"
        result = converter.convert(html, "text/html")
        assert "Title" in result
        assert "Paragraph" in result

    def test_convert_preserves_links(self, converter):
        """Test links are preserved."""
        html = '<a href="https://example.com">Link</a>'
        result = converter.convert(html, "text/html")
        assert "example.com" in result or "Link" in result

    def test_convert_preserves_lists(self, converter):
        """Test lists are converted."""
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        result = converter.convert(html, "text/html")
        assert "Item 1" in result
        assert "Item 2" in result

    def test_convert_strips_scripts(self, converter):
        """Test script tags are stripped."""
        html = "<p>Text</p><script>alert('xss')</script>"
        result = converter.convert(html, "text/html")
        assert "alert" not in result
        assert "Text" in result

    def test_convert_strips_styles(self, converter):
        """Test style tags are stripped."""
        html = "<p>Text</p><style>.class { color: red; }</style>"
        result = converter.convert(html, "text/html")
        assert "color" not in result
        assert "Text" in result

    def test_convert_plain_text_passthrough(self, converter):
        """Test plain text passes through unchanged."""
        text = "Just plain text content"
        result = converter.convert(text, "text/plain")
        assert result == text

    def test_convert_json_wrapped(self, converter):
        """Test JSON is wrapped in code block."""
        json_str = '{"key": "value"}'
        result = converter.convert(json_str, "application/json")
        assert "```json" in result
        assert '{"key": "value"}' in result

    def test_convert_binary_placeholder(self, converter):
        """Test binary content returns placeholder."""
        result = converter.convert(b"binary data", "application/octet-stream")
        assert "Binary content" in result or "binary" in result.lower()


class TestLLMSummarizer:
    """Tests for LLM summarizer protocol and context."""

    def test_summarizer_protocol_exists(self):
        """Test LLMSummarizer protocol is defined."""
        from agentic_cli.tools.webfetch.summarizer import LLMSummarizer
        from typing import Protocol
        assert issubclass(LLMSummarizer, Protocol)

    def test_context_getter_setter(self):
        """Test context getter and setter for LLM summarizer."""
        from agentic_cli.workflow.context import (
            get_context_llm_summarizer,
            set_context_llm_summarizer,
        )

        # Initially None
        assert get_context_llm_summarizer() is None

        # Set a mock summarizer
        class MockSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return "Summary"

        mock = MockSummarizer()
        set_context_llm_summarizer(mock)
        assert get_context_llm_summarizer() is mock

        # Clear
        set_context_llm_summarizer(None)
        assert get_context_llm_summarizer() is None

    def test_fast_model_map(self):
        """Test FAST_MODEL_MAP contains expected mappings."""
        from agentic_cli.tools.webfetch.summarizer import FAST_MODEL_MAP

        assert "claude-opus-4-5-20251101" in FAST_MODEL_MAP
        assert "gemini-3-pro" in FAST_MODEL_MAP
        assert "gpt-5" in FAST_MODEL_MAP

    def test_get_fast_model(self):
        """Test get_fast_model returns appropriate model."""
        from agentic_cli.tools.webfetch.summarizer import get_fast_model

        assert get_fast_model("gemini-3-pro") == "gemini-3-flash"
        assert get_fast_model("claude-opus-4-5-20251101") == "claude-haiku-4-20251101"
        assert get_fast_model("unknown-model") is None


class TestWebFetchTool:
    """Tests for the main web_fetch tool function."""

    @pytest.mark.asyncio
    async def test_web_fetch_success(self):
        """Test successful web fetch with mocked fetcher and summarizer."""
        from agentic_cli.tools.webfetch_tool import web_fetch
        from agentic_cli.tools.webfetch.fetcher import FetchResult
        from agentic_cli.workflow.context import set_context_llm_summarizer

        # Create mock summarizer
        class MockSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return f"Summary of: {content[:20]}..."

        # Set summarizer in context
        set_context_llm_summarizer(MockSummarizer())

        try:
            # Mock the fetcher
            with patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher") as mock_get_fetcher:
                mock_fetcher = AsyncMock()
                mock_fetcher.fetch.return_value = FetchResult(
                    success=True,
                    content="<html><body><h1>Test Page</h1><p>Content here</p></body></html>",
                    content_type="text/html",
                    truncated=False,
                    from_cache=False,
                )
                mock_get_fetcher.return_value = mock_fetcher

                result = await web_fetch(
                    url="https://example.com/page",
                    prompt="Summarize this page",
                )

            assert result["success"] is True
            assert "summary" in result
            assert result["url"] == "https://example.com/page"
            assert result["truncated"] is False
            assert result["cached"] is False
        finally:
            set_context_llm_summarizer(None)

    @pytest.mark.asyncio
    async def test_web_fetch_redirect(self):
        """Test web fetch returns redirect info for cross-host redirects."""
        from agentic_cli.tools.webfetch_tool import web_fetch
        from agentic_cli.tools.webfetch.fetcher import FetchResult, RedirectInfo
        from agentic_cli.workflow.context import set_context_llm_summarizer

        # Create mock summarizer
        class MockSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return "Summary"

        set_context_llm_summarizer(MockSummarizer())

        try:
            with patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher") as mock_get_fetcher:
                mock_fetcher = AsyncMock()
                mock_fetcher.fetch.return_value = FetchResult(
                    success=False,
                    redirect=RedirectInfo(
                        from_url="https://example.com/page",
                        to_url="https://other.com/page",
                        to_host="other.com",
                    ),
                    error="Cross-host redirect to other.com",
                )
                mock_get_fetcher.return_value = mock_fetcher

                result = await web_fetch(
                    url="https://example.com/page",
                    prompt="Summarize this page",
                )

            assert result["success"] is False
            assert "redirect" in result
            assert result["redirect"]["to_url"] == "https://other.com/page"
            assert result["redirect"]["to_host"] == "other.com"
        finally:
            set_context_llm_summarizer(None)

    @pytest.mark.asyncio
    async def test_web_fetch_no_summarizer(self):
        """Test web fetch returns error when no summarizer in context."""
        from agentic_cli.tools.webfetch_tool import web_fetch
        from agentic_cli.workflow.context import set_context_llm_summarizer

        # Ensure no summarizer in context
        set_context_llm_summarizer(None)

        result = await web_fetch(
            url="https://example.com/page",
            prompt="Summarize this page",
        )

        assert result["success"] is False
        assert "error" in result
        assert "summarizer" in result["error"].lower()

    def test_web_fetch_requires_decorator(self):
        """Test web_fetch has requires attribute with llm_summarizer."""
        from agentic_cli.tools.webfetch_tool import web_fetch

        assert hasattr(web_fetch, "requires")
        assert "llm_summarizer" in web_fetch.requires
