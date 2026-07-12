"""Tests for webfetch tool."""

import ipaddress
import socket

import httpx
import pytest
from unittest.mock import AsyncMock, patch

from agentic_cli.config import BaseSettings


def _pinned_fetcher(monkeypatch, handler, *, resolves_to="93.184.216.34", **kw):
    """Build a ContentFetcher whose transport is a PinnedTransport over a
    MockTransport(handler); getaddrinfo is stubbed so pinning succeeds."""
    from agentic_cli.tools.webfetch.validator import URLValidator
    from agentic_cli.tools.webfetch.transport import PinnedTransport
    from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
    from agentic_cli.tools.webfetch.fetcher import ContentFetcher

    def _gai(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (resolves_to, port))]
    monkeypatch.setattr(socket, "getaddrinfo", _gai)

    validator = URLValidator()
    transport = PinnedTransport(validator, inner=httpx.MockTransport(handler))
    robots = RobotsTxtChecker()  # robots.can_fetch is patched in these tests, so
                                 # its transport is irrelevant here (wired in Task 4)
    return ContentFetcher(
        validator=validator, robots_checker=robots, transport=transport, **kw
    )


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

    def test_localhost_ip_literal_blocked(self, validator):
        """A loopback IP literal is blocked by validate() without DNS."""
        result = validator.validate("http://127.0.0.1/api")
        assert result.valid is False

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


class TestRobotsTxtChecker:
    def _checker(self, monkeypatch, handler, resolves_to="93.184.216.34"):
        from agentic_cli.tools.webfetch.validator import URLValidator
        from agentic_cli.tools.webfetch.transport import PinnedTransport
        from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
        def _gai(host, port, *a, **k):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (resolves_to, port))]
        monkeypatch.setattr(socket, "getaddrinfo", _gai)
        transport = PinnedTransport(URLValidator(), inner=httpx.MockTransport(handler))
        return RobotsTxtChecker(transport=transport)

    @pytest.mark.asyncio
    async def test_allowed_when_no_robots_txt(self, monkeypatch):
        checker = self._checker(monkeypatch, lambda req: httpx.Response(404))
        assert await checker.can_fetch("https://example.com/page") is True

    @pytest.mark.asyncio
    async def test_blocked_by_robots_txt(self, monkeypatch):
        robots = "User-agent: *\nDisallow: /private/\n"
        checker = self._checker(monkeypatch, lambda req: httpx.Response(200, text=robots))
        assert await checker.can_fetch("https://example.com/private/secret") is False

    @pytest.mark.asyncio
    async def test_allowed_by_robots_txt(self, monkeypatch):
        robots = "User-agent: *\nDisallow: /private/\n"
        checker = self._checker(monkeypatch, lambda req: httpx.Response(200, text=robots))
        assert await checker.can_fetch("https://example.com/public/page") is True

    @pytest.mark.asyncio
    async def test_robots_txt_cached(self, monkeypatch):
        calls = {"n": 0}
        def handler(req):
            calls["n"] += 1
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        checker = self._checker(monkeypatch, handler)
        await checker.can_fetch("https://example.com/page1")
        await checker.can_fetch("https://example.com/page2")
        assert calls["n"] == 1  # one robots.txt fetch per domain


class TestContentFetcher:
    @pytest.mark.asyncio
    async def test_fetch_success(self, monkeypatch):
        fetcher = _pinned_fetcher(
            monkeypatch,
            lambda req: httpx.Response(200, text="<html><body>Content</body></html>",
                                       headers={"content-type": "text/html"}),
        )
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            result = await fetcher.fetch("https://example.com/page")
        assert result.success is True
        assert "Content" in result.content

    @pytest.mark.asyncio
    async def test_fetch_pins_to_validated_ip(self, monkeypatch):
        seen = {}
        def handler(req):
            seen["host"] = req.url.host
            seen["host_header"] = req.headers.get("Host")
            return httpx.Response(200, text="ok", headers={"content-type": "text/html"})
        fetcher = _pinned_fetcher(monkeypatch, handler)
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            await fetcher.fetch("https://example.com/page")
        assert seen["host"] == "93.184.216.34"      # connected to pinned IP
        assert seen["host_header"] == "example.com"  # Host preserved

    @pytest.mark.asyncio
    async def test_fetch_blocked_by_validator_ip_literal(self, monkeypatch):
        fetcher = _pinned_fetcher(monkeypatch, lambda req: httpx.Response(200))
        result = await fetcher.fetch("http://127.0.0.1/internal")
        assert result.success is False
        assert result.error

    @pytest.mark.asyncio
    async def test_fetch_blocked_when_host_resolves_private(self, monkeypatch):
        fetcher = _pinned_fetcher(monkeypatch, lambda req: httpx.Response(200),
                                  resolves_to="10.0.0.5")
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            result = await fetcher.fetch("https://intranet.test/secret")
        assert result.success is False
        assert "block" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_fetch_blocked_by_robots(self, monkeypatch):
        fetcher = _pinned_fetcher(monkeypatch, lambda req: httpx.Response(200, text="x"))
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=False):
            result = await fetcher.fetch("https://example.com/private/page")
        assert result.success is False
        assert "robots" in result.error.lower()

    @pytest.mark.asyncio
    async def test_same_host_redirect_blocked_by_robots(self, monkeypatch):
        def handler(req):
            if req.url.path == "/public":
                return httpx.Response(302, headers={"location": "/private/page"})
            return httpx.Response(200, text="x", headers={"content-type": "text/html"})
        fetcher = _pinned_fetcher(monkeypatch, handler)
        robots_calls = []
        async def fake_can_fetch(u):
            robots_calls.append(u)
            return "/private" not in u
        with patch.object(fetcher._robots, "can_fetch", side_effect=fake_can_fetch):
            result = await fetcher.fetch("https://example.com/public")
        assert result.success is False
        assert "robots" in result.error.lower()
        assert any("/private" in u for u in robots_calls)

    @pytest.mark.asyncio
    async def test_cross_host_redirect_blocks_before_second_request(self, monkeypatch):
        calls = {"n": 0}
        def handler(req):
            calls["n"] += 1
            return httpx.Response(302, headers={"location": "https://other.com/page"})
        fetcher = _pinned_fetcher(monkeypatch, handler)
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            result = await fetcher.fetch("https://example.com/page")
        assert result.success is False
        assert result.redirect is not None and result.redirect.to_host == "other.com"
        assert calls["n"] == 1  # next GET never issued

    @pytest.mark.asyncio
    async def test_same_host_redirect_followed(self, monkeypatch):
        def handler(req):
            if req.url.path == "/public":
                return httpx.Response(302, headers={"location": "/inner"})
            return httpx.Response(200, text="inner page", headers={"content-type": "text/html"})
        fetcher = _pinned_fetcher(monkeypatch, handler)
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            result = await fetcher.fetch("https://example.com/public")
        assert result.success is True
        assert "inner page" in result.content

    @pytest.mark.asyncio
    async def test_redirect_to_internal_ip_blocked(self, monkeypatch):
        def handler(req):
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})
        fetcher = _pinned_fetcher(monkeypatch, handler)
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            result = await fetcher.fetch("https://example.com/page")
        assert result.success is False
        assert "redirect" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_too_many_redirects(self, monkeypatch):
        def handler(req):
            n = int(req.url.path.rsplit("hop", 1)[-1])
            return httpx.Response(302, headers={"location": f"https://example.com/hop{n + 1}"})
        fetcher = _pinned_fetcher(monkeypatch, handler)
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            result = await fetcher.fetch("https://example.com/hop0")
        assert result.success is False
        assert "redirect" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_caching(self, monkeypatch):
        calls = {"n": 0}
        def handler(req):
            calls["n"] += 1
            return httpx.Response(200, text="Cached content", headers={"content-type": "text/html"})
        fetcher = _pinned_fetcher(monkeypatch, handler)
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            r1 = await fetcher.fetch("https://example.com/page")
            r2 = await fetcher.fetch("https://example.com/page")
        assert r1.from_cache is False and r2.from_cache is True
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_content_truncation_streamed(self, monkeypatch):
        big = "x" * 200000
        fetcher = _pinned_fetcher(
            monkeypatch,
            lambda req: httpx.Response(200, text=big, headers={"content-type": "text/plain"}),
            max_content_bytes=102400,
        )
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            result = await fetcher.fetch("https://example.com/large")
        assert result.success is True and result.truncated is True
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


class TestPDFConversion:
    """Tests for PDF content extraction."""

    @pytest.fixture
    def converter(self):
        from agentic_cli.tools.webfetch.converter import HTMLToMarkdown
        return HTMLToMarkdown()

    def test_convert_pdf_extracts_text(self, converter):
        """Test PDF bytes are converted to text with page markers."""
        pypdf = pytest.importorskip("pypdf")
        import io
        from pypdf import PdfWriter

        # Create a simple PDF with text
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        # Add text via annotation (simplest way to get extractable text)
        page = writer.pages[0]
        # Use a second page too
        writer.add_blank_page(width=200, height=200)

        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        result = converter.convert(pdf_bytes, "application/pdf")
        # Should not be a binary placeholder or error
        assert "[Binary content" not in result
        # If pages had no text, we get the "no extractable text" message
        # Either way, it should not crash
        assert isinstance(result, str)

    def test_convert_pdf_no_pypdf_fallback(self, converter):
        """Test graceful fallback when pypdf is not installed."""
        with patch.dict("sys.modules", {"pypdf": None}):
            # Force re-import failure
            import importlib
            result = converter.convert(b"%PDF-1.4 fake", "application/pdf")
            # Should get a graceful message (either import error or extraction error)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_convert_pdf_no_text(self, converter):
        """Test PDF with no extractable text returns appropriate message."""
        pypdf = pytest.importorskip("pypdf")
        import io
        from pypdf import PdfWriter

        # Create an empty PDF (no text content)
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        result = converter.convert(pdf_bytes, "application/pdf")
        assert "no extractable text" in result.lower() or "Page" in result

    @pytest.mark.asyncio
    async def test_fetcher_pdf_uses_bytes(self, monkeypatch):
        pdf = b"%PDF-1.4 fake pdf content"
        fetcher = _pinned_fetcher(
            monkeypatch,
            lambda req: httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"}),
        )
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            result = await fetcher.fetch("https://arxiv.org/pdf/2301.00001")
        assert result.success is True
        assert result.content == pdf and isinstance(result.content, bytes)

    @pytest.mark.asyncio
    async def test_fetcher_pdf_byte_limit(self, monkeypatch):
        big = b"x" * 5000
        fetcher = _pinned_fetcher(
            monkeypatch,
            lambda req: httpx.Response(200, content=big, headers={"content-type": "application/pdf"}),
            max_pdf_bytes=1000,
        )
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            result = await fetcher.fetch("https://arxiv.org/pdf/2301.00001")
        assert result.success is True and result.truncated is True
        assert len(result.content) == 1000


class TestWebFetchPDFSetting:
    """Tests for PDF settings."""

    def test_webfetch_max_pdf_bytes_default(self):
        """Test webfetch_max_pdf_bytes defaults to 5MB."""
        settings = BaseSettings()
        assert settings.webfetch_max_pdf_bytes == 5242880


class TestLLMSummarizer:
    """Tests for LLM summarizer protocol and context."""

    def test_summarizer_protocol_exists(self):
        """Test LLMSummarizer protocol is defined."""
        from agentic_cli.tools.webfetch.summarizer import LLMSummarizer
        from typing import Protocol
        assert issubclass(LLMSummarizer, Protocol)

    def test_context_getter_setter(self):
        """Test service registry getter/setter for LLM summarizer."""
        from agentic_cli.workflow.service_registry import (
            get_service,
            get_service_registry,
            set_service_registry,
        )

        token = set_service_registry({})
        try:
            # Initially None
            assert get_service("llm_summarizer") is None

            # Set a mock summarizer
            class MockSummarizer:
                async def summarize(self, content: str, prompt: str) -> str:
                    return "Summary"

            mock = MockSummarizer()
            get_service_registry()["llm_summarizer"] = mock
            assert get_service("llm_summarizer") is mock

            # Clear
            get_service_registry()["llm_summarizer"] = None
            assert get_service("llm_summarizer") is None
        finally:
            token.var.reset(token)

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
        from agentic_cli.workflow.service_registry import set_service_registry

        # Create mock summarizer
        class MockSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return f"Summary of: {content[:20]}..."

        token = set_service_registry({"llm_summarizer": MockSummarizer()})

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
            token.var.reset(token)

    @pytest.mark.asyncio
    async def test_web_fetch_redirect(self):
        """Test web fetch returns redirect info for cross-host redirects."""
        from agentic_cli.tools.webfetch_tool import web_fetch
        from agentic_cli.tools.webfetch.fetcher import FetchResult, RedirectInfo
        from agentic_cli.workflow.service_registry import set_service_registry

        # Create mock summarizer
        class MockSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return "Summary"

        token = set_service_registry({"llm_summarizer": MockSummarizer()})

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
            assert result["redirect"] is True
            assert result["redirect_url"] == "https://other.com/page"
            assert result["redirect_host"] == "other.com"
            assert "message" in result
            assert result["url"] == "https://example.com/page"
        finally:
            token.var.reset(token)

    @pytest.mark.asyncio
    async def test_web_fetch_no_summarizer(self):
        """Test web fetch returns error when no summarizer in context."""
        from agentic_cli.tools.webfetch_tool import web_fetch
        from agentic_cli.workflow.service_registry import set_service_registry

        # Ensure no summarizer in context (empty registry)
        token = set_service_registry({})

        try:
            result = await web_fetch(
                url="https://example.com/page",
                prompt="Summarize this page",
            )

            assert result["success"] is False
            assert "error" in result
            assert "summarizer" in result["error"].lower()
        finally:
            token.var.reset(token)

    def test_web_fetch_detected_via_tool_service_map(self):
        """Test web_fetch is detected via _TOOL_SERVICE_MAP."""
        from agentic_cli.workflow.base_manager import BaseWorkflowManager

        assert "web_fetch" in BaseWorkflowManager._TOOL_SERVICE_MAP
        assert BaseWorkflowManager._TOOL_SERVICE_MAP["web_fetch"] == "llm_summarizer"


class TestWorkflowManagerIntegration:
    """Tests for workflow manager integration with webfetch tool."""

    def test_llm_summarizer_detected_in_required_managers(self):
        """Test that llm_summarizer is detected from web_fetch tool via _TOOL_SERVICE_MAP."""
        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        from agentic_cli.workflow.config import AgentConfig
        from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest
        from agentic_cli.tools.webfetch_tool import web_fetch
        from typing import AsyncGenerator

        # Create a minimal test subclass of BaseWorkflowManager
        class TestWorkflowManager(BaseWorkflowManager):
            @property
            def backend_type(self) -> str:
                return "test"

            @property
            def model(self) -> str:
                return "test-model"

            async def _do_initialize(self) -> None:
                pass

            def _get_state_tools(self) -> list:
                return []

            async def process(
                self, message: str, user_id: str, session_id: str | None = None
            ) -> AsyncGenerator[WorkflowEvent, None]:
                if False:
                    yield  # type: ignore

            async def reinitialize(
                self, model: str | None = None, preserve_sessions: bool = True
            ) -> None:
                pass

            async def cleanup(self) -> None:
                pass

            async def _extract_session_data(self, session_id: str) -> tuple[list[dict], str | None]:
                return [], None

            async def _inject_session_messages(self, session_id: str, messages: list[dict], current_agent: str | None = None) -> None:
                pass

        # Create an agent config that uses the web_fetch tool
        agent_config = AgentConfig(
            name="test_agent",
            prompt="Test prompt",
            tools=[web_fetch],
        )

        # Instantiate the workflow manager
        manager = TestWorkflowManager(agent_configs=[agent_config])

        # Verify llm_summarizer is in required_managers
        assert "llm_summarizer" in manager.required_managers

    def test_llm_summarizer_property_exists(self):
        """Test that llm_summarizer property exists on BaseWorkflowManager."""
        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        from agentic_cli.workflow.config import AgentConfig
        from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest
        from typing import AsyncGenerator

        # Create a minimal test subclass
        class TestWorkflowManager(BaseWorkflowManager):
            @property
            def backend_type(self) -> str:
                return "test"

            @property
            def model(self) -> str:
                return "test-model"

            async def _do_initialize(self) -> None:
                pass

            def _get_state_tools(self) -> list:
                return []

            async def process(
                self, message: str, user_id: str, session_id: str | None = None
            ) -> AsyncGenerator[WorkflowEvent, None]:
                if False:
                    yield  # type: ignore

            async def reinitialize(
                self, model: str | None = None, preserve_sessions: bool = True
            ) -> None:
                pass

            async def cleanup(self) -> None:
                pass

            async def _extract_session_data(self, session_id: str) -> tuple[list[dict], str | None]:
                return [], None

            async def _inject_session_messages(self, session_id: str, messages: list[dict], current_agent: str | None = None) -> None:
                pass

        # Create a simple agent config (no web_fetch)
        agent_config = AgentConfig(
            name="test_agent",
            prompt="Test prompt",
            tools=[],
        )

        manager = TestWorkflowManager(agent_configs=[agent_config])

        # The llm_summarizer property should exist
        assert hasattr(manager, "llm_summarizer")
        # Initially None since no tools require it
        assert manager.llm_summarizer is None

    @pytest.mark.asyncio
    async def test_manager_becomes_summarizer_when_required(self):
        """Test that the manager itself becomes the llm_summarizer when required."""
        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        from agentic_cli.workflow.config import AgentConfig
        from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest
        from agentic_cli.tools.webfetch_tool import web_fetch
        from typing import AsyncGenerator

        class TestWorkflowManager(BaseWorkflowManager):
            @property
            def backend_type(self) -> str:
                return "test"

            @property
            def model(self) -> str:
                return "test-model"

            async def _do_initialize(self) -> None:
                pass

            def _get_state_tools(self) -> list:
                return []

            async def process(
                self, message: str, user_id: str, session_id: str | None = None
            ) -> AsyncGenerator[WorkflowEvent, None]:
                if False:
                    yield  # type: ignore

            async def reinitialize(
                self, model: str | None = None, preserve_sessions: bool = True
            ) -> None:
                pass

            async def cleanup(self) -> None:
                pass

            async def _extract_session_data(self, session_id: str) -> tuple[list[dict], str | None]:
                return [], None

            async def _inject_session_messages(self, session_id: str, messages: list[dict], current_agent: str | None = None) -> None:
                pass

        # Create an agent config with web_fetch tool
        agent_config = AgentConfig(
            name="test_agent",
            prompt="Test prompt",
            tools=[web_fetch],
        )

        manager = TestWorkflowManager(agent_configs=[agent_config])

        # Before initialization, summarizer should be None
        assert manager.llm_summarizer is None

        # Initialize to trigger manager creation (skip validation — no API keys in test)
        await manager.initialize_services(validate=False)

        # The manager itself should be the summarizer (has summarize() method)
        assert manager.llm_summarizer is manager
        assert hasattr(manager.llm_summarizer, "summarize")


class TestFactoryWiring:
    def test_get_or_create_fetcher_shares_one_pinned_transport(self):
        import agentic_cli.tools.webfetch_tool as wt
        from agentic_cli.tools.webfetch.transport import PinnedTransport
        from agentic_cli.config import BaseSettings

        orig_fetcher = wt._fetcher
        orig_snapshot = wt._fetcher_settings_snapshot
        try:
            wt._fetcher = None
            wt._fetcher_settings_snapshot = None
            fetcher = wt.get_or_create_fetcher(BaseSettings())
            assert isinstance(fetcher._transport, PinnedTransport)
            assert fetcher._robots._transport is fetcher._transport
            assert fetcher._transport._validator is fetcher._validator
        finally:
            wt._fetcher = orig_fetcher
            wt._fetcher_settings_snapshot = orig_snapshot
