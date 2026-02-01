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
