"""Tests for webfetch tool."""

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
