"""URL validation and SSRF protection."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class ValidationResult:
    """Result of URL validation."""

    valid: bool
    error: str | None = None
    resolved_ip: str | None = None


class URLValidator:
    """Validates URLs for security (SSRF protection) and policy compliance.

    Checks:
    - Allowed schemes (http, https only)
    - Private/internal IP addresses blocked
    - Configurable domain blocklist with wildcard support
    """

    ALLOWED_SCHEMES = {"http", "https"}

    BLOCKED_NETWORKS = [
        ipaddress.ip_network("127.0.0.0/8"),      # Loopback
        ipaddress.ip_network("10.0.0.0/8"),       # Private A
        ipaddress.ip_network("172.16.0.0/12"),    # Private B
        ipaddress.ip_network("192.168.0.0/16"),   # Private C
        ipaddress.ip_network("169.254.0.0/16"),   # Link-local
        ipaddress.ip_network("::1/128"),          # IPv6 loopback
        ipaddress.ip_network("fc00::/7"),         # IPv6 private
        ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
    ]

    def __init__(self, blocked_domains: list[str] | None = None) -> None:
        """Initialize validator.

        Args:
            blocked_domains: List of domains to block. Supports wildcards (*.example.com).
        """
        self.blocked_domains = blocked_domains or []

    def validate(self, url: str) -> ValidationResult:
        """Validate a URL for fetching.

        Args:
            url: The URL to validate.

        Returns:
            ValidationResult with valid=True if OK, or valid=False with error message.
        """
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            return ValidationResult(valid=False, error=f"Malformed URL: {e}")

        # Check scheme
        if parsed.scheme not in self.ALLOWED_SCHEMES:
            return ValidationResult(
                valid=False,
                error=f"Scheme '{parsed.scheme}' not allowed. Use http or https.",
            )

        # Check hostname exists
        hostname = parsed.hostname
        if not hostname:
            return ValidationResult(valid=False, error="URL must have a hostname")

        # Check blocked domains
        if self._is_domain_blocked(hostname):
            return ValidationResult(
                valid=False,
                error=f"Domain '{hostname}' is blocked by policy",
            )

        # Resolve hostname and check for private IPs
        try:
            ip_str = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_str)

            for network in self.BLOCKED_NETWORKS:
                if ip in network:
                    return ValidationResult(
                        valid=False,
                        error=f"Private/internal IP address blocked: {ip_str}",
                        resolved_ip=ip_str,
                    )

            return ValidationResult(valid=True, resolved_ip=ip_str)

        except socket.gaierror as e:
            return ValidationResult(
                valid=False,
                error=f"Could not resolve hostname '{hostname}': {e}",
            )

    def _is_domain_blocked(self, hostname: str) -> bool:
        """Check if hostname matches any blocked domain pattern.

        Args:
            hostname: The hostname to check.

        Returns:
            True if blocked, False otherwise.
        """
        hostname_lower = hostname.lower()

        for pattern in self.blocked_domains:
            pattern_lower = pattern.lower()

            # Exact match
            if hostname_lower == pattern_lower:
                return True

            # Wildcard match (*.example.com matches sub.example.com)
            if pattern_lower.startswith("*."):
                suffix = pattern_lower[1:]  # .example.com
                if hostname_lower.endswith(suffix) and hostname_lower != pattern_lower[2:]:
                    return True

        return False
