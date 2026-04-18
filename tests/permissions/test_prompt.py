"""Tests for prompt build + response parse."""

import pytest

from agentic_cli.workflow.events import InputType, UserInputRequest
from agentic_cli.workflow.permissions.capabilities import ResolvedCapability
from agentic_cli.workflow.permissions.prompt import (
    ALLOW_ALWAYS_CHOICE,
    ALLOW_ONCE_CHOICE,
    ALLOW_SESSION_CHOICE,
    DENY_CHOICE,
    build_request,
    parse_response,
)
from agentic_cli.workflow.permissions.rules import AskScope


class TestBuildRequest:
    def test_uses_choice_input_type(self):
        req = build_request("write_file", [ResolvedCapability("filesystem.write", "/foo")])
        assert req.input_type is InputType.CHOICE
        assert req.default == DENY_CHOICE

    def test_choices_in_fixed_order(self):
        req = build_request("write_file", [ResolvedCapability("filesystem.write", "/foo")])
        assert req.choices == [
            ALLOW_ONCE_CHOICE, ALLOW_SESSION_CHOICE, ALLOW_ALWAYS_CHOICE, DENY_CHOICE
        ]

    def test_prompt_mentions_tool_and_capabilities(self):
        """For filesystem.* the prompt shows the broadened (parent-directory)
        scope — matching what an Allow Session/Always would actually store."""
        req = build_request(
            "copy_file",
            [
                ResolvedCapability("filesystem.read", "/src/file_a.txt"),
                ResolvedCapability("filesystem.write", "/dst/file_b.txt"),
            ],
        )
        assert "copy_file" in req.prompt
        # Both entries show the parent-directory glob, not the exact file.
        assert "filesystem.read → /src/**" in req.prompt
        assert "filesystem.write → /dst/**" in req.prompt
        # The exact filenames do NOT appear — the scope the user will actually
        # grant is the directory.
        assert "file_a.txt" not in req.prompt
        assert "file_b.txt" not in req.prompt
        # And a hint explaining the widening:
        assert "parent directory" in req.prompt

    def test_prompt_preserves_exact_http_target(self):
        """Non-filesystem capabilities keep their exact display target."""
        req = build_request(
            "web_fetch",
            [ResolvedCapability("http.read", "https://example.com/path")],
        )
        assert "https://example.com/path" in req.prompt
        # No broadening hint when no filesystem capability is involved.
        assert "parent directory" not in req.prompt

    def test_prompt_handles_root_parent(self):
        """A file directly under /: broadening collapses '//**' to '/**'."""
        req = build_request(
            "write_file",
            [ResolvedCapability("filesystem.write", "/hello.txt")],
        )
        assert "filesystem.write → /**" in req.prompt
        assert "//**" not in req.prompt

    def test_request_id_has_perm_prefix(self):
        req = build_request("x", [])
        assert req.request_id.startswith("perm-")


class TestParseResponse:
    @pytest.mark.parametrize("text, scope", [
        (ALLOW_ONCE_CHOICE,    AskScope.ONCE),
        (ALLOW_SESSION_CHOICE, AskScope.SESSION),
        (ALLOW_ALWAYS_CHOICE,  AskScope.PROJECT),
        (DENY_CHOICE,          AskScope.DENY),
    ])
    def test_round_trip(self, text, scope):
        assert parse_response(text) is scope

    def test_unknown_defaults_to_deny(self):
        assert parse_response("whatever") is AskScope.DENY

    def test_empty_defaults_to_deny(self):
        assert parse_response("") is AskScope.DENY

    def test_whitespace_tolerant(self):
        assert parse_response(f"  {ALLOW_ONCE_CHOICE}  ") is AskScope.ONCE
