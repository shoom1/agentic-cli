"""Tests for docker sandbox settings defaults."""

import pytest
from pydantic import ValidationError

from agentic_cli.config import BaseSettings


def test_docker_sandbox_defaults():
    s = BaseSettings()
    assert s.sandbox_image.startswith("quay.io/jupyter/") or "jupyter" in s.sandbox_image
    assert s.sandbox_memory_mb == 2048
    assert s.sandbox_cpus == 2.0
    assert s.sandbox_pids_limit == 256
    assert s.sandbox_network == "none"
    assert s.sandbox_container_user == ""
    assert s.sandbox_data_mounts == []
    assert s.sandbox_start_timeout == 180
    # unchanged safety defaults
    assert s.sandbox_backend == "jupyter_local"
    assert s.sandbox_execute_enabled is False


def test_sandbox_network_must_be_none():
    """The docker backend's no-egress guarantee depends on --network none, so a
    non-'none' value is rejected rather than silently weakening isolation."""
    assert BaseSettings(sandbox_network="none").sandbox_network == "none"
    for bad in ("host", "bridge", "my-net"):
        with pytest.raises(ValidationError):
            BaseSettings(sandbox_network=bad)
