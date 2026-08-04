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
    # unified backend config (replaces the old two-field enable+backend pattern)
    assert s.stateful_executor_backend == "none"


def test_stateful_executor_backend_default_and_values():
    from pydantic import ValidationError
    assert BaseSettings().stateful_executor_backend == "none"
    assert BaseSettings(stateful_executor_backend="docker").stateful_executor_backend == "docker"
    assert BaseSettings(stateful_executor_backend="local").stateful_executor_backend == "local"
    with pytest.raises(ValidationError):
        BaseSettings(stateful_executor_backend="bogus")


def test_sandbox_network_must_be_none():
    """The docker backend's no-egress guarantee depends on --network none, so a
    non-'none' value is rejected rather than silently weakening isolation."""
    assert BaseSettings(sandbox_network="none").sandbox_network == "none"
    for bad in ("host", "bridge", "my-net"):
        with pytest.raises(ValidationError):
            BaseSettings(sandbox_network=bad)
