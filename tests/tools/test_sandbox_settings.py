"""Tests for docker sandbox settings defaults."""

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
