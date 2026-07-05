"""Tests for Docker availability detection."""

import subprocess

import pytest

from agentic_cli.tools.sandbox.backends import detect


@pytest.fixture(autouse=True)
def _clear_cache():
    detect.clear_detection_cache()
    yield
    detect.clear_detection_cache()


def test_available_when_cli_and_daemon_ok(monkeypatch):
    monkeypatch.setattr(detect.shutil, "which", lambda exe: "/usr/bin/docker" if exe == "docker" else None)
    monkeypatch.setattr(detect, "_probe_daemon", lambda exe: True)
    a = detect.detect_docker()
    assert a.available is True
    assert a.runtime == "docker"


def test_unavailable_when_no_cli(monkeypatch):
    monkeypatch.setattr(detect.shutil, "which", lambda exe: None)
    a = detect.detect_docker()
    assert a.available is False
    assert a.runtime == ""
    assert "not found" in a.detail.lower()


def test_unavailable_when_daemon_down(monkeypatch):
    monkeypatch.setattr(detect.shutil, "which", lambda exe: "/usr/bin/docker" if exe == "docker" else None)
    monkeypatch.setattr(detect, "_probe_daemon", lambda exe: False)
    a = detect.detect_docker()
    assert a.available is False
    assert "daemon" in a.detail.lower()


def test_result_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(detect.shutil, "which", lambda exe: "/usr/bin/docker" if exe == "docker" else None)
    monkeypatch.setattr(detect, "_probe_daemon", lambda exe: calls.append(exe) or True)
    detect.detect_docker()
    detect.detect_docker()
    assert len(calls) == 1  # cached
