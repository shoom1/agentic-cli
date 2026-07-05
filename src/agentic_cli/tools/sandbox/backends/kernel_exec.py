"""Shared kernel execution helpers. Dependency-free (stdlib + jupyter_client)
so this module can be bind-mounted next to driver.py and imported in-container."""

from __future__ import annotations

import base64
import queue
import re
import time
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BLOCKED_MAGICS = frozenset({"pip", "system", "sx"})


def validate_code(code: str) -> tuple[bool, str]:
    """Pre-scan code for blocked shell escapes and magics."""
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!"):
            return False, "Shell commands (!) are not allowed in the sandbox"
        if stripped.startswith("%") and not stripped.startswith("%%"):
            body = stripped.lstrip("%")
            magic = body.split()[0] if body else ""
            if magic in _BLOCKED_MAGICS:
                return False, f"Magic command '%{magic}' is not allowed in the sandbox"
    return True, ""


def collect_execution(kc, msg_id: str, timeout: float, working_dir) -> dict:
    """Collect iopub messages for one execution into a result dict."""
    start = time.monotonic()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    result_value: str | None = None
    artifacts: list[str] = []
    error_text = ""

    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except (queue.Empty, TimeoutError):  # timeout waiting for kernel output
            return {
                "success": False,
                "stdout": "".join(stdout_parts),
                "stderr": "".join(stderr_parts),
                "result": result_value,
                "artifacts": artifacts,
                "execution_time": time.monotonic() - start,
                "error": f"Execution timed out after {timeout}s",
            }

        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue

        msg_type = msg.get("msg_type", "")
        content: dict[str, Any] = msg.get("content", {})

        if msg_type == "stream":
            if content.get("name") == "stderr":
                stderr_parts.append(content.get("text", ""))
            else:
                stdout_parts.append(content.get("text", ""))
        elif msg_type == "execute_result":
            result_value = content.get("data", {}).get("text/plain", "")
        elif msg_type == "display_data":
            data = content.get("data", {})
            if "image/png" in data and working_dir is not None:
                artifact_dir = working_dir / "artifacts"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                path = artifact_dir / f"plot_{len(artifacts)}.png"
                path.write_bytes(base64.b64decode(data["image/png"]))
                artifacts.append(str(path))
        elif msg_type == "error":
            error_text = _ANSI_RE.sub("", "\n".join(content.get("traceback", [])))
        elif msg_type == "status" and content.get("execution_state") == "idle":
            break

    return {
        "success": not error_text,
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
        "result": result_value,
        "artifacts": artifacts,
        "execution_time": time.monotonic() - start,
        "error": error_text,
    }
