"""Keep the Gemini CLI current for Ducky-only users."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

_BIN = "gemini"
_NPM_PKG = "@google/gemini-cli"
_VERSION_RE = re.compile(r"(\d+(?:\.\d+)+)")
_REQUIRED_RE = re.compile(
    r"does not support this model;\s*version\s+(\d+(?:\.\d+)+)\s+or newer",
    re.I,
)
_MISSING_MARKERS = (
    "is not recognized",
    "no such file or directory",
    "cannot find the path",
    "not found in path",
)
_update_lock = threading.Lock()
_busy = False
_last: dict[str, Any] = {}


def plugin_package_version() -> str:
    path = Path(__file__).resolve().parent.parent / "plugin.json"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("version") or "")
    except (OSError, json.JSONDecodeError, TypeError):
        return ""


def parse_version_tuple(text: str) -> tuple[int, ...] | None:
    m = _VERSION_RE.search(text or "")
    if not m:
        return None
    try:
        return tuple(int(p) for p in m.group(1).split("."))
    except ValueError:
        return None


def is_cli_too_old_error(*chunks: str) -> bool:
    text = "\n".join(c or "" for c in chunks)
    if _REQUIRED_RE.search(text):
        return True
    low = text.lower()
    return "does not support this model" in low and "update" in low


def is_cli_missing_error(*chunks: str) -> bool:
    low = "\n".join(c or "" for c in chunks).lower()
    return any(m in low for m in _MISSING_MARKERS)


def should_heal_launch(*chunks: str) -> bool:
    return is_cli_too_old_error(*chunks) or is_cli_missing_error(*chunks)


def resolve_bin(override: str = "") -> str:
    from backend.agent.coding_agents.base import which_cli

    return which_cli(_BIN, override)


def stamp_path() -> Path:
    from frontend.settings import default_app_data_dir

    return default_app_data_dir() / "coding_agents" / "gemini_cli.json"


def read_stamp() -> dict[str, Any]:
    try:
        data = json.loads(stamp_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_stamp(data: dict[str, Any]) -> None:
    path = stamp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def status_text() -> str:
    if _busy:
        return "Updating Gemini CLI…"
    return str(_last.get("message") or "")


def _env() -> dict[str, str]:
    try:
        from frontend.ui_web.terminal.path_env import env_with_fresh_path, refresh_process_path

        refresh_process_path()
        return env_with_fresh_path()
    except Exception:
        return dict(os.environ)


def _run(argv: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": timeout_s,
        "encoding": "utf-8",
        "errors": "replace",
        "env": _env(),
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(argv, **kwargs)


def read_cli_version(binary: str) -> str:
    if not (binary or "").strip():
        return ""
    try:
        proc = _run([binary, "--version"], timeout_s=20)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    tup = parse_version_tuple((proc.stdout or "") + "\n" + (proc.stderr or ""))
    return ".".join(str(p) for p in tup) if tup else ""


def _npm_install() -> dict[str, Any]:
    from backend.agent.coding_agents.base import which_cli

    npm = which_cli("npm") or which_cli("npm.cmd")
    if not npm:
        return {"ok": False, "error": "npm not found — Ducky cannot install Gemini CLI without Node.js"}
    try:
        proc = _run([npm, "install", "-g", _NPM_PKG], timeout_s=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout or "npm install failed").strip()}
    return {"ok": True}


def update_cli(override: str = "") -> dict[str, Any]:
    global _busy, _last
    with _update_lock:
        _busy = True
        try:
            before_bin = resolve_bin(override)
            before = read_cli_version(before_bin) if before_bin else ""
            installed = _npm_install()
            if not installed.get("ok"):
                _last = installed
                return installed
            after_bin = resolve_bin(override) or before_bin
            after = read_cli_version(after_bin) if after_bin else ""
            result = {
                "ok": bool(after_bin),
                "cli_path": after_bin,
                "version_before": before,
                "version_after": after,
                "message": f"Gemini CLI {after}" if after else "Gemini CLI updated",
                "error": "" if after_bin else "Gemini CLI still not found after install/update",
            }
            if result["ok"]:
                write_stamp(
                    {
                        "plugin_version": plugin_package_version(),
                        "cli_version": after,
                        "updated_at": time.time(),
                        "ok": True,
                    }
                )
            _last = result
            return result
        finally:
            _busy = False


def needs_plugin_load_update() -> bool:
    if not resolve_bin():
        return True
    stamp = read_stamp()
    if not stamp.get("ok"):
        return True
    return str(stamp.get("plugin_version") or "") != plugin_package_version()


def schedule_cli_update_on_plugin_load() -> None:
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("DUCKY_SKIP_CLI_UPDATE"):
        return
    try:
        if not needs_plugin_load_update():
            return
    except Exception:
        return

    def _worker() -> None:
        try:
            update_cli("")
        except Exception:
            pass

    threading.Thread(target=_worker, name="gemini-cli-update", daemon=True).start()
