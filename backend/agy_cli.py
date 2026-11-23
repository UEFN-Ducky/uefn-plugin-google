"""Antigravity CLI (``agy``): find it, install Google's own build, list models, sign in.

Sign-in is Claude Code's flow: Settings → Log in opens the Google page in the person's
default browser and a box for the code the Antigravity page shows; the code goes to a
hidden ``agy acp --login`` waiting on stdin. The login lives in agy's own profile, the
same one the Antigravity CLI uses in a terminal, so usage counts against the person's
Google AI plan, never an API key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Callable

# Google's release feed for the CLI (the one T3 Code and `agy update` read).
MANIFEST_URL = "https://antigravity-cli-auto-updater-974169037036.us-central1.run.app/manifests/{target}.json"
_DOWNLOAD_PREFIX = "https://storage.googleapis.com/antigravity-public/antigravity-cli/"
SIGNED_OUT_MARKERS = ("please sign in", "authentication required", "authentication failed", "not signed in")
URL_WAIT_S = 90.0  # the one-file runtime unpacks itself on its first start
SUBMIT_WAIT_S = 60.0
_URL_RE = re.compile(r"https://accounts\.google\.com/\S+")
_MODEL_ID_RE = re.compile(r"^[a-z][a-z0-9.\-]*[0-9][a-z0-9.\-]*$")

_install: dict[str, Any] = {"busy": False, "error": ""}
_install_lock = threading.Lock()
# ponytail: one Settings sign-in at a time, in this process; Log in again replaces it.
# popup: the logins table started it, so the plugin's own code box (ui/agy_login.js) asks for the code.
_login: dict[str, Any] = {"proc": None, "url": "", "out": [], "error": "", "popup": False}
_models: dict[str, Any] = {"at": 0.0, "bin": "", "rows": [], "signed_in": None}
MODELS_TTL_S = 600.0


def hidden() -> dict[str, Any]:
    from .cli_update import hidden_run_kwargs

    return hidden_run_kwargs()


# ── find / install ──────────────────────────────────────────────────────

def managed_bin() -> Path:
    """Where Ducky puts Google's build (the CLI's own install folder)."""
    return Path.home() / ".gemini" / "bin" / ("agy.exe" if os.name == "nt" else "agy")


def resolve_bin(override: str = "") -> str:
    """Google's own Antigravity program, never a look-alike: a Google AI plan may only be used
    through Antigravity itself, so a shim named ``agy`` (or a custom path) that isn't
    Google-signed is skipped. Every launch, sign-in, sign-out and model list comes through here."""
    from backend.agent.coding_agents.base import which_cli

    own = managed_bin()
    for path in (which_cli("agy", override), str(own) if own.is_file() else ""):
        if path and google_signed(path):
            return path
    return ""


_signed: dict[tuple[str, int, int], bool] = {}
_SIGNED_PS = ("$ErrorActionPreference = 'Stop'; "
              "$s = Get-AuthenticodeSignature -LiteralPath $env:DUCKY_AGY_PATH; "
              "if ($s.Status -eq 'Valid' -and $s.SignerCertificate.Subject -like 'CN=Google LLC,*') { 'google' }")


def google_signed(path: str, *, run: Callable[..., Any] = subprocess.run) -> bool:
    """Windows: a valid Authenticode signature from Google LLC (cached per file version)."""
    if os.name != "nt":
        return True  # ponytail: no signature check off Windows (Ducky only installs agy on Windows)
    try:
        st = os.stat(path)
    except OSError:
        return False
    key = (os.path.normcase(os.path.abspath(path)), st.st_mtime_ns, st.st_size)
    if key not in _signed:
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        powershell = system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        # Use Windows PowerShell's own module search path, even when Ducky was launched from PowerShell 7.
        env = {name: value for name, value in os.environ.items() if name.upper() != "PSMODULEPATH"}
        env["DUCKY_AGY_PATH"] = path
        try:
            probe = run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", _SIGNED_PS],
                        capture_output=True, text=True, timeout=60, env=env, **hidden())
        except (OSError, subprocess.SubprocessError):
            return False  # not cached: the next look tries again
        if probe.returncode != 0:
            return False
        _signed[key] = (probe.stdout or "").strip() == "google"
    return _signed[key]


def platform_target() -> str:
    import platform

    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    osname = {"nt": "windows"}.get(os.name) or ("darwin" if os.uname().sysname == "Darwin" else "linux")
    return f"{osname}_{arch}"


def read_manifest(fetch: Callable[[str], bytes] | None = None) -> dict[str, str]:
    """``{version, url, sha512}`` for this PC, refused unless it points at Google's bucket."""
    url = MANIFEST_URL.format(target=platform_target())
    raw = fetch(url) if fetch else urllib.request.urlopen(url, timeout=30).read()
    data = json.loads(raw)
    if not (isinstance(data, dict) and str(data.get("url", "")).startswith(_DOWNLOAD_PREFIX)
            and re.fullmatch(r"[0-9a-fA-F]{128}", str(data.get("sha512", "")))):
        raise ValueError("Antigravity release feed answered something unexpected")
    return {k: str(data[k]) for k in ("version", "url", "sha512")}


def install(fetch: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """Download Google's build into :func:`managed_bin`, checked against its sha512."""
    from .cli_update import install_skip_reason

    skip = install_skip_reason()
    if skip:
        return {"ok": False, "error": f"this Ducky instance does not install Antigravity ({skip})"}
    if os.name != "nt":  # ponytail: Google ships a tar.gz there; install it yourself
        return {"ok": False, "error": "Install the Antigravity CLI from antigravity.google/download."}
    with _install_lock:
        _install.update(busy=True, error="")
        try:
            man = read_manifest(fetch)
            dest = managed_bin()
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(".download")
            digest = hashlib.sha512()
            with urllib.request.urlopen(man["url"], timeout=60) as resp, open(part, "wb") as out:
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest().lower() != man["sha512"].lower():
                part.unlink(missing_ok=True)
                raise ValueError("Antigravity download did not match Google's checksum")
            os.replace(part, dest)
            refresh_agents()  # the row read "Not found" while this ran: its Log in shows now
            return {"ok": True, "cli_path": str(dest), "version": man["version"]}
        except Exception as exc:
            _install["error"] = f"Antigravity install failed: {exc}"
            return {"ok": False, "error": _install["error"]}
        finally:
            _install["busy"] = False


def refresh_agents() -> None:
    try:
        from .gemini_auth import _refresh_agents

        _refresh_agents()
    except Exception:
        pass


def install_in_background() -> None:
    if _install["busy"] or os.environ.get("PYTEST_CURRENT_TEST"):
        return
    _install["busy"] = True  # before the thread: the row's next status already says so
    threading.Thread(target=install, name="agy-install", daemon=True).start()


def install_status() -> str:
    if _install["busy"]:
        return "Downloading the Antigravity CLI from Google (about 240 MB)…"
    return str(_install["error"] or "")


# ── models (doubles as the signed-in check: one fast call) ─────────────

def parse_models(text: str) -> list[dict[str, str]]:
    """Rows from ``agy models``. ponytail: the format isn't documented; any line that starts
    with a model-like id counts, the rest of the line is its name."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        parts = line.strip().lstrip("-*•").strip().split(None, 1)
        if not parts or not _MODEL_ID_RE.match(parts[0]) or parts[0] in seen:
            continue
        seen.add(parts[0])
        name = parts[1].strip(" -–—()") if len(parts) > 1 else parts[0]
        rows.append({"id": parts[0], "name": name or parts[0], "provider": "Antigravity"})
    return rows


def signed_out(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in SIGNED_OUT_MARKERS)


def account_models(binary: str, *, env: dict[str, str] | None = None, force: bool = False,
                   run: Callable[..., Any] = subprocess.run) -> tuple[bool | None, list[dict[str, str]]]:
    """(signed in?, models) from ``agy models``; None when it couldn't tell. Cached 10 min."""
    now = time.monotonic()
    if not force and _models["bin"] == binary and now - _models["at"] < MODELS_TTL_S:
        return _models["signed_in"], list(_models["rows"])
    try:
        proc = run([binary, "models"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                   stdin=subprocess.DEVNULL, timeout=45, env={**os.environ, **(env or {})}, **hidden())
        text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        state: bool | None = False if signed_out(text) else (True if proc.returncode == 0 else None)
        rows = parse_models(proc.stdout or "") if state else []
    except (OSError, subprocess.TimeoutExpired):
        state, rows = None, []
    _models.update(at=now, bin=binary, rows=rows, signed_in=state)
    return state, list(rows)


def forget_models() -> None:
    _models.update(at=0.0, bin="")


# ── Settings sign-in (code box) ─────────────────────────────────────────

def _stop_login() -> None:
    proc = _login.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    _login.update(proc=None, url="", out=[], error="", popup=False)


def _last_line(lines: list[str]) -> str:
    for line in reversed(lines):
        if line.strip() and not _URL_RE.search(line):
            return line.strip()[:300]
    return ""


def start_login(cli_path: str = "", *, popen: Callable[..., Any] = subprocess.Popen,
                open_url: Callable[[str], Any] = webbrowser.open, wait_s: float = URL_WAIT_S) -> dict[str, Any]:
    binary = resolve_bin(cli_path)
    if not binary:
        install_in_background()
        return {"ok": False, "error": "Ducky is downloading the Antigravity CLI from Google (about 240 MB). "
                                      "Press Log in again when the row says Found."}
    _stop_login()
    try:
        proc = popen([binary, "acp", "--login"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", **hidden())
    except OSError as exc:
        return {"ok": False, "error": f"Could not start Antigravity: {exc}"}
    out: list[str] = []
    _login.update(proc=proc, out=out)
    threading.Thread(target=lambda: out.extend(proc.stdout), name="agy-login-out", daemon=True).start()
    deadline = time.monotonic() + wait_s
    url = ""
    while time.monotonic() < deadline and not url and proc.poll() is None:
        time.sleep(0.2)
        url = next((m.group(0) for line in list(out) if (m := _URL_RE.search(line))), "")
    if not url:
        err = _last_line(out) or "Antigravity did not show a Google sign-in link."
        _stop_login()
        return {"ok": False, "error": err}
    _login["url"] = url
    try:
        open_url(url)  # the person's default browser (their Chrome profile), never Ducky's window
    except Exception:
        pass
    return {"ok": True, "needs_login": True, "logged_in": False, "login_ui": "code_modal", "auth_url": url,
            "message": "Sign in with Google in your browser, then paste the code from the Antigravity page."}


def login_status() -> dict[str, Any]:
    proc = _login.get("proc")
    if proc is None:
        return {"ok": True, "logged_in": False, "needs_login": True, "login_ui": "code_modal", "auth_url": ""}
    code = proc.poll()
    if code == 0:
        _stop_login()
        forget_models()
        return {"ok": True, "logged_in": True, "needs_login": False}
    pending = {"ok": True, "logged_in": False, "needs_login": True, "login_ui": "code_modal",
               "auth_url": _login.get("url") or "", "session_alive": code is None}
    if code is not None:
        pending["error"] = _last_line(_login["out"]) or "Sign-in stopped. Press Log in again."
    return pending


def submit_code(code: str, *, wait_s: float = SUBMIT_WAIT_S) -> dict[str, Any]:
    text = (code or "").strip().strip("`")
    if not text or " " in text:
        return {"ok": False, "error": "Paste the code from the Antigravity page, or that page's full address."}
    proc = _login.get("proc")
    if proc is None or proc.poll() is not None:
        return {"ok": False, "error": "This sign-in ended. Press Log in again."}
    seen = len(_login["out"])
    try:
        proc.stdin.write(text + "\n")
        proc.stdin.flush()
    except (OSError, ValueError):
        return {"ok": False, "error": "This sign-in ended. Press Log in again."}
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.25)
    if proc.poll() == 0:
        _stop_login()
        forget_models()
        return {"ok": True, "logged_in": True, "message": "Signed in to Antigravity with Google."}
    if proc.poll() is None:
        return {"ok": False, "logged_in": False, "error": "Still waiting for Google to accept the code."}
    err = _last_line(_login["out"][seen:]) or "Google did not accept that code. Press Log in again."
    _stop_login()
    return {"ok": False, "logged_in": False, "error": err}


def popup_state() -> dict[str, Any]:
    """For the plugin's code box: open while a table-started sign-in waits for its code."""
    proc = _login.get("proc")
    alive = bool(_login.get("popup")) and proc is not None and proc.poll() is None
    return {"ok": True, "open": alive, "auth_url": _login.get("url") or "" if alive else ""}


def reopen_browser(*, open_url: Callable[[str], Any] = webbrowser.open) -> dict[str, Any]:
    url = _login.get("url") or ""
    if url:
        open_url(url)
    return {"ok": bool(url)}


def cancel_login() -> dict[str, Any]:
    _stop_login()
    return {"ok": True, "cancelled": True}


def logout(cli_path: str = "", *, env: dict[str, str] | None = None,
           popen: Callable[..., Any] = subprocess.Popen, wait_s: float = 90.0) -> dict[str, Any]:
    """ACP ``logout`` (agy has no logout command): initialize, then logout, then quit."""
    binary = resolve_bin(cli_path)
    if not binary:
        return {"ok": False, "error": "Antigravity isn't installed."}
    _stop_login()
    forget_models()
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": 1, "clientCapabilities": {}}},
        {"jsonrpc": "2.0", "id": 2, "method": "logout", "params": {}},
    ]
    try:
        proc = popen([binary, "acp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                     env={**os.environ, **(env or {})}, **hidden())
    except OSError as exc:
        return {"ok": False, "error": f"Could not start Antigravity: {exc}"}
    replies: dict[int, dict[str, Any]] = {}

    def read() -> None:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("id") in (1, 2):
                replies[msg["id"]] = msg
                if msg["id"] == 1 and "error" not in msg:
                    proc.stdin.write(json.dumps(msgs[1]) + "\n")
                    proc.stdin.flush()
                if msg["id"] == 2 or "error" in msg:
                    return

    reader = threading.Thread(target=read, name="agy-logout", daemon=True)
    reader.start()
    try:
        proc.stdin.write(json.dumps(msgs[0]) + "\n")
        proc.stdin.flush()
    except (OSError, ValueError):
        pass
    reader.join(wait_s)
    try:
        proc.kill()
    except OSError:
        pass
    done = replies.get(2) or replies.get(1) or {}
    if 2 in replies and "error" not in done:
        return {"ok": True, "logged_in": False, "message": "Signed out of Antigravity."}
    err = (done.get("error") or {}).get("message") if isinstance(done.get("error"), dict) else ""
    return {"ok": False, "error": err or "Antigravity sign-out did not finish. Try again."}
