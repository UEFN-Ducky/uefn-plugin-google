"""Gemini CLI "Login with Google" with no terminal: a hidden CLI run does the OAuth.

``gemini -p`` with no cached login and ``GOOGLE_GENAI_USE_GCA=true`` opens Google's
sign-in page in the default browser and waits for its local callback. We watch the
account's ``.gemini/oauth_creds.json``; once it exists the run is stopped and the row
reads logged in. Tokens never leave the account's own folder and are never logged.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

LOGIN_TIMEOUT_S = 300.0
# After sign-in the CLI's own first run ("Reply with OK.") tells whether Google lets this
# account use Gemini: it is waited for this long instead of stopped at once.
ACCESS_CHECK_S = 60.0
ACCESS_FILE = "ducky_access.json"  # in the account's .gemini folder
NO_ACCESS = "No Gemini access: Google no longer allows personal accounts in Gemini CLI. Add a free API key above."
NOT_FINISHED = "Sign-in didn't finish, try again."
WAITING = "Waiting for Google sign-in in your browser…"
NOT_INSTALLED = "Gemini CLI isn't installed yet. Ducky is installing it; sign-in starts by itself."

# ponytail: one sign-in per account folder, in this process; a second Log in replaces it.
_LOCK = threading.Lock()
_STATE: dict[str, dict[str, Any]] = {}  # .gemini folder -> {proc, pending, error}
# Log in's install of a missing CLI. The row reads Installing while it runs and the app starts
# the sign-in by itself when it ends (one click); it used to say "press Log in again in a minute".
_INSTALL: dict[str, Any] = {"busy": False, "error": ""}


def _key(dot: Path) -> str:
    return os.path.normcase(str(dot))


def login_state(dot: Path) -> dict[str, Any]:
    """``{pending: True}`` while signing in, ``{error}`` after a failed try, else ``{}``."""
    with _LOCK:
        st = _STATE.get(_key(dot))
    if not st:
        return {}
    if st["pending"]:
        return {"pending": True, "repairing": True} if st["repairing"] else {"pending": True}
    return {"error": st["error"]} if st["error"] else {}


def install_state() -> dict[str, Any]:
    """``{pending, installing}`` while Log in's install runs, ``{error}`` after it failed, else ``{}``."""
    if _INSTALL["busy"]:
        return {"pending": True, "installing": True}
    return {"error": _INSTALL["error"]} if _INSTALL["error"] else {}


def _install_in_background() -> None:
    from .cli_update import update_cli

    def run() -> None:
        try:
            res = update_cli() or {}
            _INSTALL["error"] = "" if res.get("ok") else f"Gemini CLI install failed: {str(res.get('error') or 'no reason given')[:300]}"
        except Exception as exc:  # never leave the row on Installing
            _INSTALL["error"] = f"Gemini CLI install failed: {exc}"
        finally:
            _INSTALL["busy"] = False

    if _INSTALL["busy"]:
        return
    _INSTALL.update(busy=True, error="")  # before the thread: the row's first status sees it
    threading.Thread(target=run, name="gemini-cli-install", daemon=True).start()


def _stop(proc: Any) -> None:
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass


def write_access(dot: Path, ok: bool, reason: str = "") -> None:
    """What Google said about this account last: allowed, or refused (and why)."""
    try:
        dot.mkdir(parents=True, exist_ok=True)
        (dot / ACCESS_FILE).write_text(json.dumps({"ok": bool(ok), "reason": reason[:300], "at": time.time()}), encoding="utf-8")
    except OSError:
        pass


def read_access(dot: Path) -> dict[str, Any] | None:
    try:
        got = json.loads((dot / ACCESS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return got if isinstance(got, dict) else None


def clear_access(dot: Path) -> None:
    try:
        (dot / ACCESS_FILE).unlink()
    except OSError:
        pass


def start_browser_login(
    config_dir: str | None,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    timeout_s: float = LOGIN_TIMEOUT_S,
    poll_s: float = 0.5,
    access_check_s: float | None = None,
) -> dict[str, Any]:
    """Start the hidden sign-in for this account (``None`` = the CLI's own home)."""
    from .cli_update import hidden_run_kwargs, resolve_bin
    from .gemini_cli_adapter import _dot_gemini, account_env

    binary = resolve_bin("")
    if not binary:
        _install_in_background()
        return {"error": NOT_INSTALLED}
    _INSTALL["error"] = ""  # it is installed now: an older failed install says nothing about this sign-in
    dot = _dot_gemini(config_dir)
    dot.mkdir(parents=True, exist_ok=True)
    clear_access(dot)  # a new sign-in: what Google says about it comes from this run
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    env.update(account_env(config_dir))
    # ponytail: a settings.json that picks another auth type wins over this env; that
    # login then times out with NOT_FINISHED. Write security.auth.selectedType if it bites.
    env["GOOGLE_GENAI_USE_GCA"] = "true"
    key = _key(dot)
    with _LOCK:
        old = _STATE.pop(key, None)
    if old:
        _stop(old["proc"])
    def launch() -> Any:
        """The hidden run; its output (no tokens: the CLI prints none) is kept for errors."""
        proc = popen(
            [resolve_bin("") or binary, "-p", "Reply with OK."],
            env=env,
            cwd=str(dot),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **hidden_run_kwargs(),
        )
        # Headless runs ask "Opening authentication page… Do you want to continue? [Y/n]"
        # on stdin before opening the browser; an empty stdin never answers and it waits.
        stdin = getattr(proc, "stdin", None)
        if stdin is not None:
            try:
                stdin.write("y\n")
                stdin.flush()
            except (OSError, ValueError):
                pass
        out: list[str] = []
        proc.ducky_output = out
        if getattr(proc, "stdout", None) is not None:
            proc.ducky_reader = threading.Thread(target=lambda: out.extend(proc.stdout), name="gemini-login-out", daemon=True)
            proc.ducky_reader.start()
        return proc

    try:
        proc = launch()
    except OSError as exc:
        return {"error": f"Could not start Gemini CLI: {exc}"}
    state = {"proc": proc, "pending": True, "repairing": False, "error": ""}
    with _LOCK:
        _STATE[key] = state
    threading.Thread(
        target=_watch,
        args=(state, dot / "oauth_creds.json", timeout_s, poll_s, launch,
              ACCESS_CHECK_S if access_check_s is None else access_check_s),
        name="gemini-login", daemon=True,
    ).start()
    return {"message": WAITING}


def _output(proc: Any) -> str:
    reader = getattr(proc, "ducky_reader", None)
    if reader is not None:
        reader.join(2.0)  # the process ended: its last lines are on their way
    return "".join(getattr(proc, "ducky_output", []))


def _watch(state: dict[str, Any], creds: Path, timeout_s: float, poll_s: float, launch: Callable[[], Any],
           access_check_s: float = 0.0) -> None:
    from .cli_update import BROKEN_INSTALL, error_line, is_broken_install_error, update_cli

    deadline = time.monotonic() + timeout_s
    repaired = False
    error = NOT_FINISHED
    while True:
        proc = state["proc"]
        while time.monotonic() < deadline and not creds.is_file() and proc.poll() is None:
            time.sleep(poll_s)
        if creds.is_file() or proc.poll() is None:
            break  # signed in, or timed out
        text = _output(proc)  # the CLI gave up: cancelled in the browser, or it can't start
        if not is_broken_install_error(text):
            line = error_line(text)
            error = f"{NOT_FINISHED} {line}" if line else NOT_FINISHED
            break
        if repaired:  # still broken after one reinstall: say so, never loop
            error = f"{BROKEN_INSTALL}: {error_line(text)}"
            break
        repaired = True
        with _LOCK:
            state["repairing"] = True
        upd = update_cli("")
        with _LOCK:
            state["repairing"] = False
        if not upd.get("ok"):
            error = f"{BROKEN_INSTALL}: {error_line(str(upd.get('error') or ''))}"
            break
        try:
            state["proc"] = launch()
        except OSError as exc:
            error = f"Could not start Gemini CLI: {exc}"
            break
    ok = creds.is_file()
    with _LOCK:
        state["pending"] = False
        state["error"] = "" if ok else error
    if not ok:
        _stop(state["proc"])
        return
    _refresh_agents()  # the row reads signed in now; whether Google allows it follows
    proc = state["proc"]
    end = time.monotonic() + max(0.0, access_check_s)
    while proc.poll() is None and time.monotonic() < end:
        time.sleep(poll_s)
    finished = proc.poll() is not None
    _stop(proc)
    if not finished:
        return  # no answer in time: unknown, the first chat tells
    from .gemini_cli_adapter import personal_login_refused

    text = _output(proc)
    if personal_login_refused(text):
        write_access(creds.parent, False, NO_ACCESS)
    elif getattr(proc, "returncode", 1) == 0:
        write_access(creds.parent, True)
    _refresh_agents()


def _refresh_agents() -> None:
    """Settings' Google dot and the model picker read a 10-minute detect cache: re-probe now."""
    try:
        from backend.agent.coding_agents.base import invalidate_detect_cache, kick_detect_refresh

        invalidate_detect_cache()
        kick_detect_refresh()
    except Exception:
        pass
