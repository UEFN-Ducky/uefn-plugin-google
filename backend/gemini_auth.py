"""Gemini CLI "Login with Google" with no terminal: a hidden CLI run does the OAuth.

``gemini -p`` with no cached login and ``GOOGLE_GENAI_USE_GCA=true`` opens Google's
sign-in page in the default browser and waits for its local callback. We watch the
account's ``.gemini/oauth_creds.json``; once it exists the run is stopped and the row
reads logged in. Tokens never leave the account's own folder and are never logged.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

LOGIN_TIMEOUT_S = 300.0
NOT_FINISHED = "Sign-in didn't finish, try again."
WAITING = "Waiting for Google sign-in in your browser…"
NOT_INSTALLED = "Gemini CLI isn't installed yet. Ducky is installing it now; press Log in again in a minute."

# ponytail: one sign-in per account folder, in this process; a second Log in replaces it.
_LOCK = threading.Lock()
_STATE: dict[str, dict[str, Any]] = {}  # .gemini folder -> {proc, pending, error}


def _key(dot: Path) -> str:
    return os.path.normcase(str(dot))


def login_state(dot: Path) -> dict[str, Any]:
    """``{pending: True}`` while signing in, ``{error}`` after a failed try, else ``{}``."""
    with _LOCK:
        st = _STATE.get(_key(dot))
    if not st:
        return {}
    if st["pending"]:
        return {"pending": True}
    return {"error": st["error"]} if st["error"] else {}


def _install_in_background() -> None:
    from .cli_update import update_cli

    threading.Thread(target=update_cli, name="gemini-cli-install", daemon=True).start()


def _stop(proc: Any) -> None:
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass


def start_browser_login(
    config_dir: str | None,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    timeout_s: float = LOGIN_TIMEOUT_S,
    poll_s: float = 0.5,
) -> dict[str, Any]:
    """Start the hidden sign-in for this account (``None`` = the CLI's own home)."""
    from .cli_update import hidden_run_kwargs, resolve_bin
    from .gemini_cli_adapter import _dot_gemini, account_env

    binary = resolve_bin("")
    if not binary:
        _install_in_background()
        return {"error": NOT_INSTALLED}
    dot = _dot_gemini(config_dir)
    dot.mkdir(parents=True, exist_ok=True)
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
    try:
        proc = popen(
            [binary, "-p", "Reply with OK."],
            env=env,
            cwd=str(dot),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_run_kwargs(),
        )
    except OSError as exc:
        return {"error": f"Could not start Gemini CLI: {exc}"}
    state = {"proc": proc, "pending": True, "error": ""}
    with _LOCK:
        _STATE[key] = state
    threading.Thread(
        target=_watch, args=(state, dot / "oauth_creds.json", timeout_s, poll_s), name="gemini-login", daemon=True
    ).start()
    return {"message": WAITING}


def _watch(state: dict[str, Any], creds: Path, timeout_s: float, poll_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    proc = state["proc"]
    while time.monotonic() < deadline and not creds.is_file():
        if proc.poll() is not None:
            break  # the CLI gave up (cancelled in the browser, or an error)
        time.sleep(poll_s)
    ok = creds.is_file()
    _stop(proc)  # signed in: the prompt it would answer is not needed
    with _LOCK:
        state["pending"] = False
        state["error"] = "" if ok else NOT_FINISHED
