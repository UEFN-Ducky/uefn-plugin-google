"""Gemini CLI adapter — headless ``gemini -p`` + stream-json events.

Install: ``npm install -g @google/gemini-cli``. Auth via ``GEMINI_API_KEY``
(from the Google gateway API key) or an existing Gemini CLI login.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import threading
from pathlib import Path
from typing import Any

from backend.agent.coding_agents.base import (
    CodingAgentCapabilities,
    CodingAgentInfo,
    CodingAgentLaunchResult,
    which_cli,
)
from backend.agent.coding_agents.cli_shared import finalize_cli_turn, truncate_tool_result
from backend.agent.coding_agents.proc_exec import run_streaming_process
from backend.agent.coding_agents.settings_helpers import coding_agent_cfg

_GEMINI_INSTALL = "npm install -g @google/gemini-cli"


#: The CLI's own aliases: it resolves each to the newest model the signed-in account may use,
#: so a Google login (no API key, no models.list) still has a picker that never goes stale.
CLI_ALIAS_ROWS = (
    {"id": "auto", "name": "Auto", "provider": "Gemini CLI"},
    {"id": "pro", "name": "Pro", "provider": "Gemini CLI"},
    {"id": "flash", "name": "Flash", "provider": "Gemini CLI"},
    {"id": "flash-lite", "name": "Flash Lite", "provider": "Gemini CLI"},
)


def _gemini_cli_model_rows() -> list[dict[str, str]]:
    """Live Gemini models.list with the saved key; the CLI's aliases without one (or when it fails)."""
    aliases = [dict(r) for r in CLI_ALIAS_ROWS]
    try:
        from backend.agent.secrets import get_key

        key = (get_key("gemini") or "").strip()
    except Exception:
        key = ""
    if not key:
        return aliases
    try:
        from .model_fetch import fetch_models

        rows = [
            {"id": info.id, "name": info.display_name or info.id, "provider": "Gemini CLI"}
            for info in fetch_models(key)
            if info.id
        ]
    except Exception:
        rows = []
    return aliases + rows


def _gemini_missing_status() -> str:
    return (
        "Gemini CLI not found — Ducky will install it automatically. "
        "If this stays, send another message or Settings → Store → Update Google. "
        f"Manual fallback: {_GEMINI_INSTALL}"
    )


# ── accounts ────────────────────────────────────────────────────────────
# A login account is its own Gemini CLI home: the CLI reads ``$GEMINI_CLI_HOME/.gemini``
# instead of ``~/.gemini``. Key accounts belong to the ``gemini`` provider (core's list).

HOME_ENV = "GEMINI_CLI_HOME"
SHARED_DIRS = ("extensions", "commands", "tmp", "history")
SHARED_FILES = ("settings.json", "GEMINI.md")
_CREDS = ("oauth_creds.json", "google_accounts.json")


def _dot_gemini(config_dir: str | Path | None) -> Path:
    """The account's ``.gemini`` folder. Core passes None for ``default``: the CLI's own home."""
    home = config_dir or os.environ.get(HOME_ENV) or Path.home()
    return Path(home) / ".gemini"


def account_env(config_dir: str | None) -> dict[str, str]:
    """Environment that runs ``gemini`` as the login kept in ``config_dir`` ({} for ``default``)."""
    if not config_dir:
        return {}
    folder = _dot_gemini(config_dir)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        from backend.agent.accounts import share_config

        share_config(Path.home() / ".gemini", folder, SHARED_DIRS, SHARED_FILES)
    except Exception:
        pass  # ponytail: older core without accounts.py; the login still works unshared.
    return {HOME_ENV: str(config_dir)}


def account_login(config_dir: str | None) -> dict[str, Any]:
    """Gemini CLI has no ``auth login``: its first interactive run asks for the Google login."""
    from .cli_update import resolve_bin

    return {"argv": [resolve_bin("") or "gemini"], "env": account_env(config_dir)}


def account_status(config_dir: str | None) -> dict[str, Any]:
    folder = _dot_gemini(config_dir)
    status: dict[str, Any] = {"loggedIn": (folder / "oauth_creds.json").is_file()}
    try:
        email = json.loads((folder / "google_accounts.json").read_text(encoding="utf-8")).get("active")
        if status["loggedIn"] and isinstance(email, str) and email:
            status["email"] = email
    except Exception:
        pass
    if not status["loggedIn"]:
        from .gemini_auth import login_state

        status.update(login_state(folder))  # pending / error of a browser sign-in
    return status


def account_browser_login(config_dir: str | None) -> dict[str, Any]:
    """Settings → Log in: Google sign-in in the browser, no terminal tab."""
    from .gemini_auth import start_browser_login

    return start_browser_login(config_dir)


def is_gemini_logged_in() -> bool:
    """A Google login in the CLI's home, or an API key it can run on (env or the saved Google key)."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return True
    if account_status(None).get("loggedIn"):
        return True
    return bool(pick_gemini_key({})[1])


def account_logout(config_dir: str | None) -> None:
    for name in _CREDS:
        try:
            (_dot_gemini(config_dir) / name).unlink()
        except OSError:
            pass
    from .gemini_auth import _refresh_agents

    _refresh_agents()


# "Quota exceeded for quota metric …", "429 Too Many Requests", "RESOURCE_EXHAUSTED",
# "You have exhausted your capacity on this model", "Rate limit exceeded".
_LIMIT_RE = re.compile(
    r"quota exceeded|resource[_ ]exhausted|\b429\b|too many requests|rate limit"
    r"|exhausted your (?:daily )?(?:quota|capacity)",
    re.I,
)


def is_limit_error(text: str) -> bool:
    return isinstance(text, str) and bool(_LIMIT_RE.search(text))


def pick_gemini_key(env: dict[str, str]) -> tuple[str, str]:
    """(key account id, key) for this turn; ``("", "")`` when a login account or a key is already set."""
    if env.get("GEMINI_API_KEY") or env.get(HOME_ENV):
        return "", ""
    try:
        from backend.agent import accounts

        acc = accounts.pick_account("gemini")
        key = (accounts.account_key("gemini", acc) or "").strip()
        if key:
            return acc, key
    except Exception:
        pass
    try:
        from backend.agent.secrets import get_key

        return "default", (get_key("gemini") or "").strip()
    except Exception:
        return "", ""


def note_key_limit(account_id: str, *texts: Any) -> None:
    """A turn on this key hit its quota: the next turn moves to the next key."""
    if not account_id or not any(is_limit_error(t) for t in texts if t):
        return
    try:
        from backend.agent.accounts import mark_exhausted

        mark_exhausted("gemini", account_id)
    except Exception:
        pass


def gemini_include_dirs(cwd: str) -> list[str]:
    """Added projects besides cwd, for --include-directories."""
    try:
        from frontend.ui_web.recent_projects import load_recent_projects
    except Exception:
        return []
    import os

    cwd_key = os.path.normcase(str(Path(cwd).resolve()))
    out: list[str] = []
    for item in load_recent_projects():
        path = Path(item)
        if not path.is_dir():
            continue
        if os.path.normcase(str(path.resolve())) == cwd_key:
            continue
        out.append(str(path))
    return out


def build_gemini_argv(
    *,
    binary: str,
    prompt: str,
    model: str,
    extra_args: str,
    auto_approve: bool = True,
    include_dirs: list[str] | None = None,
) -> list[str]:
    argv = [binary, "-p", prompt]
    if auto_approve:
        # -y approves every action, including file writes and shell commands.
        argv.append("-y")
    for directory in include_dirs or []:
        argv.extend(["--include-directories", directory])
    # Ducky's tools only. Gemini merges a project's .gemini/settings.json servers in, and
    # Epic's ModelContextProtocol.GenerateClientConfig writes one with unreal-mcp: a line
    # to the editor past Ducky.
    argv.extend(["--allowed-mcp-server-names", "uefn"])
    argv.extend(["--output-format", "stream-json"])
    mid = (model or "").strip()
    if mid and mid.lower() not in ("default", "auto"):
        argv.extend(["-m", mid])
    extra = (extra_args or "").strip()
    if extra:
        argv.extend(shlex.split(extra, posix=False))
    return argv


class _GeminiStream:
    def __init__(self, conv_id: str, run_id: str, push: Any) -> None:
        self.conv_id = conv_id
        self.run_id = run_id
        self.push = push
        self.text_parts: list[str] = []
        self.blocks: list[dict[str, Any]] = []
        self.session_id = ""
        self.error_text = ""
        self.usage: dict[str, Any] = {}
        self._open_tools: dict[str, str] = {}

    def on_line(self, line: str) -> None:
        text = (line or "").strip()
        if not text:
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        et = str(data.get("type") or "").strip().lower()
        if et == "init":
            sid = str(data.get("session_id") or data.get("sessionId") or "").strip()
            if sid:
                self.session_id = sid
            return
        if et == "message":
            role = str(data.get("role") or "assistant").lower()
            if role != "assistant":
                return
            chunk = str(data.get("content") or data.get("text") or "")
            if not chunk:
                return
            self.text_parts.append(chunk)
            self.push(
                {
                    "type": "assistant_delta",
                    "text": chunk,
                    "conv_id": self.conv_id,
                    "run_id": self.run_id,
                }
            )
            return
        if et == "tool_use":
            tid = str(data.get("tool_id") or data.get("id") or data.get("call_id") or "")
            name = str(data.get("tool_name") or data.get("name") or "tool")
            args = data.get("parameters") or data.get("args") or {}
            if not isinstance(args, dict):
                args = {"raw": args}
            if tid:
                self._open_tools[tid] = name
            self.push(
                {
                    "type": "tool_start",
                    "conv_id": self.conv_id,
                    "run_id": self.run_id,
                    "tool": {"id": tid or name, "name": name, "args": args},
                }
            )
            return
        if et == "tool_result":
            tid = str(data.get("tool_id") or data.get("id") or data.get("call_id") or "")
            name = self._open_tools.pop(tid, str(data.get("tool_name") or "tool"))
            result = truncate_tool_result(str(data.get("content") or data.get("result") or ""))
            self.blocks.append(
                {"type": "tool", "name": name, "id": tid, "result": result, "ok": True}
            )
            self.push(
                {
                    "type": "tool_done",
                    "conv_id": self.conv_id,
                    "run_id": self.run_id,
                    "success": True,
                    "tool": {"id": tid or name, "name": name, "result": result},
                }
            )
            return
        if et == "error":
            msg = str(data.get("message") or data.get("error") or "").strip()
            if msg:
                self.error_text = msg
            return
        if et == "result":
            stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
            if stats:
                self.usage = {
                    "input_tokens": int(stats.get("input") or stats.get("input_tokens") or 0),
                    "output_tokens": int(stats.get("output") or stats.get("output_tokens") or 0),
                }
            resp = str(data.get("response") or "").strip()
            if resp and not self.text_parts:
                self.text_parts.append(resp)
                self.push(
                    {
                        "type": "assistant_delta",
                        "text": resp,
                        "conv_id": self.conv_id,
                        "run_id": self.run_id,
                    }
                )


class GeminiCliAdapter:
    id = "gemini_cli"
    label = "Gemini CLI"
    capabilities = CodingAgentCapabilities(
        terminal_agent=True,
        chat_api=False,
        a2a=True,
        # headless `-p` takes no MCP flag, so launch() cannot forward the host's
        # config — claiming injection would hide that UEFN tools are unavailable.
        mcp_inject=False,
        needs_api_key=False,
        needs_cli=True,
        resume=False,
    )

    def _auto_approve(self) -> bool:
        try:
            from frontend.settings import PanelSettings

            cfg = coding_agent_cfg(PanelSettings.load(), self.id)
        except Exception:
            return True
        return bool(cfg.get("auto_approve", True))

    def detect(self, settings: Any) -> CodingAgentInfo:
        cfg = coding_agent_cfg(settings, self.id)
        enabled = bool(cfg.get("enabled", True))
        override = str(cfg.get("cli_path") or "")
        path = which_cli("gemini", override)
        default_args = str(cfg.get("default_args") or "")
        logged_in: bool | None = None
        if path:
            from .cli_update import read_cli_version, status_text

            logged_in = is_gemini_logged_in()
            ver = read_cli_version(path)
            extra = status_text()
            status = f"Found: {path}"
            if ver:
                status += f" · v{ver}"
            status += " · logged in" if logged_in else " · not logged in"
            if extra and extra not in status:
                status += f" · {extra}"
            available = enabled
        else:
            status = _gemini_missing_status()
            available = False
        info = CodingAgentInfo(
            id=self.id,
            label=self.label,
            enabled=enabled,
            available=available,
            status=status,
            cli_path=path or override,
            default_args=default_args,
            capabilities=self.capabilities,
            models=_gemini_cli_model_rows(),
        )
        # Unset reads as "no account to sign in to" and showed Gemini Connected with no login.
        info.logged_in = logged_in  # type: ignore[attr-defined]
        return info

    def launch(
        self,
        *,
        prompt: str,
        system_prompt: str,
        cwd: str,
        conv_id: str,
        model: str,
        mcp_config_path: str,
        extra_args: str,
        cli_path: str,
        env: dict[str, str],
        push: Any,
        session_id: str = "",
        run_id: str = "",
        cancel: threading.Event | None = None,
        timeout_s: float = 0.0,
        image_paths: list[str] | None = None,
    ) -> CodingAgentLaunchResult:
        del mcp_config_path, session_id, image_paths  # ponytail: headless -p has no resume/MCP flags yet
        model_id = (model or "").strip()
        if not model_id or model_id.lower() == "default":
            return CodingAgentLaunchResult(
                ok=False,
                error="No Gemini CLI model selected. Pick a concrete model for this chat.",
                status="error",
            )
        from .cli_update import resolve_bin, should_heal_launch, update_cli

        binary = resolve_bin(cli_path)
        if not binary:
            push(
                {
                    "type": "status",
                    "text": "Gemini CLI missing — installing automatically…",
                    "conv_id": conv_id,
                    "run_id": run_id,
                }
            )
            upd = update_cli(cli_path)
            binary = resolve_bin(cli_path) or str(upd.get("cli_path") or "")
            if not binary:
                return CodingAgentLaunchResult(
                    ok=False,
                    error=str(upd.get("error") or "Gemini CLI not found and auto-install failed"),
                    status="error",
                )
        full_prompt = prompt
        if system_prompt.strip():
            full_prompt = system_prompt.strip() + "\n\n" + prompt

        # Prefer the Google gateway API key (the picked key account) unless a login account runs.
        key_account, key = pick_gemini_key(env)
        if key:
            env = {**env, "GEMINI_API_KEY": key}

        argv = build_gemini_argv(
            binary=binary,
            prompt=full_prompt,
            model=model_id,
            extra_args=extra_args,
            auto_approve=self._auto_approve(),
            include_dirs=gemini_include_dirs(cwd),
        )
        state = _GeminiStream(conv_id, run_id, push)
        push({"type": "status", "text": "Starting Gemini CLI…", "conv_id": conv_id, "run_id": run_id})
        proc = run_streaming_process(
            argv=argv,
            cwd=cwd,
            env_extra=env,
            conv_id=conv_id,
            on_line=state.on_line,
            timeout_s=timeout_s,
            cancel=cancel,
        )
        streamed_all = "".join(state.text_parts).strip()
        reply = streamed_all
        result = finalize_cli_turn(
            proc=proc,
            reply=reply,
            streamed=bool(streamed_all) or bool(state.blocks),
            blocks=state.blocks,
            session_id="",
            new_session=state.session_id,
            usage=state.usage,
            agent_label="Gemini CLI",
            timeout_s=timeout_s,
            error_text=state.error_text,
        )
        if not result.ok:
            note_key_limit(key_account, result.error, state.error_text, proc.stderr_tail)
        if result.ok or not should_heal_launch(
            result.error or "",
            result.reply_text or "",
            proc.stderr_tail,
            proc.raw_tail,
            state.error_text,
        ):
            return result
        push(
            {
                "type": "status",
                "text": "Gemini CLI is stale or missing — updating automatically…",
                "conv_id": conv_id,
                "run_id": run_id,
            }
        )
        upd = update_cli(binary)
        if not upd.get("ok"):
            result.error = (
                (result.error or "")
                + "\n\nDucky tried to update Gemini CLI automatically and failed: "
                + str(upd.get("error") or "unknown")
            )
            return result
        binary = resolve_bin(cli_path) or str(upd.get("cli_path") or binary)
        argv[0] = binary
        state = _GeminiStream(conv_id, run_id, push)
        proc = run_streaming_process(
            argv=argv,
            cwd=cwd,
            env_extra=env,
            conv_id=conv_id,
            on_line=state.on_line,
            timeout_s=timeout_s,
            cancel=cancel,
        )
        streamed_all = "".join(state.text_parts).strip()
        retried = finalize_cli_turn(
            proc=proc,
            reply=streamed_all,
            streamed=bool(streamed_all) or bool(state.blocks),
            blocks=state.blocks,
            session_id="",
            new_session=state.session_id,
            usage=state.usage,
            agent_label="Gemini CLI",
            timeout_s=timeout_s,
            error_text=state.error_text,
        )
        from .cli_update import BROKEN_INSTALL, error_line, is_broken_install_error

        tails = (retried.error or "", proc.stderr_tail, proc.raw_tail, state.error_text)
        if not retried.ok and is_broken_install_error(*tails):  # once only: never a reinstall loop
            retried.error = f"{BROKEN_INSTALL}: {error_line(chr(10).join(t or '' for t in tails))}"
        return retried
