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
    """The CLI's own aliases, with or without a key. It used to repeat every API model here
    once a key was saved: people on an API key picked one of those rows and ran Gemini CLI
    without knowing (its 429 retries and per-turn overhead burned their free quota)."""
    return [dict(r) for r in CLI_ALIAS_ROWS]


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
    from .gemini_auth import read_access

    access = read_access(folder)
    if access is not None and access.get("ok") is False:
        # Signed in, but Google refuses this account (personal accounts, June 2026):
        # the row says so instead of a green Default.
        status["access"] = "none"
        status["error"] = str(access.get("reason") or "No Gemini access for this account.")
    return status


def account_browser_login(config_dir: str | None) -> dict[str, Any]:
    """Settings → Log in: Google sign-in in the browser, no terminal tab."""
    from .gemini_auth import start_browser_login

    return start_browser_login(config_dir)


def is_gemini_logged_in() -> bool:
    """A Google login in the CLI's home, or an API key it can run on (env or the saved Google key)."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return True
    own = account_status(None)
    if own.get("loggedIn") and own.get("access") != "none":
        return True
    return bool(pick_gemini_key({})[1])


def account_logout(config_dir: str | None) -> None:
    for name in _CREDS:
        try:
            (_dot_gemini(config_dir) / name).unlink()
        except OSError:
            pass
    from .gemini_auth import _refresh_agents, clear_access

    clear_access(_dot_gemini(config_dir))
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


# Keeps "rate limit" and "429" so is_limit_error still reads it (core's account switch).
KEY_RATE_LIMITED = (
    "This Gemini key hit Google's rate limit (429). Free keys allow only a few requests a "
    "minute: wait a minute and send again, pick a Flash model, or add another key in "
    "Settings → LLMs → Google. This chat runs through Gemini CLI, which uses many more tokens "
    "per message than the API: to use the API directly, pick the model under Google in the "
    "model picker."
)
# A chat left on a Gemini CLI row from when it repeated the API models (before 1.0.47).
API_MODEL_ON_CLI = (
    "This chat runs through Gemini CLI. For fewer tokens per message, pick this model under "
    "Google in the model picker (the direct API)."
)
LOGIN_RATE_LIMITED = (
    "This Google account hit its Gemini rate limit (429). Wait a minute and send again, or "
    "add an API key in Settings → LLMs → Google."
)
# What the CLI says when --resume names a session it no longer has.
STALE_SESSION_MARKERS = ("session not found", "no session", "invalid session", "could not find session",
                         "error resuming session", "no previous sessions")


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


KEY_URL = "https://aistudio.google.com/apikey"
PERSONAL_LOGIN_REFUSED = (
    "Google no longer lets personal Google accounts use Gemini CLI (since June 2026). "
    f"Add a free Gemini API key instead ({KEY_URL}) in Settings → LLMs → Google → Add API key: "
    "Gemini CLI then runs on the key. Company (paid Code Assist) accounts still sign in."
)


def personal_login_refused(*chunks: str) -> bool:
    """Google refused a free personal login (IneligibleTierError / UNSUPPORTED_CLIENT)."""
    low = "\n".join(c or "" for c in chunks).lower()
    return "ineligibletiererror" in low or "no longer supported for gemini code assist for individuals" in low


def run_env(env: dict[str, str], key: str, *, auto_approve: bool) -> dict[str, str]:
    """A chat run's environment. With a key: the key. Without one: the Google login
    (``GOOGLE_GENAI_USE_GCA``), else headless runs exit 41 "Please set an Auth method".
    Auto-approve also trusts the folder, else the CLI drops ``-y`` ("current folder is
    not trusted") and every tool call is refused headless."""
    out = dict(env)
    if key:
        out["GEMINI_API_KEY"] = key
    elif not out.get("GEMINI_API_KEY") and not out.get("GOOGLE_API_KEY"):
        out["GOOGLE_GENAI_USE_GCA"] = "true"
    if auto_approve:
        out["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
    return out


#: The longest message kept on the command line; Windows' limit is 32,767 for the whole line.
ARG_PROMPT_MAX = 4000
LONG_PROMPT_ARG = "Do what the message above asks."


def split_gemini_prompt(system_prompt: str, prompt: str) -> tuple[str, str]:
    """(``-p`` text, stdin text). ``gemini -p X`` with piped stdin S answers "S\\n\\nX", so the
    system prompt rides stdin and the message stays the question; a long message goes too."""
    system = (system_prompt or "").strip()
    if len(prompt) <= ARG_PROMPT_MAX:
        return prompt, system  # "" still closes stdin: the CLI never waits on an open pipe
    return LONG_PROMPT_ARG, (system + "\n\n" + prompt) if system else prompt


def build_gemini_argv(
    *,
    binary: str,
    prompt: str,
    model: str,
    extra_args: str,
    auto_approve: bool = True,
    include_dirs: list[str] | None = None,
    session_id: str = "",
) -> list[str]:
    argv = [binary, "-p", prompt]
    sid = (session_id or "").strip()
    if sid:
        argv.extend(["--resume", sid])
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


def usage_from_stats(stats: dict[str, Any]) -> dict[str, int]:
    """The CLI's result stats in core's usage shape: ``input_tokens`` is the whole prompt and
    ``cache_read_tokens`` the part Google served from its cache (the Gemini API provider's
    shape). The CLI's ``input`` is only the uncached part: reading that alone made the
    Context panel show 2K for a 47K-token request."""
    def num(key: str) -> int:
        try:
            return max(0, int(stats.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    cached = num("cached")
    return {
        "input_tokens": num("input_tokens") or (num("input") + cached),
        "output_tokens": num("output_tokens") or num("output"),
        "cache_read_tokens": cached,
    }


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
                self.usage = usage_from_stats(stats)
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
        # launch() puts the host's config in the chat's own CLI home (gemini_home).
        mcp_inject=True,
        needs_api_key=False,
        needs_cli=True,
        resume=True,
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
        del image_paths  # ponytail: headless -p takes no picture paths yet
        model_id = (model or "").strip()
        if not model_id or model_id.lower() == "default":
            return CodingAgentLaunchResult(
                ok=False,
                error="No Gemini CLI model selected. Pick a concrete model for this chat.",
                status="error",
            )
        from .cli_update import resolve_bin, should_heal_launch, update_cli
        from .gemini_home import TOOLS_NOTE, prepare_home

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

        # Prefer the Google gateway API key (the picked key account) unless a login account runs.
        key_account, key = pick_gemini_key(env)
        own_home = env.get(HOME_ENV) or ""
        login_dot = _dot_gemini(own_home or None)  # the account's own folder
        env = run_env(env, key, auto_approve=self._auto_approve())
        # The chat's own CLI home: Ducky's tools, a short retry budget, and its sessions.
        home = prepare_home(conv_id, mcp_config_path, None if key else login_dot)
        if home:
            env[HOME_ENV] = home
            system_prompt = f"{system_prompt}\n\n{TOOLS_NOTE}" if (system_prompt or "").strip() else TOOLS_NOTE
        tails: list[str] = []

        def run(sid: str, exe: str) -> CodingAgentLaunchResult:
            # Windows caps a command line at ~32K characters ([Errno 206]): the system prompt
            # (and a long message) go in on stdin, which the CLI puts ahead of the -p text.
            # A resumed session already holds the system prompt: it is not sent again.
            arg_prompt, stdin_prompt = split_gemini_prompt("" if sid else system_prompt, prompt)
            argv = build_gemini_argv(
                binary=exe,
                prompt=arg_prompt,
                model=model_id,
                extra_args=extra_args,
                auto_approve=self._auto_approve(),
                include_dirs=gemini_include_dirs(cwd),
                session_id=sid,
            )
            state = _GeminiStream(conv_id, run_id, push)
            proc = run_streaming_process(
                argv=argv,
                cwd=cwd,
                env_extra=env,
                conv_id=conv_id,
                on_line=state.on_line,
                timeout_s=timeout_s,
                cancel=cancel,
                stdin_data=stdin_prompt,
            )
            streamed_all = "".join(state.text_parts).strip()
            tails[:] = [proc.stderr_tail or "", proc.raw_tail or "", state.error_text or ""]
            return finalize_cli_turn(
                proc=proc,
                reply=streamed_all,
                streamed=bool(streamed_all) or bool(state.blocks),
                blocks=state.blocks,
                session_id=sid,
                new_session=state.session_id or sid,
                usage=state.usage,
                agent_label="Gemini CLI",
                timeout_s=timeout_s,
                error_text=state.error_text,
                stale_session_markers=STALE_SESSION_MARKERS,
            )

        def limited(res: Any) -> bool:
            return any(is_limit_error(t) for t in (res.error or "", *tails) if t)

        on_api_model = bool(key) and model_id.lower() not in {r["id"] for r in CLI_ALIAS_ROWS}
        push({"type": "status", "text": API_MODEL_ON_CLI if on_api_model else "Starting Gemini CLI…",
              "conv_id": conv_id, "run_id": run_id})
        sid = (session_id or "").strip()
        result = run(sid, binary)
        if (not key and home and not result.ok and not limited(result) and not result.reply_text
                and getattr(result, "status", "") in ("error", "needs_login")
                and not personal_login_refused(result.error or "", *tails)):
            # A Google sign-in that doesn't work from the chat's copied home (untested with a
            # company account): once more in the account's own home, as before 1.0.47. That
            # run has no Ducky tools or session, but the chat answers.
            if own_home:
                env[HOME_ENV] = own_home
            else:
                env.pop(HOME_ENV, None)
            home, sid = "", ""
            result = run(sid, binary)
            result.upstream_session_id = ""  # that session lives in another home: never resumed
        if (sid and not result.ok and getattr(result, "status", "") == "error" and not limited(result)
                and not result.reply_text and not getattr(result, "blocks", None)):
            # The CLI no longer has that session (or can't open it): once more as a new one,
            # system prompt included, so the chat never wedges on a dead session id.
            sid = ""
            result = run(sid, binary)
        if result.ok:
            return result
        note_key_limit(key_account, result.error, *tails)
        if personal_login_refused(result.error or "", *tails):
            result.error = PERSONAL_LOGIN_REFUSED
            if not key:  # the login account was refused: its row says so from now on
                from .gemini_auth import NO_ACCESS, _refresh_agents, write_access

                write_access(login_dot, False, NO_ACCESS)
                _refresh_agents()
            return result  # not a stale install: updating the CLI changes nothing
        if limited(result):
            # Said plainly: the CLI's own text is "status: 429 | } | ...[object Object]".
            result.error = KEY_RATE_LIMITED if key else LOGIN_RATE_LIMITED
            return result
        if not should_heal_launch(result.error or "", result.reply_text or "", *tails):
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
        retried = run(sid, resolve_bin(cli_path) or str(upd.get("cli_path") or binary))
        from .cli_update import BROKEN_INSTALL, error_line, is_broken_install_error

        after = (retried.error or "", *tails)
        if not retried.ok and is_broken_install_error(*after):  # once only: never a reinstall loop
            retried.error = f"{BROKEN_INSTALL}: {error_line(chr(10).join(after))}"
        return retried
