"""Antigravity coding agent: headless ``agy -p --output-format stream-json`` on the person's
Google sign-in (their Google AI plan), with Ducky's ``uefn`` tools in agy's MCP config.

agy reads one MCP config for the whole PC (``~/.gemini/config/mcp_config.json``), so the
``uefn`` server written there carries no chat identity: each chat's ``DUCKY_*`` identity
rides the agy process environment, which the bridge it starts inherits. Two chats at
once never rewrite each other's file.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Any

from backend.agent.coding_agents.base import CodingAgentCapabilities, CodingAgentInfo, CodingAgentLaunchResult
from backend.agent.coding_agents.cli_shared import finalize_cli_turn, truncate_tool_result
from backend.agent.coding_agents.proc_exec import run_streaming_process
from backend.agent.coding_agents.settings_helpers import coding_agent_cfg

from . import agy_cli

AGENT_ID = "antigravity"
LABEL = "Antigravity"
AUTO_ROW = {"id": "auto", "name": "Auto (Antigravity picks)", "provider": LABEL}
NEEDS_LOGIN = "Sign in to Antigravity with Google: Settings → LLMs → Google → Antigravity → Log in."
LIMITED = ("Your Google AI plan hit its Antigravity limit (429 / quota). Wait and send again, or pick "
           "another model.")
STALE_SESSION_MARKERS = ("conversation not found", "unknown conversation", "invalid conversation",
                         "no conversation", "could not find conversation")
_LIMIT_RE = re.compile(r"quota exceeded|resource[_ ]exhausted|\b429\b|too many requests|rate limit"
                       r"|exhausted your (?:daily |weekly )?(?:quota|capacity|limit)", re.I)
ARG_PROMPT_MAX = 6000  # the rest of a message goes in a file: Windows caps a command line at ~32K
_RUN_ID_FLAG = "--ducky-run-id"
# A long agy conversation can go silent mid-turn: no output, no exit, no error (Nov 23 2026).
# ponytail: fixed limits; a tool running counts as alive for longer (a Verse build takes minutes).
IDLE_S = 240.0
TOOL_IDLE_S = 900.0
STALLED = ("Antigravity stopped responding twice (no output for minutes). Send your message "
           "again, or pick another model.")


def recover_note(conv_id: str) -> str:
    return ("(Ducky: this chat's earlier Antigravity conversation stopped responding and was "
            f"replaced by this new one. First catch up with ducky_read_chat(conv_id=\"{conv_id}\"), "
            "check what was already changed, then do the message below.)\n\n")


def watch_stall(state: "AgyStream", stop: threading.Event, user_cancel: threading.Event | None, *,
                idle_s: float = IDLE_S, tool_idle_s: float = TOOL_IDLE_S, poll: float = 2.0) -> threading.Event:
    """Sets ``stop`` (which kills agy) on the person's Stop, or when agy prints nothing for
    too long; the returned event says it was a stall."""
    stalled = threading.Event()

    def loop() -> None:
        while not stop.wait(poll):
            if user_cancel is not None and user_cancel.is_set():
                stop.set()
                return
            limit = tool_idle_s if state._open else idle_s
            if time.monotonic() - state.last_seen > limit:
                stalled.set()
                stop.set()
                return

    threading.Thread(target=loop, daemon=True, name=f"agy-watch-{state.conv_id[:8]}").start()
    return stalled


def is_limit_error(text: str) -> bool:
    return isinstance(text, str) and bool(_LIMIT_RE.search(text))


def mcp_config_file() -> Path:
    return Path.home() / ".gemini" / "config" / "mcp_config.json"


def _strip_run_id(args: list[Any]) -> list[Any]:
    out: list[Any] = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a == _RUN_ID_FLAG:
            skip = True
            continue
        if isinstance(a, str) and a.startswith(_RUN_ID_FLAG + "="):
            continue
        out.append(a)
    return out


def shared_server(server: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """(the block for agy's PC-wide config, the env this chat's agy process carries)."""
    env = {str(k): str(v) for k, v in (server.get("env") or {}).items()}
    proc_env = dict(env)
    block: dict[str, Any] = {k: v for k, v in server.items() if k not in ("env", "args")}
    block["args"] = _strip_run_id(list(server.get("args") or []))
    keep = {k: v for k, v in env.items() if not k.startswith("DUCKY_")}
    bridge = env.get("DUCKY_BRIDGE_ARGV")
    if bridge:
        try:
            keep["DUCKY_BRIDGE_ARGV"] = json.dumps(_strip_run_id(json.loads(bridge)))
        except (ValueError, TypeError):
            pass
    # ponytail: assumes agy hands its own env to the servers it starts (every MCP client
    # here does); identity in the file instead would race between chats.
    if keep:
        block["env"] = keep
    block["disabled"] = False
    return block, proc_env


def merge_mcp(host_config_path: str, target: Path | None = None) -> dict[str, str]:
    """Put Ducky's ``uefn`` server in agy's config (the person's other servers stay); returns
    the env this chat's agy process needs. The file is only rewritten when it changes."""
    try:
        servers = json.loads(Path(host_config_path).read_text(encoding="utf-8")).get("mcpServers") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    server = servers.get("uefn") or next(iter(servers.values()), None)
    if not isinstance(server, dict):
        return {}
    block, proc_env = shared_server(server)
    target = target or mcp_config_file()
    try:
        current = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            current = {}
    except (OSError, ValueError):
        current = {}
    mine = current.setdefault("mcpServers", {})
    if not isinstance(mine, dict):
        current["mcpServers"] = mine = {}
    if mine.get("uefn") != block:
        mine["uefn"] = block
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".ducky.tmp")
        part.write_text(json.dumps(current, indent=2), encoding="utf-8")
        os.replace(part, target)
    return proc_env


def prompt_dir(conv_id: str) -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "UEFN-Ducky" / "antigravity"
    folder = base / re.sub(r"[^\w.-]", "_", conv_id or "chat")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def build_prompt(folder: Path, system_prompt: str, prompt: str, *, resumed: bool) -> str:
    """The ``-p`` text. The system prompt (new conversations only) and a long message go in
    files in ``folder``, which agy is given with ``--add-dir``."""
    lines: list[str] = []
    system = (system_prompt or "").strip()
    if system and not resumed:
        path = folder / "instructions.md"
        path.write_text(system, encoding="utf-8")
        lines.append(f"First read {path} and follow it for this whole conversation.")
    if len(prompt) > ARG_PROMPT_MAX or "\x00" in prompt:
        path = folder / "message.md"
        path.write_text(prompt, encoding="utf-8")
        lines.append(f"The person's message is in {path}. Do what it asks.")
    else:
        lines.append(prompt)
    return "\n\n".join(lines)


def build_argv(*, binary: str, prompt: str, model: str, extra_args: str, auto_approve: bool,
               add_dir: str = "", session_id: str = "") -> list[str]:
    argv = [binary, "-p", prompt, "--output-format", "stream-json"]
    mid = (model or "").strip()
    if mid and mid.lower() not in ("auto", "default"):
        argv += ["--model", mid]
    if (session_id or "").strip():
        argv += ["--conversation", session_id.strip()]
    if auto_approve:
        argv.append("--dangerously-skip-permissions")
    if add_dir:
        argv += ["--add-dir", add_dir]
    if (extra_args or "").strip():
        argv += shlex.split(extra_args, posix=False)
    return argv


def _num(d: dict[str, Any], *keys: str) -> int:
    for k in keys:
        try:
            v = int(d.get(k) or 0)
        except (TypeError, ValueError):
            continue
        if v:
            return max(0, v)
    return 0


class AgyStream:
    """agy's stream-json, as agy prints it:
    ``{"event":"init","conversation_id":…}`` →
    ``{"event":"step_update","step_update":{step_index, state ACTIVE|DONE, step_type
    agent_response (text_delta) | tool (tool_name, tool_info{parameters, output}) | user_input}}``…
    → ``{"event":"result","result":{conversation_id, status, response, usage}}``."""

    def __init__(self, conv_id: str, run_id: str, push: Any) -> None:
        self.conv_id, self.run_id, self.push = conv_id, run_id, push
        self.text_parts: list[str] = []
        self.blocks: list[dict[str, Any]] = []
        self.session_id = ""
        self.error_text = ""
        self.usage: dict[str, int] = {}
        self._open: dict[str, str] = {}
        self._closed: set[str] = set()
        self.last_seen = time.monotonic()

    def _emit(self, event: dict[str, Any]) -> None:
        self.push({**event, "conv_id": self.conv_id, "run_id": self.run_id})

    def _text(self, chunk: str) -> None:
        if chunk:
            self.text_parts.append(chunk)
            self._emit({"type": "assistant_delta", "text": chunk})

    def on_line(self, line: str) -> None:
        self.last_seen = time.monotonic()
        try:
            data = json.loads((line or "").strip())
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        et = str(data.get("event") or data.get("type") or "").lower()
        body = data.get(et) if isinstance(data.get(et), dict) else data
        sid = str(data.get("conversation_id") or body.get("conversation_id") or "")
        if sid and et in ("init", "result"):
            self.session_id = sid
        if et == "step_update":
            self._step(body)
        elif et == "error":
            err = data.get("error")
            msg = err.get("message") if isinstance(err, dict) else err
            self.error_text = str(msg or data.get("message") or "").strip() or self.error_text
        elif et == "result":
            usage = body.get("usage")
            if isinstance(usage, dict):
                self.usage = {"input_tokens": _num(usage, "input_tokens"),
                              "output_tokens": _num(usage, "output_tokens"),
                              "cache_read_tokens": _num(usage, "cache_read_tokens")}
            if str(body.get("status") or "SUCCESS").upper() not in ("SUCCESS", "OK", "DONE"):
                err = body.get("error")
                msg = err.get("message") if isinstance(err, dict) else err
                self.error_text = str(msg or body.get("message") or f"Antigravity stopped: {body.get('status')}")
            resp = body.get("response") if body is not data else data.get("result")
            if isinstance(resp, str) and resp.strip() and not self.text_parts:
                self._text(resp.strip())

    def _step(self, step: dict[str, Any]) -> None:
        sid = str(step.get("step_index", ""))
        kind = str(step.get("step_type") or "").lower()
        state = str(step.get("state") or "").upper()
        if kind == "user_input":
            return
        if kind == "agent_response":
            self._text(str(step.get("text_delta") or ""))  # streamed: every update carries the next piece
            return
        info = step.get("tool_info") if isinstance(step.get("tool_info"), dict) else {}
        name = str(step.get("tool_name") or info.get("name") or kind or "step")
        if sid in self._closed:  # agy may repeat a finished step's last update
            return
        if sid not in self._open:
            args = info.get("parameters") or {}
            self._open[sid] = name
            self._emit({"type": "tool_start", "tool": {"id": sid or name, "name": name,
                                                       "args": args if isinstance(args, dict) else {"raw": args}}})
        if state in ("DONE", "ERROR", "FAILED", "CANCELED", "CANCELLED"):
            name = self._open.pop(sid, name)
            self._closed.add(sid)
            ok = state == "DONE"
            result = truncate_tool_result(str(info.get("output") or info.get("error") or step.get("error") or ""))
            self.blocks.append({"type": "tool", "name": name, "id": sid, "result": result, "ok": ok})
            self._emit({"type": "tool_done", "success": ok, "tool": {"id": sid or name, "name": name, "result": result}})


def _cfg(settings: Any = None) -> dict[str, Any]:
    try:
        if settings is None:
            from frontend.settings import PanelSettings

            settings = PanelSettings.load()
        return coding_agent_cfg(settings, AGENT_ID)
    except Exception:
        return {}


class AntigravityAdapter:
    id = AGENT_ID
    label = LABEL
    capabilities = CodingAgentCapabilities(terminal_agent=True, chat_api=False, a2a=True, mcp_inject=True,
                                           needs_api_key=False, needs_cli=True, resume=True)

    def detect(self, settings: Any) -> CodingAgentInfo:
        cfg = _cfg(settings)
        enabled = bool(cfg.get("enabled", True))
        override = str(cfg.get("cli_path") or "")
        path = agy_cli.resolve_bin(override)
        logged_in: bool | None = None
        rows: list[dict[str, str]] = []
        if path:
            logged_in, rows = agy_cli.account_models(path)
            status = f"Found: {path} · " + {True: "logged in", False: "not logged in"}.get(
                logged_in, "sign-in state unknown")
        else:
            agy_cli.install_in_background()
            status = agy_cli.install_status() or "Antigravity CLI not found — Ducky is downloading it from Google."
        info = CodingAgentInfo(id=self.id, label=self.label, enabled=enabled, available=bool(path) and enabled,
                               status=status, cli_path=path or override,
                               default_args=str(cfg.get("default_args") or ""), capabilities=self.capabilities,
                               models=[dict(AUTO_ROW), *[r for r in rows if r["id"] != "auto"]])
        info.logged_in = logged_in  # type: ignore[attr-defined]
        return info

    def launch(self, *, prompt: str, system_prompt: str, cwd: str, conv_id: str, model: str,
               mcp_config_path: str, extra_args: str, cli_path: str, env: dict[str, str], push: Any,
               session_id: str = "", run_id: str = "", cancel: threading.Event | None = None,
               timeout_s: float = 0.0, image_paths: list[str] | None = None, **_kw: Any) -> CodingAgentLaunchResult:
        del image_paths  # ponytail: -p takes no pictures; add when agy grows a flag for them
        binary = agy_cli.resolve_bin(cli_path)
        if not binary:
            push({"type": "status", "text": "Downloading the Antigravity CLI from Google…", "conv_id": conv_id, "run_id": run_id})
            agy_cli.install()
            binary = agy_cli.resolve_bin(cli_path)
            if not binary:
                return CodingAgentLaunchResult(ok=False, status="error",
                                               error=agy_cli.install_status() or "Antigravity could not be installed.")
        run_env = {**env, **(merge_mcp(mcp_config_path) if mcp_config_path else {})}
        folder = prompt_dir(conv_id)
        auto = bool(_cfg().get("auto_approve", True))
        tails: list[str] = []

        def run(sid: str, note: str = "") -> CodingAgentLaunchResult:
            argv = build_argv(binary=binary, prompt=build_prompt(folder, system_prompt, note + prompt,
                                                                  resumed=bool(sid)),
                              model=model, extra_args=extra_args, auto_approve=auto, add_dir=str(folder),
                              session_id=sid)
            state = AgyStream(conv_id, run_id, push)
            stop = threading.Event()
            stalled = watch_stall(state, stop, cancel)
            proc = run_streaming_process(argv=argv, cwd=cwd, env_extra=run_env, conv_id=conv_id,
                                         on_line=state.on_line, timeout_s=timeout_s, cancel=stop, stdin_data="")
            stop.set()
            reply = "".join(state.text_parts).strip()
            tails[:] = [proc.stderr_tail or "", proc.raw_tail or "", state.error_text]
            out = finalize_cli_turn(proc=proc, reply=reply, streamed=bool(reply) or bool(state.blocks),
                                    blocks=state.blocks, session_id=sid, new_session=state.session_id or sid,
                                    usage=state.usage, agent_label=LABEL, timeout_s=timeout_s,
                                    error_text=state.error_text, stale_session_markers=STALE_SESSION_MARKERS)
            out.stalled = stalled.is_set()  # type: ignore[attr-defined]
            return out

        def user_stopped() -> bool:
            return cancel is not None and cancel.is_set()

        push({"type": "status", "text": "Starting Antigravity…", "conv_id": conv_id, "run_id": run_id})
        sid = (session_id or "").strip()
        result = run(sid)
        if getattr(result, "stalled", False) and not user_stopped():
            push({"type": "status", "text": "Antigravity went quiet. Starting it again in a fresh conversation…",
                  "conv_id": conv_id, "run_id": run_id})
            result = run("", recover_note(conv_id))  # once: a fresh conversation, never a wedged chat
            if getattr(result, "stalled", False) and not user_stopped():
                result.ok, result.status, result.error = False, "error", STALLED
            return result
        if result.ok or user_stopped():
            return result
        texts = (result.error or "", *tails)
        if any(agy_cli.signed_out(t) for t in texts):
            agy_cli.forget_models()
            result.status, result.error = "needs_login", NEEDS_LOGIN
            return result
        if any(is_limit_error(t) for t in texts):
            result.error = LIMITED
            return result
        if sid and not result.reply_text and not getattr(result, "blocks", None):
            result = run("", recover_note(conv_id))  # agy lost that conversation: once more as a new one
        return result


# ── Settings → LLMs → Google → Antigravity: Log in (code box), like Claude Code ──

def login(cwd: str = "", cli_path: str = "", push: Any = None, **_kw: Any) -> dict[str, Any]:
    return agy_cli.start_login(cli_path or str(_cfg().get("cli_path") or ""))


def login_submit(code: str = "", cwd: str = "", cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    out = agy_cli.submit_code(code)
    if out.get("logged_in"):
        _refresh()
    return out


def login_status(cwd: str = "", cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    return agy_cli.login_status()


def login_cancel(**_kw: Any) -> dict[str, Any]:
    return agy_cli.cancel_login()


def logout(cwd: str = "", cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    out = agy_cli.logout(cli_path or str(_cfg().get("cli_path") or ""))
    _refresh()
    return out


def _refresh() -> None:
    agy_cli.refresh_agents()


# ── "Antigravity logins" table (core's account hooks) ──
# The table's Log in (and Log in another account): Google opens in the person's own browser and
# the plugin's code box pops up (ui/agy_login.js asks popup_state); the table spins until signed in.
# ponytail: agy keeps one Google sign-in per PC, so every row is that sign-in; per-account homes
# via `--gemini-dir` if people want two Google accounts.

def account_env(config_dir: str | None) -> dict[str, str]:
    return {}


def account_browser_login(config_dir: str | None) -> dict[str, Any]:
    out = agy_cli.start_login(str(_cfg().get("cli_path") or ""))
    if not out.get("ok"):
        return {"error": out.get("error") or "Could not start the Antigravity sign-in."}
    agy_cli._login["popup"] = True
    return {"message": "Sign in with Google in your browser, then paste the code in the box."}


def account_status(config_dir: str | None) -> dict[str, Any]:
    binary = agy_cli.resolve_bin(str(_cfg().get("cli_path") or ""))
    if not binary:
        return {"loggedIn": False, "error": agy_cli.install_status() or "Antigravity isn't installed yet."}
    if agy_cli.popup_state()["open"]:
        return {"loggedIn": False, "pending": True}
    # Asked fresh (about 3 s): a sign-in made anywhere else (a terminal, another app) shows on
    # the table's Refresh, and a change re-reads the model picker's list, cached 10 minutes.
    before = agy_cli._models.get("signed_in") if agy_cli._models.get("bin") == binary else None
    state, _rows = agy_cli.account_models(binary, force=True)
    if state != before:
        _refresh()
    return {"loggedIn": state is True}


def popup_submit(code: str = "") -> dict[str, Any]:
    out = agy_cli.submit_code(code)
    if out.get("logged_in"):
        _refresh()
    return out


def account_logout(config_dir: str | None) -> None:
    logout()


