"""Gemini CLI adapter — headless ``gemini -p`` + stream-json events.

Install: ``npm install -g @google/gemini-cli``. Auth via ``GEMINI_API_KEY``
(from the Google gateway API key) or an existing Gemini CLI login.
"""

from __future__ import annotations

import json
import shlex
import threading
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

# Rolling aliases, not pinned versions: pinned ids rot into 404 "no longer
# available to new users" as Google retires generations.
_CLI_MODELS: tuple[dict[str, str], ...] = (
    {"id": "gemini-pro-latest", "name": "Gemini Pro (latest)", "provider": "Gemini CLI"},
    {"id": "gemini-flash-latest", "name": "Gemini Flash (latest)", "provider": "Gemini CLI"},
    {
        "id": "gemini-flash-lite-latest",
        "name": "Gemini Flash Lite (latest)",
        "provider": "Gemini CLI",
    },
)


def _gemini_missing_status() -> str:
    return (
        "Gemini CLI not found — needs the `gemini` terminal command. "
        f"Install: {_GEMINI_INSTALL}. Then run gemini --version, restart Ducky, and click Detect."
    )


def build_gemini_argv(
    *,
    binary: str,
    prompt: str,
    model: str,
    extra_args: str,
    auto_approve: bool = True,
) -> list[str]:
    argv = [binary, "-p", prompt]
    if auto_approve:
        # -y approves every action, including file writes and shell commands.
        argv.append("-y")
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
        if path:
            status = f"Found: {path}"
            available = enabled
        else:
            status = _gemini_missing_status()
            available = False
        return CodingAgentInfo(
            id=self.id,
            label=self.label,
            enabled=enabled,
            available=available,
            status=status,
            cli_path=path or override,
            default_args=default_args,
            capabilities=self.capabilities,
            models=list(_CLI_MODELS),
        )

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
        binary = which_cli("gemini", cli_path) or "gemini"
        full_prompt = prompt
        if system_prompt.strip():
            full_prompt = system_prompt.strip() + "\n\n" + prompt

        # Prefer the Google gateway API key when present.
        try:
            from backend.agent.secrets import get_key

            key = (get_key("gemini") or "").strip()
            if key:
                env = {**env, "GEMINI_API_KEY": key}
        except Exception:
            pass

        argv = build_gemini_argv(
            binary=binary,
            prompt=full_prompt,
            model=model_id,
            extra_args=extra_args,
            auto_approve=self._auto_approve(),
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
        return finalize_cli_turn(
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
