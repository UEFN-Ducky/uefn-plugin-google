"""A chat's own Gemini CLI home: where a run gets Ducky's tools and a short retry budget.

The CLI reads MCP servers only from a settings.json (it has no flag for one, and refuses a
system settings file kept in a user folder), so each chat runs with ``GEMINI_CLI_HOME`` on a
folder Ducky owns. The person's ``~/.gemini`` is never written. The CLI keeps the chat's
sessions there too, which is what ``--resume`` continues.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

#: The CLI's default is 10 tries with 5-30 s waits, per request: minutes of silence on a 429.
MAX_ATTEMPTS = 2
CREDS = ("oauth_creds.json", "google_accounts.json")
# Gemini CLI declares every MCP tool it is given in every request, and Ducky has about a
# thousand: 112,000 input tokens per request, measured Nov 11 2026 (a free key's whole
# minute in two requests). It gets the tools every agent has up front, plus the two that
# reach all the others.
_REACH_ALL = ("ducky_get_tools", "ducky_call_tool", "skill_search", "skill_read_subskill", "ducky_create_plan")
_CORE_FALLBACK = (
    "workspace_read_file", "workspace_read_files", "workspace_edit_file", "workspace_multi_edit",
    "workspace_replace_lines", "workspace_write_file", "workspace_find", "workspace_search",
    "workspace_git", "ducky_terminal_open", "ducky_terminal_run", "ducky_ask_user",
    "ducky_plan_update_node", "ducky_rename_self", "view_image",
)
TOOLS_NOTE = (
    "Ducky tools here: only a few are listed. Every other Ducky tool is still yours: find it "
    "with ducky_get_tools(pattern=\"what you need\") and run it with "
    "ducky_call_tool(name, arguments). There is no ToolSearch in this agent."
)


def listed_tools() -> list[str]:
    try:
        from backend.agent.coding_agents.core_tools import CORE_TOOLS as core
    except Exception:  # noqa: BLE001 - an older core
        core = _CORE_FALLBACK
    return [*core, *(t for t in _REACH_ALL if t not in core)]


def home_root() -> Path:
    from frontend.settings import default_app_data_dir

    return default_app_data_dir() / "coding_agents" / "gemini_cli_home"


def _uefn_server(mcp_config_path: str) -> dict[str, Any] | None:
    try:
        servers = json.loads(Path(mcp_config_path).read_text(encoding="utf-8")).get("mcpServers") or {}
    except (OSError, ValueError, AttributeError):
        return None
    server = servers.get("uefn") if isinstance(servers, dict) else None
    return server if isinstance(server, dict) else None


def prepare_home(conv_id: str, mcp_config_path: str, login_dot: Path | None = None, *,
                 root: Path | None = None) -> str:
    """Write the chat's home and return it ('' when it can't be made: the run then uses the
    CLI's own home, as before). ``login_dot``: the signed-in account's ``.gemini`` folder,
    whose sign-in files are copied in (None for a run on an API key)."""
    try:
        name = re.sub(r"[^A-Za-z0-9_-]", "_", conv_id or "") or "session"
        home = (root or home_root()) / name
        dot = home / ".gemini"
        dot.mkdir(parents=True, exist_ok=True)
        settings: dict[str, Any] = {"general": {"maxAttempts": MAX_ATTEMPTS}}
        server = _uefn_server(mcp_config_path)
        if server:
            settings["mcpServers"] = {"uefn": {**server, "includeTools": listed_tools()}}
        (dot / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        for cred in CREDS:
            src = login_dot / cred if login_dot else None
            if src is not None and src.is_file():
                shutil.copy2(src, dot / cred)
            elif login_dot is None:
                (dot / cred).unlink(missing_ok=True)  # a key run never rides an old sign-in
        return str(home)
    except Exception:  # noqa: BLE001 - the chat still runs, in the CLI's own home
        return ""
