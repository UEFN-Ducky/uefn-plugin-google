from __future__ import annotations

from typing import Any

from backend import register
from backend.gemini_cli_adapter import GeminiCliAdapter, build_gemini_argv

# Model generations Google no longer serves; a key test on one rejects a good key.
RETIRED_PREFIXES = ("gemini-1.0", "gemini-1.5", "gemini-pro-vision")


class _RecordingApi:
    """Captures what the plugin contributes, standing in for the host registry."""

    def __init__(self) -> None:
        self.providers: dict[str, dict[str, Any]] = {}
        self.coding_agents: dict[str, dict[str, Any]] = {}
        self.ide_hookups: list[str] = []
        self.logs: list[str] = []

    def register_llm_provider(self, provider_id: str, **kw: Any) -> None:
        self.providers[provider_id] = kw

    def register_coding_agent(self, agent_id: str, **kw: Any) -> None:
        self.coding_agents[agent_id] = kw

    def register_ide_hookup(self, kind: str, **_kw: Any) -> None:
        self.ide_hookups.append(kind)

    def log(self, message: str) -> None:
        self.logs.append(message)


def _registered() -> _RecordingApi:
    api = _RecordingApi()
    register(api)
    return api


def test_the_key_test_uses_a_model_the_api_still_serves() -> None:
    """A retired test model makes Ducky reject a perfectly valid API key."""
    model = _registered().providers["gemini"]["test_key_model"]

    assert not model.startswith(RETIRED_PREFIXES)


def test_auto_approve_is_exposed_as_a_setting() -> None:
    defaults = _registered().coding_agents["gemini_cli"]["settings_defaults"]

    assert defaults["auto_approve"] is True


def test_account_hooks_are_registered() -> None:
    api = _registered()
    agent = api.coding_agents["gemini_cli"]
    hooks = ("account_env", "account_login", "account_browser_login", "account_status", "account_logout", "is_limit_error")
    for hook in hooks:
        assert callable(agent[hook]), hook
    assert api.providers["gemini"]["is_quota_error"]("429 Too Many Requests")


def test_mcp_injection_is_advertised_because_the_config_reaches_the_cli(tmp_path) -> None:
    """The CLI has no MCP flag: the host's config goes in the chat's own CLI home."""
    import json

    from backend.gemini_home import listed_tools, prepare_home

    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"uefn": {"command": "node", "args": ["bridge.js"]},
                                              "other": {"command": "x"}}}), encoding="utf-8")
    home = prepare_home("chat/1", str(cfg), root=tmp_path / "homes")
    settings = json.loads((tmp_path / "homes" / "chat_1" / ".gemini" / "settings.json").read_text(encoding="utf-8"))

    assert home == str(tmp_path / "homes" / "chat_1")
    assert list(settings["mcpServers"]) == ["uefn"]  # Ducky's only
    uefn = settings["mcpServers"]["uefn"]
    assert (uefn["command"], uefn["args"]) == ("node", ["bridge.js"])
    # Not all ~1,000 tools (112K input tokens a request): a few, plus the two that reach the rest.
    assert uefn["includeTools"] == listed_tools() and len(listed_tools()) <= 25
    assert {"workspace_read_file", "ducky_get_tools", "ducky_call_tool"} <= set(listed_tools())
    argv = build_gemini_argv(binary="gemini", prompt="x", model="m", extra_args="")
    assert argv[argv.index("--allowed-mcp-server-names") + 1] == "uefn"
    assert GeminiCliAdapter.capabilities.mcp_inject is True


def test_a_chat_home_gives_up_on_a_429_fast_and_carries_a_login(tmp_path) -> None:
    """The CLI's default is 10 tries with 5-30 s waits per request: minutes of silence."""
    import json

    from backend.gemini_home import MAX_ATTEMPTS, prepare_home

    login = tmp_path / "account" / ".gemini"
    login.mkdir(parents=True)
    (login / "oauth_creds.json").write_text("{}", encoding="utf-8")
    home = tmp_path / "homes" / "c1" / ".gemini"

    prepare_home("c1", "", login, root=tmp_path / "homes")  # no tool config: still a home
    settings = json.loads((home / "settings.json").read_text(encoding="utf-8"))
    assert settings == {"general": {"maxAttempts": MAX_ATTEMPTS}} and MAX_ATTEMPTS <= 2
    assert (home / "oauth_creds.json").is_file()

    prepare_home("c1", "", None, root=tmp_path / "homes")  # the same chat, now on an API key
    assert not (home / "oauth_creds.json").exists()
