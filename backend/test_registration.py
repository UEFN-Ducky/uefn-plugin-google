from __future__ import annotations

from typing import Any

from backend import register
from backend.gemini_cli_adapter import GeminiCliAdapter, build_gemini_argv
from backend.test_gemini_cli_adapter import RETIRED_PREFIXES


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


def test_mcp_injection_is_only_advertised_when_the_config_reaches_the_cli() -> None:
    """launch() drops mcp_config_path, so claiming mcp_inject hides missing tools."""
    argv = build_gemini_argv(binary="gemini", prompt="x", model="m", extra_args="")
    config_reaches_cli = any("mcp" in arg.lower() for arg in argv)

    assert GeminiCliAdapter.capabilities.mcp_inject is config_reaches_cli
