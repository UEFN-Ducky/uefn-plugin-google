"""Google gateway — Gemini provider + Gemini CLI via host registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_INSTALL_HELP = (
    "Ducky installs and updates the Gemini CLI when this plugin is installed or "
    "updated — you should not run npm yourself. Uses this provider’s Google API "
    "key when saved, or an existing Gemini CLI login."
)


def _fetch_models(api_key: str, **_kw: Any) -> Any:
    from .model_fetch import fetch_models

    return fetch_models(api_key)


def _skills_dir() -> str:
    return str(Path.home() / ".claude" / "skills")


def register(api) -> None:
    from .gemini_cli_adapter import GeminiCliAdapter
    from .gemini_provider import GeminiProvider

    from .model_fetch import clear_model_cache

    api.register_llm_provider(
        "gemini",
        factory=lambda api_key, model, **kw: GeminiProvider(api_key, model, **kw),
        fetch_models=_fetch_models,
        test_key_model="gemini-flash-latest",
        tool_schema="gemini",
        clear_model_cache=clear_model_cache,
        cache_mode="implicit",
        shows_thinking_effort=True,
    )
    api.register_coding_agent(
        "gemini_cli",
        factory=lambda: GeminiCliAdapter(),
        aliases=["gemini", "google_gemini"],
        skills_dir=_skills_dir,
        settings_defaults={
            "enabled": True,
            "cli_path": "",
            "default_args": "",
            "auto_approve": True,
        },
        install_help=_INSTALL_HELP,
        token_provider="gemini",
    )
    api.register_ide_hookup("antigravity", label="Antigravity")
    try:
        from .cli_update import schedule_cli_update_on_plugin_load

        schedule_cli_update_on_plugin_load()
    except Exception:
        pass
    api.log("Google gateway contribution active (Providers + Gemini CLI + Antigravity IDE)")
