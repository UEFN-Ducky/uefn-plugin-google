from __future__ import annotations

from backend.gemini_cli_adapter import GeminiCliAdapter, build_gemini_argv


def _argv(**overrides: object) -> list[str]:
    kwargs: dict[str, object] = {
        "binary": "gemini",
        "prompt": "salut",
        "model": "gemini-flash-latest",
        "extra_args": "",
    }
    kwargs.update(overrides)
    return build_gemini_argv(**kwargs)  # type: ignore[arg-type]


def test_picker_is_empty_without_a_live_catalog(monkeypatch) -> None:
    monkeypatch.setattr("backend.gemini_cli_adapter._gemini_cli_model_rows", lambda: [])
    assert GeminiCliAdapter().detect(settings=None).models == []


def test_auto_approve_is_on_by_default() -> None:
    assert "-y" in _argv()


def test_auto_approve_can_be_turned_off() -> None:
    """-y approves file writes and shell commands: it must not be forced on."""
    assert "-y" not in _argv(auto_approve=False)


def test_include_directories_lists_added_projects() -> None:
    argv = _argv(include_dirs=[r"C:\Fortnite\Roguelike"])
    assert argv[argv.index("--include-directories") + 1] == r"C:\Fortnite\Roguelike"


def test_model_and_extra_args_still_reach_the_cli() -> None:
    argv = _argv(model="gemini-pro-latest", extra_args="--debug")

    assert argv[:2] == ["gemini", "-p"]
    assert "salut" in argv
    assert argv[argv.index("-m") + 1] == "gemini-pro-latest"
    assert "--debug" in argv


def test_only_duckys_mcp_server_may_load() -> None:
    # Gemini merges a project's .gemini/settings.json servers in; Epic's
    # ModelContextProtocol.GenerateClientConfig writes one with unreal-mcp.
    argv = _argv()
    at = argv.index("--allowed-mcp-server-names")
    assert argv[at + 1] == "uefn"
    assert argv[at + 2].startswith("--")


def test_login_account_runs_in_its_own_gemini_home(tmp_path) -> None:
    from backend.gemini_cli_adapter import account_env, account_logout, account_status

    home = tmp_path / "acc2"
    assert account_env(str(home)) == {"GEMINI_CLI_HOME": str(home)}
    assert account_status(str(home)) == {"loggedIn": False}
    (home / ".gemini" / "oauth_creds.json").write_text("{}", encoding="utf-8")
    (home / ".gemini" / "google_accounts.json").write_text('{"active": "a@b.c"}', encoding="utf-8")
    assert account_status(str(home)) == {"loggedIn": True, "email": "a@b.c"}
    account_logout(str(home))
    assert account_status(str(home))["loggedIn"] is False


def test_quota_text_is_a_limit_error() -> None:
    from backend.gemini_cli_adapter import is_limit_error

    assert is_limit_error("[API Error: 429 Too Many Requests]")
    assert is_limit_error("Quota exceeded for quota metric 'Generate Content API requests'")
    assert is_limit_error("RESOURCE_EXHAUSTED")
    assert not is_limit_error("Context window full")


def test_second_key_is_used_after_the_first_hits_429(monkeypatch) -> None:
    """Gemini key 'Free tier 2' runs the next turn once key 1 is at its quota."""
    import types

    from backend import gemini_cli_adapter as gca

    exhausted: set[str] = set()
    fake = types.SimpleNamespace(
        pick_account=lambda p: "k2" if "default" in exhausted else "default",
        account_key=lambda p, a: {"default": "KEY1", "k2": "KEY2"}[a],
        mark_exhausted=lambda p, a, until=None: exhausted.add(a),
    )
    import backend.agent as agent_pkg

    monkeypatch.setattr(agent_pkg, "accounts", fake, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "backend.agent.accounts", fake)

    assert gca.pick_gemini_key({}) == ("default", "KEY1")
    gca.note_key_limit("default", "429 Too Many Requests")
    assert gca.pick_gemini_key({}) == ("k2", "KEY2")
    # A login account (own home) or a key core already set is left alone.
    assert gca.pick_gemini_key({"GEMINI_CLI_HOME": "x"}) == ("", "")
