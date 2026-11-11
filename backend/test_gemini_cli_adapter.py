from __future__ import annotations

from types import SimpleNamespace

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


def test_google_login_without_a_key_still_lists_the_cli_aliases(monkeypatch) -> None:
    import backend.agent.secrets as secrets

    monkeypatch.setattr(secrets, "get_key", lambda _p: "")
    from backend import gemini_cli_adapter as gca

    assert [r["id"] for r in gca._gemini_cli_model_rows()] == ["auto", "pro", "flash", "flash-lite"]


def test_a_saved_key_does_not_repeat_the_api_models_as_cli_rows(monkeypatch) -> None:
    """People on an API key picked one of the repeated rows and ran Gemini CLI unknowingly."""
    import backend.agent.secrets as secrets

    monkeypatch.setattr(secrets, "get_key", lambda _p: "AIza-key")
    from backend import gemini_cli_adapter as gca

    assert [r["id"] for r in gca._gemini_cli_model_rows()] == ["auto", "pro", "flash", "flash-lite"]


def _launch(monkeypatch, tmp_path, results, *, session_id="", key=("", ""), home="", stderr=""):
    """Run launch() against fake CLI runs; returns (result, [run kwargs], [finalize kwargs])."""
    from backend import gemini_cli_adapter as gca

    runs: list[dict] = []
    finals: list[dict] = []
    monkeypatch.setattr(gca, "run_streaming_process", lambda **kw: runs.append(kw) or SimpleNamespace(
        returncode=0, stderr_tail=stderr, raw_tail="", stdout_lines=0, timed_out=False, cancelled=False))
    monkeypatch.setattr("backend.cli_update.resolve_bin", lambda _p="": "gemini.cmd")
    monkeypatch.setattr("backend.gemini_home.prepare_home", lambda *a, **k: home)
    monkeypatch.setattr(gca, "pick_gemini_key", lambda _env: key)
    monkeypatch.setattr(gca, "note_key_limit", lambda *a: None)
    queue = list(results)
    monkeypatch.setattr(gca, "finalize_cli_turn", lambda **kw: finals.append(kw) or queue.pop(0))
    result = gca.GeminiCliAdapter().launch(
        prompt="hey", system_prompt="RULES", cwd=str(tmp_path), conv_id="c1", model="auto",
        mcp_config_path="", extra_args="", cli_path="", env={}, push=lambda _e: None, session_id=session_id,
    )
    return result, runs, finals


def _res(ok, *, error="", status="", reply=""):
    return SimpleNamespace(ok=ok, error=error, status=status or ("done" if ok else "error"),
                           reply_text=reply, blocks=[], upstream_session_id="s2")


def test_a_followup_resumes_the_session_and_does_not_resend_the_system_prompt(monkeypatch, tmp_path) -> None:
    """Every turn used to be a new CLI run carrying the whole system prompt and history."""
    _result, runs, finals = _launch(monkeypatch, tmp_path, [_res(True)], session_id="sess-1", home=str(tmp_path / "h"))
    argv = runs[0]["argv"]
    assert argv[argv.index("--resume") + 1] == "sess-1"
    assert runs[0]["stdin_data"] == ""  # the session already holds it
    assert runs[0]["env_extra"]["GEMINI_CLI_HOME"] == str(tmp_path / "h")
    _result, runs, _finals = _launch(monkeypatch, tmp_path, [_res(True)], home=str(tmp_path / "h"))
    assert runs[0]["stdin_data"].startswith("RULES") and "ducky_call_tool" in runs[0]["stdin_data"]
    assert finals[0]["session_id"] == "sess-1" and finals[0]["new_session"] == "sess-1"


def test_a_first_turn_sends_the_system_prompt_once(monkeypatch, tmp_path) -> None:
    _result, runs, _finals = _launch(monkeypatch, tmp_path, [_res(True)])
    assert "--resume" not in runs[0]["argv"] and runs[0]["stdin_data"] == "RULES"
    assert "GEMINI_CLI_HOME" not in runs[0]["env_extra"]  # no home could be made: the CLI's own


def test_a_session_the_cli_lost_runs_once_more_as_a_new_one(monkeypatch, tmp_path) -> None:
    result, runs, _finals = _launch(
        monkeypatch, tmp_path, [_res(False, error="Gemini CLI exited with code 1"), _res(True, reply="hi")],
        session_id="gone")
    assert result.ok and len(runs) == 2
    assert "--resume" in runs[0]["argv"] and "--resume" not in runs[1]["argv"]
    assert runs[1]["stdin_data"] == "RULES"


def test_a_429_is_said_plainly_and_never_run_again(monkeypatch, tmp_path) -> None:
    """'Gemini CLI exited with code 1: status: 429 | } | An unexpected critical error occurred:[object Object]'"""
    from backend import gemini_cli_adapter as gca

    raw = "Gemini CLI exited with code 1: status: 429 | } | An unexpected critical error occurred:[object Object]"
    result, runs, _finals = _launch(monkeypatch, tmp_path, [_res(False, error=raw)], session_id="s1", key=("default", "K"))
    assert len(runs) == 1  # a rate limit is not a lost session: no second run on the same quota
    assert result.error == gca.KEY_RATE_LIMITED and "object Object" not in result.error
    assert gca.is_limit_error(result.error)  # core still moves to the next key

    result, _runs, _finals = _launch(monkeypatch, tmp_path, [_res(False, error=raw)])
    assert result.error == gca.LOGIN_RATE_LIMITED


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


def test_default_account_hooks_take_none(tmp_path, monkeypatch) -> None:
    """Core passes None for ``default``: Log in raised TypeError (Path(None))."""
    from backend import cli_update
    from backend.gemini_cli_adapter import account_env, account_login, account_logout, account_status

    monkeypatch.setattr(cli_update, "resolve_bin", lambda _p: "gemini.cmd")
    monkeypatch.setenv("GEMINI_CLI_HOME", str(tmp_path))
    assert account_env(None) == {}
    assert account_login(None) == {"argv": ["gemini.cmd"], "env": {}}  # the CLI's own home
    assert account_status(None) == {"loggedIn": False}
    (tmp_path / ".gemini").mkdir()
    (tmp_path / ".gemini" / "oauth_creds.json").write_text("{}", encoding="utf-8")
    assert account_status(None)["loggedIn"] is True
    account_logout(None)
    assert account_status(None)["loggedIn"] is False
    fresh = tmp_path / "not-made-yet"
    assert account_login(str(fresh))["env"] == {"GEMINI_CLI_HOME": str(fresh)}
    assert (fresh / ".gemini").is_dir()


def test_system_prompt_rides_stdin_and_the_message_stays_the_question() -> None:
    from backend.gemini_cli_adapter import ARG_PROMPT_MAX, LONG_PROMPT_ARG, split_gemini_prompt

    assert split_gemini_prompt("  rules  ", "hey") == ("hey", "rules")
    assert split_gemini_prompt("", "hey") == ("hey", "")  # stdin still closes
    long_msg = "x" * (ARG_PROMPT_MAX + 1)
    assert split_gemini_prompt("rules", long_msg) == (LONG_PROMPT_ARG, "rules\n\n" + long_msg)


def test_run_env_signs_in_with_google_and_trusts_the_folder() -> None:
    """Exit 41 'Please set an Auth method' and 'Approval mode overridden ... not trusted'."""
    from backend.gemini_cli_adapter import run_env

    login = run_env({"GEMINI_CLI_HOME": "C:/acc"}, "", auto_approve=True)
    assert login["GOOGLE_GENAI_USE_GCA"] == "true" and login["GEMINI_CLI_TRUST_WORKSPACE"] == "true"
    assert "GEMINI_API_KEY" not in login and login["GEMINI_CLI_HOME"] == "C:/acc"
    keyed = run_env({}, "AIza-k", auto_approve=False)
    assert keyed["GEMINI_API_KEY"] == "AIza-k" and "GOOGLE_GENAI_USE_GCA" not in keyed
    assert "GEMINI_CLI_TRUST_WORKSPACE" not in keyed  # asks before changes: the CLI's own trust


def test_a_refused_personal_login_says_what_to_do(tmp_path, monkeypatch) -> None:
    """Google, since June 2026: IneligibleTierError for personal accounts in Gemini CLI."""
    from backend import gemini_cli_adapter as gca

    refused = ("Error authenticating: IneligibleTierError: This client is no longer supported for "
               "Gemini Code Assist for individuals. To continue using Gemini, please migrate to Antigravity")
    monkeypatch.setattr(gca, "run_streaming_process", lambda **kw: SimpleNamespace(
        returncode=1, stderr_tail=refused, raw_tail="", stdout_lines=0, timed_out=False, cancelled=False))
    monkeypatch.setattr("backend.cli_update.resolve_bin", lambda _p="": "gemini.cmd")
    monkeypatch.setattr(gca, "pick_gemini_key", lambda _env: ("", ""))
    monkeypatch.setattr(gca, "finalize_cli_turn", lambda **kw: SimpleNamespace(ok=False, error="Gemini CLI exited with code 1", reply_text=""))
    updates = []
    monkeypatch.setattr("backend.cli_update.update_cli", lambda *a: updates.append(a) or {"ok": True})
    result = gca.GeminiCliAdapter().launch(
        prompt="hey", system_prompt="", cwd=str(tmp_path), conv_id="c1", model="auto",
        mcp_config_path="", extra_args="", cli_path="", env={}, push=lambda _e: None,
    )
    assert result.error == gca.PERSONAL_LOGIN_REFUSED and gca.KEY_URL in result.error
    assert updates == []  # never "updating automatically" for a refusal


def test_a_huge_system_prompt_never_reaches_the_command_line(tmp_path, monkeypatch) -> None:
    """[Errno 206] Prompt/command too long for Windows: the 100K system prompt went in argv."""
    from backend import gemini_cli_adapter as gca

    seen: dict[str, object] = {}

    def fake_run(**kw):
        seen.update(kw)
        return SimpleNamespace(returncode=0, stderr_tail="", raw_tail="", stdout_lines=0, timed_out=False, cancelled=False)

    monkeypatch.setattr(gca, "run_streaming_process", fake_run)
    monkeypatch.setattr("backend.cli_update.resolve_bin", lambda _p="": "gemini.cmd")
    monkeypatch.setattr(gca, "pick_gemini_key", lambda _env: ("", ""))
    monkeypatch.setattr(gca, "finalize_cli_turn", lambda **kw: SimpleNamespace(ok=True, error=""))
    system = "S" * 100_000
    gca.GeminiCliAdapter().launch(
        prompt="hey", system_prompt=system, cwd=str(tmp_path), conv_id="c1", model="auto",
        mcp_config_path="", extra_args="", cli_path="", env={}, push=lambda _e: None,
    )
    argv = seen["argv"]
    assert sum(len(a) + 3 for a in argv) < 8000  # far under Windows' 32,767
    assert argv[argv.index("-p") + 1] == "hey"
    assert seen["stdin_data"] == system


def test_installed_cli_without_a_login_or_key_is_not_connected(tmp_path, monkeypatch) -> None:
    """A fresh PC with gemini on PATH showed 'Connected' in setup: detect left logged_in unset."""
    from backend import gemini_cli_adapter as gca

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_CLI_HOME", str(tmp_path))
    monkeypatch.setattr(gca, "which_cli", lambda *_a: "gemini.cmd")
    monkeypatch.setattr(gca, "pick_gemini_key", lambda _env: ("", ""))
    monkeypatch.setattr(gca, "_gemini_cli_model_rows", lambda: [])
    monkeypatch.setattr("backend.cli_update.read_cli_version", lambda _p: "", raising=False)
    monkeypatch.setattr("backend.cli_update.status_text", lambda: "", raising=False)

    assert gca.GeminiCliAdapter().detect(settings=None).logged_in is False

    (tmp_path / ".gemini").mkdir()
    (tmp_path / ".gemini" / "oauth_creds.json").write_text("{}", encoding="utf-8")
    assert gca.GeminiCliAdapter().detect(settings=None).logged_in is True
