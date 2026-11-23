from __future__ import annotations

import io
import json
import os
import subprocess
from typing import Any

from backend import agy_cli
from backend import antigravity_adapter as aga
from backend.test_registration import _registered


def test_registered_with_a_code_box_login_under_google() -> None:
    agent = _registered().coding_agents["antigravity"]
    for hook in ("login", "login_submit", "login_status", "login_cancel", "logout", "is_limit_error",
                 "account_env", "account_browser_login", "account_status", "account_logout"):
        assert callable(agent[hook]), hook
    assert agent["settings_defaults"]["auto_approve"] is True
    assert isinstance(agent["factory"](), aga.AntigravityAdapter)


def test_plugin_json_lists_the_agent() -> None:
    from pathlib import Path

    manifest = json.loads((Path(__file__).parent.parent / "plugin.json").read_text(encoding="utf-8"))
    assert "antigravity" in [a["id"] for a in manifest["contributes"]["llm.coding_agents"]]


def _up(i: int, state: str, kind: str, **kw: Any) -> dict[str, Any]:
    return {"event": "step_update", "step_update": {"conversation_id": "conv-9", "step_index": i,
                                                    "state": state, "step_type": kind, **kw}}


def test_stream_as_agy_prints_it() -> None:
    """Lines as `agy -p … --output-format stream-json` prints them (Nov 2026). The old parser
    read `type`/`status`/`text` keys agy never sends: every chat came back empty."""
    events: list[dict[str, Any]] = []
    s = aga.AgyStream("c1", "r1", events.append)
    tool = {"name": "call_mcp_tool", "parameters": {"ServerName": "uefn", "ToolName": "workspace_find"}}
    for line in [
        {"event": "init", "conversation_id": "conv-9", "init": {"model": "gemini-4-neon-low"}},
        _up(0, "DONE", "user_input"),
        _up(1, "DONE", "agent_response", usage={"input_tokens": 1}),
        _up(2, "ACTIVE", "tool", tool_name="call_mcp_tool", tool_info=tool),
        _up(2, "DONE", "tool", tool_name="call_mcp_tool", tool_info={**tool, "output": "a.verse"}),
        _up(2, "DONE", "tool", tool_name="call_mcp_tool", tool_info={**tool, "output": "again"}),
        _up(3, "ACTIVE", "agent_response", text_delta="Found"),
        _up(3, "DONE", "agent_response", text_delta=" it.\n"),
        {"event": "result", "result": {"conversation_id": "conv-9", "status": "SUCCESS", "response": "Found it.\n",
                                       "usage": {"input_tokens": 120, "output_tokens": 8, "cache_read_tokens": 40}}},
    ]:
        s.on_line(json.dumps(line))
    s.on_line("not json")
    kinds = [e["type"] for e in events]
    assert kinds == ["tool_start", "tool_done", "assistant_delta", "assistant_delta"]  # repeated done ignored
    assert events[0]["tool"]["name"] == "call_mcp_tool" and events[0]["tool"]["args"]["ToolName"] == "workspace_find"
    assert events[1]["tool"]["result"] == "a.verse" and events[1]["success"] is True
    assert all(e["conv_id"] == "c1" and e["run_id"] == "r1" for e in events)
    assert "".join(s.text_parts) == "Found it.\n" and s.session_id == "conv-9"
    assert s.usage == {"input_tokens": 120, "output_tokens": 8, "cache_read_tokens": 40}
    assert s.error_text == ""


def test_result_text_used_when_nothing_streamed_and_errors_kept() -> None:
    s = aga.AgyStream("c", "r", lambda e: None)
    s.on_line(json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "Hi.\n"}}))
    assert s.text_parts == ["Hi."]
    s.on_line(json.dumps({"event": "error", "error": {"message": "boom"}}))
    assert s.error_text == "boom"
    s.on_line(json.dumps({"event": "result", "result": {"status": "FAILED", "error": "quota exceeded"}}))
    assert s.error_text == "quota exceeded"

def test_argv() -> None:
    argv = aga.build_argv(binary="agy", prompt="hi", model="gemini-3-pro", extra_args="--foo 1",
                          auto_approve=True, add_dir="D", session_id="conv-9")
    assert argv[:5] == ["agy", "-p", "hi", "--output-format", "stream-json"]
    assert argv[argv.index("--model") + 1] == "gemini-3-pro"
    assert argv[argv.index("--conversation") + 1] == "conv-9"
    assert "--dangerously-skip-permissions" in argv and argv[-2:] == ["--foo", "1"]
    plain = aga.build_argv(binary="agy", prompt="hi", model="auto", extra_args="", auto_approve=False)
    assert "--model" not in plain and "--conversation" not in plain and "--dangerously-skip-permissions" not in plain


def test_system_prompt_and_long_message_go_in_files(tmp_path) -> None:
    text = aga.build_prompt(tmp_path, "Be Ducky.", "x" * (aga.ARG_PROMPT_MAX + 1), resumed=False)
    assert (tmp_path / "instructions.md").read_text(encoding="utf-8") == "Be Ducky."
    assert (tmp_path / "message.md").is_file() and len(text) < 1000
    assert aga.build_prompt(tmp_path, "Be Ducky.", "hi", resumed=True) == "hi"


def test_mcp_merge_keeps_other_servers_and_no_chat_identity(tmp_path) -> None:
    host = tmp_path / "host.json"
    bridge = ["node", "bridge.js", "--ducky-run-id", "run-1"]
    host.write_text(json.dumps({"mcpServers": {"uefn": {
        "command": "node", "args": ["bridge.js", "--ducky-run-id", "run-1", "--x"],
        "env": {"DUCKY_CONV_ID": "c1", "DUCKY_BRIDGE_ARGV": json.dumps(bridge), "PATH_EXTRA": "p"}}}}))
    target = tmp_path / "agy" / "mcp_config.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"mcpServers": {"mine": {"command": "x"}}}))

    env = aga.merge_mcp(str(host), target)

    saved = json.loads(target.read_text(encoding="utf-8"))["mcpServers"]
    assert saved["mine"] == {"command": "x"}
    assert saved["uefn"]["args"] == ["bridge.js", "--x"]
    assert "DUCKY_CONV_ID" not in saved["uefn"]["env"] and saved["uefn"]["env"]["PATH_EXTRA"] == "p"
    assert json.loads(saved["uefn"]["env"]["DUCKY_BRIDGE_ARGV"]) == ["node", "bridge.js"]
    assert env["DUCKY_CONV_ID"] == "c1"  # the chat's identity rides the agy process instead
    before = target.stat().st_mtime_ns
    aga.merge_mcp(str(host), target)  # the same server again: the file is left alone
    assert target.stat().st_mtime_ns == before


def test_models_and_sign_in_state() -> None:
    rows = agy_cli.parse_models("Available models:\n  gemini-3-pro   Gemini 3 Pro\n- claude-sonnet-4-5 (Claude)\nauto\n")
    assert [r["id"] for r in rows] == ["gemini-3-pro", "claude-sonnet-4-5"]
    assert rows[0]["name"] == "Gemini 3 Pro"

    def fake(out: str, rc: int = 0):
        return lambda *a, **k: subprocess.CompletedProcess(a, rc, out, "")

    agy_cli.forget_models()
    assert agy_cli.account_models("agy", run=fake("Please sign in to use Antigravity.", 1), force=True) == (False, [])
    state, got = agy_cli.account_models("agy", run=fake("gemini-3-pro Gemini 3 Pro"), force=True)
    assert state is True and got[0]["id"] == "gemini-3-pro"
    agy_cli.forget_models()


class _FakeLogin:
    def __init__(self, lines: list[str], exit_after_code: int | None = 0, says: str = "") -> None:
        self.stdout = iter(lines)
        self.stdin = io.StringIO()
        self._rc: int | None = None
        self._exit = exit_after_code
        self._says = says

    def poll(self) -> int | None:
        if self._rc is None and self.stdin.getvalue():
            if self._says:
                agy_cli._login["out"].append(self._says)
            self._rc = self._exit
        return self._rc

    def kill(self) -> None:
        self._rc = -9


def test_login_opens_the_default_browser_and_takes_the_code(monkeypatch) -> None:
    monkeypatch.setattr(agy_cli, "resolve_bin", lambda override="": "agy")
    url = "https://accounts.google.com/o/oauth2/auth?client_id=x&state=y"
    proc = _FakeLogin(["Starting...\n", f"Open this link: {url}\n"])
    opened: list[str] = []

    out = agy_cli.start_login(popen=lambda *a, **k: proc, open_url=opened.append, wait_s=5)

    assert out["login_ui"] == "code_modal" and out["auth_url"] == url and opened == [url]
    assert agy_cli.login_status()["session_alive"] is True
    done = agy_cli.submit_code(" 4/0Abc-code ", wait_s=2)
    assert done["logged_in"] is True and proc.stdin.getvalue() == "4/0Abc-code\n"
    assert agy_cli.login_status()["logged_in"] is False  # nothing pending any more


def test_a_refused_code_says_why(monkeypatch) -> None:
    monkeypatch.setattr(agy_cli, "resolve_bin", lambda override="": "agy")
    proc = _FakeLogin(["https://accounts.google.com/x\n"], exit_after_code=1, says="invalid_grant: Bad Request\n")
    agy_cli.start_login(popen=lambda *a, **k: proc, open_url=lambda u: None, wait_s=5)
    out = agy_cli.submit_code("bad", wait_s=2)
    assert out["ok"] is False and "invalid_grant" in out["error"]
    assert agy_cli.submit_code("again")["error"].startswith("This sign-in ended")


def test_the_tables_log_in_opens_the_browser_and_the_plugins_code_box(monkeypatch) -> None:
    """Old: a terminal tab waiting for the code, no box. Now Google opens in the browser and
    ui/agy_login.js shows the box while popup_state says open."""
    monkeypatch.setattr(agy_cli, "resolve_bin", lambda override="": "agy.exe")
    proc = _FakeLogin(["https://accounts.google.com/o/oauth2/auth?x=1\n"])
    opened: list[str] = []
    real = agy_cli.start_login
    monkeypatch.setattr(agy_cli, "start_login", lambda cli_path="": real(
        cli_path, popen=lambda *a, **k: proc, open_url=opened.append, wait_s=5))
    assert agy_cli.popup_state()["open"] is False

    assert "message" in aga.account_browser_login("C:/acc")
    assert opened == ["https://accounts.google.com/o/oauth2/auth?x=1"]
    assert agy_cli.popup_state() == {"ok": True, "open": True, "auth_url": opened[0]}
    assert aga.account_status(None) == {"loggedIn": False, "pending": True}  # the table spins

    assert aga.popup_submit("4/0code")["logged_in"] is True
    assert agy_cli.popup_state()["open"] is False


def test_the_rows_code_box_never_also_pops_the_plugins_box(monkeypatch) -> None:
    monkeypatch.setattr(agy_cli, "resolve_bin", lambda override="": "agy.exe")
    proc = _FakeLogin(["https://accounts.google.com/x\n"])
    agy_cli.start_login(popen=lambda *a, **k: proc, open_url=lambda u: None, wait_s=5)
    assert agy_cli.popup_state()["open"] is False
    agy_cli.cancel_login()


def test_a_failed_start_tells_the_table(monkeypatch) -> None:
    monkeypatch.setattr(agy_cli, "start_login", lambda cli_path="": {"ok": False, "error": "downloading"})
    assert aga.account_browser_login(None) == {"error": "downloading"}
    assert aga.account_env("C:/acc") == {}
    monkeypatch.setattr(agy_cli, "account_models", lambda b, **k: (False, []))
    assert aga.account_status(None) == {"loggedIn": False}
    monkeypatch.setattr(agy_cli, "account_models", lambda b, **k: (True, []))
    assert aga.account_status("C:/acc") == {"loggedIn": True}
    monkeypatch.setattr(agy_cli, "resolve_bin", lambda override="": "")
    assert aga.account_status(None)["loggedIn"] is False


def test_a_sign_in_made_elsewhere_reaches_the_model_picker_on_refresh(monkeypatch) -> None:
    """Signed in in a terminal: the picker kept a signed-out list (no models) for 20 minutes."""
    monkeypatch.setattr(agy_cli, "resolve_bin", lambda override="": "agy.exe")
    refreshed: list[int] = []
    monkeypatch.setattr(agy_cli, "refresh_agents", lambda: refreshed.append(1))
    agy_cli._models.update(bin="agy.exe", signed_in=False, at=1e18)  # cached: signed out
    answers = iter([(True, [{"id": "gemini-4-neon-low"}]), (True, [])])
    monkeypatch.setattr(agy_cli, "account_models", lambda b, force=False, **k: (
        agy_cli._models.update(signed_in=(st := next(answers))[0]) or st))
    assert aga.account_status(None) == {"loggedIn": True} and refreshed == [1]
    assert aga.account_status(None) == {"loggedIn": True} and refreshed == [1]  # no change: no re-read
    agy_cli.forget_models()


def test_only_googles_signed_agy_runs_never_a_shim(monkeypatch, tmp_path) -> None:
    """A Google AI plan is only allowed through Antigravity itself: a look-alike `agy` on PATH
    (a shim that talks to Google with the person's sign-in) gets their account banned."""
    shim, real = tmp_path / "agy.cmd", tmp_path / "agy.exe"
    shim.write_text("@echo off")
    real.write_bytes(b"MZ")
    from backend.agent.coding_agents import base

    monkeypatch.setattr(base, "which_cli", lambda name, override="": str(shim))
    monkeypatch.setattr(agy_cli, "managed_bin", lambda: real)
    monkeypatch.setattr(agy_cli, "google_signed", lambda p: p == str(real))
    assert agy_cli.resolve_bin() == str(real)
    monkeypatch.setattr(agy_cli, "google_signed", lambda p: False)
    assert agy_cli.resolve_bin() == ""  # nothing Google-signed: nothing runs


def test_google_signed_reads_the_signer_once(monkeypatch, tmp_path) -> None:
    import pytest

    if os.name != "nt":
        pytest.skip("Authenticode is Windows only")
    exe = tmp_path / "agy.exe"
    exe.write_bytes(b"MZ")
    calls: list[str] = []

    def run(argv, **kw):
        calls.append(kw["env"]["DUCKY_AGY_PATH"])
        return subprocess.CompletedProcess(argv, 0, "google\n" if "good" in str(exe) else "", "")

    monkeypatch.setattr(agy_cli, "hidden", lambda: {})
    agy_cli._signed.clear()
    assert agy_cli.google_signed(str(exe), run=run) is False and agy_cli.google_signed(str(exe), run=run) is False
    assert calls == [str(exe)]  # cached per file version
    exe = tmp_path / "good.exe"
    exe.write_bytes(b"MZ")
    assert agy_cli.google_signed(str(exe), run=run) is True
    assert agy_cli.google_signed(str(tmp_path / "missing.exe"), run=run) is False
    agy_cli._signed.clear()


def test_the_real_install_is_google_signed() -> None:
    """On a PC with Ducky's Antigravity installed: Google's build passes the check."""
    import pytest

    if os.name != "nt" or not agy_cli.managed_bin().is_file():
        pytest.skip("no Antigravity install here")
    assert agy_cli.google_signed(str(agy_cli.managed_bin())) is True


def test_google_signed_uses_system_powershell_and_retries_a_failed_probe(monkeypatch, tmp_path) -> None:
    import pytest

    if os.name != "nt":
        pytest.skip("Authenticode is Windows only")

    exe = tmp_path / "agy.exe"
    exe.write_bytes(b"MZ")
    system_root = tmp_path / "Windows"
    powershell = system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    powershell.parent.mkdir(parents=True)
    powershell.touch()
    path = r"%SystemRoot%\System32\WindowsPowerShell\v1.0"
    module_path = str(tmp_path / "PowerShell 7" / "Modules")
    monkeypatch.setenv("SystemRoot", str(system_root))
    monkeypatch.setenv("PATH", path)
    monkeypatch.setenv("PSModulePath", module_path)
    monkeypatch.setattr(agy_cli, "_signed", {})
    monkeypatch.setattr(agy_cli, "hidden", lambda: {})

    results = iter([(1, "", "Module loading failed"), (0, "google\n", "")])

    def run(argv, **kwargs):
        assert argv[0] == str(powershell)
        env = kwargs["env"]
        assert env["DUCKY_AGY_PATH"] == str(exe)
        assert all(key.casefold() != "psmodulepath" for key in env)
        returncode, stdout, stderr = next(results)
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    assert agy_cli.google_signed(str(exe), run=run) is False
    assert agy_cli.google_signed(str(exe), run=run) is True
    assert agy_cli.google_signed(str(exe), run=run) is True
    assert os.environ["PATH"] == path
    assert os.environ["PSModulePath"] == module_path


def test_limit_errors() -> None:
    assert aga.is_limit_error("RESOURCE_EXHAUSTED: 429") and not aga.is_limit_error("syntax error")


def _fake_agy(monkeypatch, tmp_path, hangs: int) -> list[dict[str, Any]]:
    """agy that goes silent ``hangs`` times (until killed), then answers."""
    import functools
    from types import SimpleNamespace

    calls: list[dict[str, Any]] = []

    def fake(**kw: Any) -> SimpleNamespace:
        calls.append(kw)
        if len(calls) <= hangs:
            assert kw["cancel"].wait(5), "the watchdog never killed a silent agy"
            return SimpleNamespace(cancelled=True, stderr_tail="", raw_tail="")
        kw["on_line"](json.dumps({"event": "result", "result": {"conversation_id": "c2", "status": "SUCCESS",
                                                               "response": "done"}}))
        return SimpleNamespace(cancelled=False, stderr_tail="", raw_tail="")

    monkeypatch.setattr(aga, "run_streaming_process", fake)
    monkeypatch.setattr(aga, "finalize_cli_turn", lambda *, proc, reply, **kw: aga.CodingAgentLaunchResult(
        ok=not proc.cancelled, status="cancelled" if proc.cancelled else "ok", reply=reply))
    monkeypatch.setattr(aga, "watch_stall", functools.partial(aga.watch_stall, idle_s=0.05, poll=0.01))
    monkeypatch.setattr(agy_cli, "resolve_bin", lambda override="": "agy")
    monkeypatch.setattr(aga, "prompt_dir", lambda conv_id: tmp_path)
    return calls


def _launch(**kw: Any):
    return aga.AntigravityAdapter().launch(prompt="fix the lag", system_prompt="sys", cwd=".", conv_id="chat1",
                                           model="", mcp_config_path="", extra_args="", cli_path="", env={},
                                           push=lambda e: None, session_id="c1", **kw)


def test_a_silent_agy_is_restarted_once_in_a_fresh_conversation(monkeypatch, tmp_path) -> None:
    calls = _fake_agy(monkeypatch, tmp_path, hangs=1)
    result = _launch()
    assert result.ok and len(calls) == 2
    assert "--conversation" in calls[0]["argv"] and "--conversation" not in calls[1]["argv"]
    prompt = calls[1]["argv"][calls[1]["argv"].index("-p") + 1]
    assert 'ducky_read_chat(conv_id="chat1")' in prompt and "fix the lag" in prompt


def test_two_stalls_end_with_a_clear_error_never_a_third_run(monkeypatch, tmp_path) -> None:
    calls = _fake_agy(monkeypatch, tmp_path, hangs=2)
    result = _launch()
    assert not result.ok and result.error == aga.STALLED and len(calls) == 2


def test_the_persons_stop_is_never_retried(monkeypatch, tmp_path) -> None:
    import threading

    calls = _fake_agy(monkeypatch, tmp_path, hangs=1)
    stop = threading.Event()
    stop.set()
    result = _launch(cancel=stop)
    assert len(calls) == 1 and result.error != aga.STALLED


def test_a_running_tool_gets_the_longer_limit() -> None:
    import threading

    state = aga.AgyStream("chat1", "r", lambda e: None)
    state._open["1"] = "workspace_compile_verse"
    stop = threading.Event()
    stalled = aga.watch_stall(state, stop, None, idle_s=0.01, tool_idle_s=60, poll=0.01)
    assert not stop.wait(0.1) and not stalled.is_set()
    stop.set()
