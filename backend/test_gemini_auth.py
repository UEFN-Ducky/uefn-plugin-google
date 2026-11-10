from __future__ import annotations

import time

import pytest

from backend import cli_update, gemini_auth
from backend.gemini_cli_adapter import account_status


class FakeCli:
    """Stands in for the hidden ``gemini -p`` run; writes the creds when told to."""

    def __init__(self, home, writes: bool) -> None:
        self.home, self.writes, self.killed, self.argv, self.env = home, writes, False, None, None

    def __call__(self, argv, **kw):
        self.argv, self.env = argv, kw["env"]
        if self.writes:
            (self.home / ".gemini" / "oauth_creds.json").write_text("{}", encoding="utf-8")
            (self.home / ".gemini" / "google_accounts.json").write_text('{"active": "you@gmail.com"}', encoding="utf-8")
        return self

    def poll(self):
        return 0 if self.killed else None

    def kill(self):
        self.killed = True


@pytest.fixture()
def cli(monkeypatch):
    monkeypatch.setattr(cli_update, "resolve_bin", lambda _o="": "gemini.cmd")
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-reach-the-cli")
    gemini_auth._STATE.clear()


def _settle(home, timeout=3.0):
    folder = None if home is None else str(home)
    end = time.monotonic() + timeout
    while account_status(folder).get("pending") and time.monotonic() < end:
        time.sleep(0.02)
    return account_status(folder)


def test_browser_sign_in_writes_creds_then_logged_in(cli, tmp_path):
    home = tmp_path / "acc"  # a new account whose folder doesn't exist yet
    fake = FakeCli(home, writes=True)
    out = gemini_auth.start_browser_login(str(home), popen=fake, poll_s=0.01)
    assert out == {"message": gemini_auth.WAITING}
    assert fake.argv[0] == "gemini.cmd" and fake.env["GEMINI_CLI_HOME"] == str(home)
    assert fake.env["GOOGLE_GENAI_USE_GCA"] == "true" and "GEMINI_API_KEY" not in fake.env
    assert _settle(home) == {"loggedIn": True, "email": "you@gmail.com"}
    assert fake.killed  # the prompt run is stopped once signed in


def test_sign_in_that_never_finishes_times_out(cli, tmp_path):
    fake = FakeCli(tmp_path, writes=False)
    gemini_auth.start_browser_login(str(tmp_path), popen=fake, timeout_s=0.1, poll_s=0.01)
    assert account_status(str(tmp_path)) == {"loggedIn": False, "pending": True}
    assert _settle(tmp_path) == {"loggedIn": False, "error": gemini_auth.NOT_FINISHED}
    assert fake.killed


def test_default_account_uses_the_cli_home(cli, tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_CLI_HOME", str(tmp_path))
    fake = FakeCli(tmp_path, writes=True)
    assert "message" in gemini_auth.start_browser_login(None, popen=fake, poll_s=0.01)
    assert _settle(None)["loggedIn"] is True


def test_missing_cli_says_so_and_installs(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(cli_update, "resolve_bin", lambda _o="": "")
    monkeypatch.setattr(gemini_auth, "_install_in_background", lambda: started.append(1))
    assert gemini_auth.start_browser_login(str(tmp_path), popen=None) == {"error": gemini_auth.NOT_INSTALLED}
    assert started == [1]


ENOENT = (
    "Failed to parse default sandbox policy: Error: ENOENT: no such file or directory, open "
    "'C:\\npm\\node_modules\\@google\\gemini-cli\\bundle\\policies\\sandbox-default.toml'\n"
    "    at loadCliConfig\n"
)


class BrokenCli:
    """A half-installed CLI: dies at start with the ENOENT, before any sign-in."""

    def __init__(self) -> None:
        self.stdout = iter([ENOENT])

    def poll(self):
        return 1

    def kill(self):
        pass


def _launches(*procs):
    queue = list(procs)
    return lambda argv, **kw: queue.pop(0)(argv, **kw) if callable(queue[0]) else queue.pop(0)


def test_broken_install_is_repaired_once_then_signs_in(cli, tmp_path, monkeypatch):
    updates = []
    monkeypatch.setattr(cli_update, "update_cli", lambda o="": updates.append(o) or {"ok": True})
    fake = FakeCli(tmp_path, writes=True)
    gemini_auth.start_browser_login(str(tmp_path), popen=_launches(BrokenCli(), fake), poll_s=0.01)
    assert _settle(tmp_path) == {"loggedIn": True, "email": "you@gmail.com"}
    assert updates == [""] and fake.argv[0] == "gemini.cmd"  # one reinstall, then the retry


def test_still_broken_after_reinstall_says_so_without_a_loop(cli, tmp_path, monkeypatch):
    updates = []
    monkeypatch.setattr(cli_update, "update_cli", lambda o="": updates.append(o) or {"ok": True})
    gemini_auth.start_browser_login(str(tmp_path), popen=_launches(BrokenCli(), BrokenCli()), poll_s=0.01)
    st = _settle(tmp_path)
    assert st["loggedIn"] is False and st["error"].startswith(cli_update.BROKEN_INSTALL + ": Failed to parse")
    assert updates == [""]


def test_reinstall_failure_shows_the_npm_error(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli_update, "update_cli", lambda o="": {"ok": False, "error": "npm WARN x\nnpm ERR! code EACCES"})
    gemini_auth.start_browser_login(str(tmp_path), popen=_launches(BrokenCli()), poll_s=0.01)
    assert _settle(tmp_path)["error"] == f"{cli_update.BROKEN_INSTALL}: npm ERR! code EACCES"


def test_repairing_shows_while_npm_runs(cli, tmp_path, monkeypatch):
    import threading

    gate = threading.Event()
    monkeypatch.setattr(cli_update, "update_cli", lambda o="": gate.wait(3) and {"ok": False, "error": "x"})
    gemini_auth.start_browser_login(str(tmp_path), popen=_launches(BrokenCli()), poll_s=0.01)
    end = time.monotonic() + 3
    while not account_status(str(tmp_path)).get("repairing") and time.monotonic() < end:
        time.sleep(0.01)
    assert account_status(str(tmp_path)) == {"loggedIn": False, "pending": True, "repairing": True}
    gate.set()
