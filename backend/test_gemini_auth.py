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
