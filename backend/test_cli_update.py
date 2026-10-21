from __future__ import annotations

import os
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

import cli_update
from cli_update import (
    hidden_run_kwargs,
    npm_argv,
    is_cli_missing_error,
    is_cli_too_old_error,
    parse_version_tuple,
    plugin_package_version,
    schedule_cli_update_on_plugin_load,
    should_heal_launch,
    update_cli,
)


def test_parse_and_heal():
    assert parse_version_tuple("0.9.1") == (0, 9, 1)
    assert is_cli_too_old_error("does not support this model; version 1.0.0 or newer")
    assert is_cli_missing_error("gemini: no such file or directory")
    assert should_heal_launch("cannot find the path")
    assert not should_heal_launch("quota exceeded")


def test_npm_argv_skips_cmd(tmp_path, monkeypatch):
    node = tmp_path / "node.exe"
    node.write_bytes(b"")
    cli = tmp_path / "node_modules" / "npm" / "bin" / "npm-cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("", encoding="utf-8")
    npm = tmp_path / "npm.cmd"
    npm.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr("cli_update.shutil.which", lambda _name: None)
    argv = npm_argv(str(npm), ["install", "-g", "@google/gemini-cli"])
    assert argv[:2] == [str(node), str(cli)]
    if os.name == "nt":
        assert hidden_run_kwargs().get("startupinfo") is not None


def test_plugin_wires_auto_update():
    init = Path(__file__).with_name("__init__.py").read_text(encoding="utf-8")
    adapter = Path(__file__).with_name("gemini_cli_adapter.py").read_text(encoding="utf-8")
    assert "schedule_cli_update_on_plugin_load" in init
    assert "update_cli" in adapter
    manifest = json.loads(Path(__file__).resolve().parents[1].joinpath("plugin.json").read_text(encoding="utf-8"))
    assert plugin_package_version() == manifest["version"]


def _core_reason(monkeypatch, reason):
    core = types.ModuleType("frontend.ide_apply")
    core.auto_apply_skip_reason = lambda: reason
    monkeypatch.setitem(sys.modules, "frontend.ide_apply", core)


@pytest.mark.parametrize("older_core", [False, True])
def test_sandboxed_ducky_never_runs_npm(monkeypatch, older_core):
    """A sandboxed test build (panel port 4300) ran `npm install -g
    @google/gemini-cli` into the person's real npm; neither plugin load nor a
    launch with no Gemini CLI may."""
    run = Mock(side_effect=AssertionError("npm or gemini spawned"))
    started = []
    monkeypatch.setattr(cli_update, "_run", run)
    monkeypatch.setattr(cli_update, "needs_plugin_load_update", Mock(return_value=True))
    monkeypatch.setattr(cli_update, "threading", types.SimpleNamespace(
        Thread=lambda **kw: types.SimpleNamespace(start=lambda: started.append(kw["name"])),
    ))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("DUCKY_SKIP_CLI_UPDATE", raising=False)
    monkeypatch.setenv("UEFN_DUCKY_PANEL_PORT", "4300")
    reason = "panel port 4300 is not the default 4200"
    if older_core:
        monkeypatch.setitem(sys.modules, "frontend.ide_apply", None)  # core without the check
    else:
        _core_reason(monkeypatch, reason)
    schedule_cli_update_on_plugin_load()
    assert started == []
    result = update_cli("")
    assert result["ok"] is False and reason in result["error"]
    run.assert_not_called()

    _core_reason(monkeypatch, "")  # the person's own Ducky still updates
    schedule_cli_update_on_plugin_load()
    assert started == ["gemini-cli-update"]


if __name__ == "__main__":
    test_parse_and_heal()
    test_plugin_wires_auto_update()
    print("ok")
