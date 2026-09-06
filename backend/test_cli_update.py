from __future__ import annotations

from pathlib import Path

from cli_update import (
    is_cli_missing_error,
    is_cli_too_old_error,
    parse_version_tuple,
    plugin_package_version,
    should_heal_launch,
)


def test_parse_and_heal():
    assert parse_version_tuple("0.9.1") == (0, 9, 1)
    assert is_cli_too_old_error("does not support this model; version 1.0.0 or newer")
    assert is_cli_missing_error("gemini: no such file or directory")
    assert should_heal_launch("cannot find the path")
    assert not should_heal_launch("quota exceeded")


def test_plugin_wires_auto_update():
    init = Path(__file__).with_name("__init__.py").read_text(encoding="utf-8")
    adapter = Path(__file__).with_name("gemini_cli_adapter.py").read_text(encoding="utf-8")
    assert "schedule_cli_update_on_plugin_load" in init
    assert "update_cli" in adapter
    assert plugin_package_version() == "1.0.23"


if __name__ == "__main__":
    test_parse_and_heal()
    test_plugin_wires_auto_update()
    print("ok")
