from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_zip import build_zip  # noqa: E402


def _names(tmp_path: Path) -> list[str]:
    dest = tmp_path / "plugin.zip"
    build_zip(out=dest)
    with zipfile.ZipFile(dest) as zf:
        return zf.namelist()


def test_the_plugin_payload_is_packed(tmp_path: Path) -> None:
    names = _names(tmp_path)

    assert "plugin.json" in names
    assert "backend/__init__.py" in names
    assert "backend/gemini_provider.py" in names


def test_tests_are_not_shipped_to_users(tmp_path: Path) -> None:
    """Test harness and host stubs have no business in an installed plugin."""
    names = _names(tmp_path)

    shipped_tests = [n for n in names if "test_" in n or "conftest" in n]
    assert shipped_tests == []


def test_compiled_build_is_what_the_store_release_calls(monkeypatch, tmp_path: Path) -> None:
    """release_platforms calls build_compiled_zip(release=True[, target=...]) on this module."""
    import build_zip as bz

    calls = []
    monkeypatch.setattr(bz, "_find_ducky_repo", lambda: tmp_path)
    monkeypatch.setattr(bz, "OUT_DIR", tmp_path)

    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-2]).write_bytes(b"zip")

    monkeypatch.setattr(bz.subprocess, "run", fake_run)
    dest = bz.build_compiled_zip(out=tmp_path / "x-compiled.ducky-plugin.zip", release=True)
    assert dest.is_file()
    assert calls and calls[0][-1] == "--release"
    assert "backend.uefn_plugins.compile" in calls[0]


def test_a_bare_linux_image_gets_the_pinned_cpu_check_dependency(monkeypatch, tmp_path: Path) -> None:
    """The Linux build runs in python:3.13-bookworm, which has no iced_x86 until we install it."""
    import types

    import build_zip as bz

    (tmp_path / "requirements.txt").write_text("requests==2\niced-x86==1.22.0  # cpu check\n", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr(bz.subprocess, "run", fake_run)
    bz._ensure_build_deps(tmp_path, "py")
    assert calls[-1][:4] == ["py", "-m", "pip", "install"]
    assert calls[-1][-1] == "iced-x86==1.22.0"

    calls.clear()
    monkeypatch.setattr(bz.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or types.SimpleNamespace(returncode=0))
    bz._ensure_build_deps(tmp_path, "py")
    assert len(calls) == 1  # already importable: nothing installed
