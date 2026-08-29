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
