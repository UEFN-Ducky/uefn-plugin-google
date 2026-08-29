#!/usr/bin/env python3
"""Zip plugin.json + backend (+ assets) for Store upload. No secrets."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deploy"
SKIP_NAMES = {".git", "scripts", "deploy", ".gitignore", "README.md", "__pycache__"}
SKIP_SUFFIX = {".pyc", ".pyo", ".zip", ".ducky-plugin"}
SKIP_FILES = {"conftest.py"}


def _is_packable(rel_parts: tuple[str, ...], suffix: str) -> bool:
    if not rel_parts or rel_parts[0] in SKIP_NAMES:
        return False
    # Dotted anywhere in the path, not just the leaf: .pytest_cache, .venv, .github.
    if any(part.startswith(".") for part in rel_parts):
        return False
    if suffix in SKIP_SUFFIX:
        return False
    name = rel_parts[-1]
    return name not in SKIP_FILES and not name.startswith("test_")


def build_zip(*, out: Path | None = None) -> Path:
    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    pid = str(manifest.get("id") or "").strip()
    version = manifest.get("version") or 1
    if not pid:
        raise SystemExit("plugin.json missing id")
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".dat", ".env", ".pem", ".key"}:
            raise SystemExit(f"refusing to pack secret-looking file: {path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = out or (OUT_DIR / f"{pid}-{version}.ducky-plugin.zip")
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(ROOT).parts
            if not _is_packable(rel_parts, path.suffix.lower()):
                continue
            zf.write(path, arcname="/".join(rel_parts))
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    return dest


if __name__ == "__main__":
    build_zip()
