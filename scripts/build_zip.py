#!/usr/bin/env python3
"""Zip plugin.json + backend (+ assets) for Store upload. No secrets.

``--compiled`` builds through the UEFN Ducky build engine instead: the backend becomes an
extension module (no Python source ships). The Store release uses it.
``--target linux-x86_64|linux-arm64|macos-arm64|macos-x64`` is the compiled build on that host (Nuitka can't
cross-compile); the zip name carries the target.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deploy"


def _prune_superseded_zips(keep: pathlib.Path | None = None, pattern: str = "*.ducky-plugin.zip") -> None:
    """Delete every built zip except the one just built.

    deploy/ is gitignored build output; superseded zips are stale files that
    pile up one per version bump and can be re-created by re-running this
    script. Keeping them only invites installing an old build by mistake.
    """
    try:
        zips = sorted(OUT_DIR.glob(pattern))
    except OSError:
        return
    survivor = keep
    if survivor is None and zips:
        survivor = max(zips, key=lambda p: p.stat().st_mtime)
    for z in zips:
        try:
            if survivor is not None and z.samefile(survivor):
                continue
            z.unlink()
        except OSError:
            pass
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
    _prune_superseded_zips(dest)
    return dest


def _find_ducky_repo() -> Path:
    """The UEFN-Ducky checkout holding the build engine: ``DUCKY_REPO``, else a sibling of
    this repo's (or, from a sandbox worktree, its main checkout's) uefn-plugins/ folder."""
    candidates: list[Path] = []
    env = os.environ.get("DUCKY_REPO", "").strip()
    if env:
        candidates.append(Path(env))
    roots = [ROOT]
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=str(ROOT), capture_output=True, text=True, check=False)
    if common.returncode == 0 and common.stdout.strip():
        roots.append(Path(common.stdout.strip()).parent)
    for root in roots:
        github = root.parent.parent
        for name in ("UEFN-Ducky-Release", "UEFN-Ducky"):
            candidates.append(github / name)
    for cand in candidates:
        if (cand / "ducky_app" / "backend" / "uefn_plugins" / "compile.py").is_file():
            return cand
    raise SystemExit(
        "UEFN-Ducky checkout not found — set DUCKY_REPO to it (must contain "
        "ducky_app/backend/uefn_plugins/compile.py)."
    )


TARGETS = ("linux-x86_64", "linux-arm64", "macos-arm64", "macos-x64")


def _host_target() -> str:
    """'linux-x86_64' / 'linux-arm64' on Linux, 'macos-arm64' / 'macos-x64' on a Mac, '' elsewhere
    (Windows builds stay untagged)."""
    import platform

    arch = platform.machine().lower()
    arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(arch, arch)
    if sys.platform.startswith("linux"):
        return f"linux-{arch}"
    if sys.platform == "darwin":
        return "macos-" + {"x86_64": "x64"}.get(arch, arch)
    return ""


def _ducky_python(repo: Path, target: str = "") -> str:
    venv = repo / ".venv" / "bin" / "python" if target else repo / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.is_file() else sys.executable


def _ensure_build_deps(repo: Path, python: str) -> None:
    """A bare Linux build image has none of Ducky's requirements: install the engine's CPU
    check disassembler (iced-x86) at the version Ducky's requirements.txt pins."""
    if subprocess.run([python, "-c", "import iced_x86"], capture_output=True, check=False).returncode == 0:
        return
    pins = [line.split("#")[0].strip() for line in (repo / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    pins = [p for p in pins if p.lower().startswith("iced-x86")]
    if not pins:
        raise SystemExit(f"{repo / 'requirements.txt'} pins no iced-x86 (the build engine's CPU check needs it)")
    print(f"installing {' '.join(pins)} for the CPU check")
    subprocess.run([python, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                    "--root-user-action=ignore", *pins], check=True)


def build_compiled_zip(*, out: Path | None = None, release: bool = False, target: str = "") -> Path:
    """Compiled Store zip via the Ducky build engine CLI. ``release`` bakes the Store license
    check in, so a Store upload is never unlicensed."""
    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    pid = str(manifest.get("id") or "").strip()
    version = manifest.get("version") or 1
    if not pid:
        raise SystemExit("plugin.json missing id")
    host = _host_target()
    target = target or host
    if target and target != host:
        raise SystemExit(f"--target {target} builds on a {target} machine (Nuitka can't cross-compile); "
                         f"this host is {host or sys.platform}")
    tag = f"-{target}" if target else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = out or (OUT_DIR / f"{pid}-{version}{tag}-compiled.ducky-plugin.zip")
    repo = _find_ducky_repo()
    ducky_app = repo / "ducky_app"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ducky_app) + os.pathsep + env.get("PYTHONPATH", "")
    # One C compiler per core can run an agent terminal out of memory; DUCKY_NUITKA_JOBS overrides.
    env.setdefault("DUCKY_NUITKA_JOBS", "2")
    python = _ducky_python(repo, target)
    if target:
        _ensure_build_deps(repo, python)
    cmd = [python, "-m", "backend.uefn_plugins.compile", str(ROOT), str(dest)]
    if release:
        cmd.append("--release")
    print(f"compiling via {repo}")
    subprocess.run(cmd, cwd=str(ducky_app), env=env, check=True)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    _prune_superseded_zips(dest, f"*{tag}-compiled.ducky-plugin.zip" if target else "*.ducky-plugin.zip")
    return dest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiled", action="store_true",
                    help="compile the backend via the Ducky build engine (no Python source ships)")
    ap.add_argument("--target", choices=TARGETS, default="",
                    help="Linux compiled build for this host's CPU (implies --compiled)")
    args = ap.parse_args()
    if args.compiled or args.target:
        build_compiled_zip(target=args.target)
    else:
        _prune_superseded_zips(build_zip())
