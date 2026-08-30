"""Gemini CLI argv + historical tool-turn text. No host imports."""

from __future__ import annotations

import json
import shlex
from typing import Any


def build_gemini_argv(
    *,
    binary: str,
    prompt: str,
    model: str,
    extra_args: str,
    session_id: str = "",
) -> list[str]:
    argv = [binary, "-p", prompt, "-y", "--output-format", "stream-json"]
    sid = (session_id or "").strip()
    if sid:
        argv.extend(["--resume", sid])
    mid = (model or "").strip()
    if mid and mid.lower() not in ("default", "auto"):
        argv.extend(["-m", mid])
    extra = (extra_args or "").strip()
    if extra:
        argv.extend(shlex.split(extra, posix=False))
    return argv


def flatten_historical_tool_text(
    content: str,
    calls: list[tuple[str, str, dict[str, Any]]],
    results: list[tuple[str, str]],
) -> str:
    """Text stand-in for a prior tool turn (no thought_signature on replay)."""
    lines: list[str] = []
    if (content or "").strip():
        lines.append(content.strip())
    result_by_id = {tid: body for tid, body in results if tid}
    used: set[str] = set()
    for tid, name, args in calls:
        lines.append(f"[Called tool {name}({json.dumps(args, ensure_ascii=False)})]")
        body = result_by_id.get(tid, "")
        if body:
            lines.append(f"[Tool result {name}]: {body}")
            used.add(tid)
    for tid, body in results:
        if tid in used or not body:
            continue
        lines.append(f"[Tool result {tid or 'tool'}]: {body}")
    return "\n".join(lines)
