from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_argv import build_gemini_argv, flatten_historical_tool_text


def test_first_turn_has_no_resume() -> None:
    argv = build_gemini_argv(
        binary="gemini",
        prompt="hello",
        model="gemini-2.5-pro",
        extra_args="",
    )
    assert "--resume" not in argv
    assert argv[:4] == ["gemini", "-p", "hello", "-y"]


def test_followup_passes_resume_id() -> None:
    argv = build_gemini_argv(
        binary="gemini",
        prompt="what roles",
        model="gemini-2.5-flash",
        extra_args="",
        session_id="sess-abc",
    )
    assert argv[argv.index("--resume") + 1] == "sess-abc"
    assert "-p" in argv
    assert "what roles" in argv


def test_flatten_keeps_tool_result() -> None:
    text = flatten_historical_tool_text(
        "Checking…",
        [("t1", "ping", {})],
        [("t1", '{"ok":true}')],
    )
    assert "Checking" in text
    assert "ping" in text
    assert '{"ok":true}' in text
