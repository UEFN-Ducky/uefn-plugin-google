from __future__ import annotations

from backend.gemini_provider import gemini_supports_thinking, gemini_thinking_config


def test_gemini_thinking_config() -> None:
    assert gemini_supports_thinking("gemini-3.7-flash")
    assert gemini_supports_thinking("gemini-2.5-pro")
    assert not gemini_supports_thinking("gemini-2.0-flash")
    assert gemini_thinking_config("gemini-3.7-flash", "off") is None
    assert gemini_thinking_config("gemini-3.7-flash", "low") == {"thinking_level": "low"}
    assert gemini_thinking_config("gemini-3.1-pro", "medium") == {"thinking_level": "high"}
    assert gemini_thinking_config("gemini-2.5-flash", "off") == {"thinking_budget": 0}
    assert gemini_thinking_config("gemini-2.5-pro", "off") is None
    assert gemini_thinking_config("gemini-2.5-flash", "high") == {"thinking_budget": 16384}
