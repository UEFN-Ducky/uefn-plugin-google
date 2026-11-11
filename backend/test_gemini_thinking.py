from __future__ import annotations

from backend.gemini_provider import gemini_supports_thinking, gemini_thinking_config, thinking_menu


def test_gemini_3_off_asks_for_the_least_thinking() -> None:
    """No setting at all is Gemini 3's default, the most thinking: Off burned output tokens.
    "low", not "minimal": gemini-3.8-flash answers minimal with a 400."""
    assert gemini_thinking_config("gemini-3.8-flash", "off") == {"thinking_level": "low"}
    assert gemini_thinking_config("gemini-3.1-pro", "off") == {"thinking_level": "low"}


def test_gemini_thinking_config() -> None:
    assert gemini_supports_thinking("gemini-3.7-flash")
    assert gemini_supports_thinking("gemini-2.5-pro")
    assert not gemini_supports_thinking("gemini-2.0-flash")
    assert gemini_thinking_config("gemini-3.7-flash", "low") == {"thinking_level": "low"}
    assert gemini_thinking_config("gemini-3.1-pro", "medium") == {"thinking_level": "high"}
    assert gemini_thinking_config("gemini-2.5-flash", "off") == {"thinking_budget": 0}
    assert gemini_thinking_config("gemini-2.5-pro", "off") is None
    assert gemini_thinking_config("gemini-2.5-flash", "high") == {"thinking_budget": 16384}
    flash = thinking_menu("gemini-2.5-flash")
    assert flash and flash["levels"][0]["thinking_tokens"] == 0
    assert flash["levels"][-1]["thinking_tokens"] == 16384
    assert thinking_menu("gemini-3.7-flash")["levels"][0]["id"] == "off"
