from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema_sanitize import sanitize_gemini_schema


def test_required_names_missing_from_properties_are_dropped() -> None:
    """400 'parameters.required[0]: property is not defined' on a chat through the API."""
    raw = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "required": {"type": "boolean"},  # a parameter that is itself named "required"
            "opts": {"type": "object", "properties": {}, "required": ["gone"]},
        },
        "required": ["path", "missing", "required"],
    }
    out = sanitize_gemini_schema(raw)
    assert out["required"] == ["path", "required"]
    assert out["properties"]["required"] == {"type": "boolean"}
    assert "required" not in out["properties"]["opts"]  # none defined: the key goes
    assert "required" not in sanitize_gemini_schema({"type": "object", "required": ["x"]})


def test_sanitize_strips_additional_properties() -> None:
    raw = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "meta": {"type": "object", "additional_properties": True, "title": "Meta"},
        },
    }
    slim = sanitize_gemini_schema(raw)
    dumped = str(slim)
    assert "additionalProperties" not in dumped
    assert "additional_properties" not in dumped
    assert "title" not in slim["properties"]["meta"]
    assert slim["properties"]["nodes"]["items"]["type"] == "object"
