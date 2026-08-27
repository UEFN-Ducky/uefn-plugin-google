from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema_sanitize import sanitize_gemini_schema


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
