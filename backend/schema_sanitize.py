"""Strip JSON Schema keys Gemini's FunctionDeclaration proto rejects."""

from __future__ import annotations

from typing import Any

# Unknown name "additional_properties" → 400 INVALID_ARGUMENT.
_GEMINI_SCHEMA_DROP = frozenset(
    {
        "additionalProperties",
        "additional_properties",
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "unevaluatedProperties",
        "examples",
        "prefixItems",
        "title",
    }
)


def sanitize_gemini_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [sanitize_gemini_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    return {
        k: sanitize_gemini_schema(v)
        for k, v in node.items()
        if k not in _GEMINI_SCHEMA_DROP
    }
