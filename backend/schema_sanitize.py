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
    out = {
        k: sanitize_gemini_schema(v)
        for k, v in node.items()
        if k not in _GEMINI_SCHEMA_DROP
    }
    # "required[0]: property is not defined" → 400: Gemini refuses a required name missing
    # from properties (other providers ignore it). Keep only names that are defined.
    required = out.get("required")
    if isinstance(required, list):
        props = out.get("properties")
        names = set(props) if isinstance(props, dict) else set()
        kept = [r for r in required if isinstance(r, str) and r in names]
        if kept:
            out["required"] = kept
        else:
            out.pop("required")
    return out
