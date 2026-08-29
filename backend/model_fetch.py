"""Vendor model list / pricing fetch for this gateway plugin."""

from __future__ import annotations

import html
import logging
import re
import time
from typing import Any

from backend.agent.model_fetch import (
    ModelInfo,
    _PricingRow,
    _cache_put,
    _float_from_record,
    _merge_prices,
)

_log = logging.getLogger(__name__)
_CACHE_TTL_S = 6 * 3600.0


_GEMINI_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
_PROVIDER_PRICING_CACHE: dict[str, tuple[float, dict[str, _PricingRow]]] = {}


def _pricing_catalog() -> dict[str, _PricingRow]:
    hit = _PROVIDER_PRICING_CACHE.get("gemini")
    if hit is not None and (time.time() - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    catalog = _fetch_gemini_pricing_catalog()
    _cache_put(_PROVIDER_PRICING_CACHE, "gemini", (time.time(), catalog))
    return catalog


def clear_model_cache() -> None:
    _PROVIDER_PRICING_CACHE.clear()


def fetch_models(api_key: str, **_kw: Any) -> list[ModelInfo]:
    return _fetch_gemini(api_key)

def _resolve_gemini_price(catalog: dict[str, _PricingRow], model_id: str) -> _PricingRow | None:
    if not catalog:
        return None
    mid = (model_id or "").strip().lower()
    if not mid:
        return None
    if mid in catalog:
        return catalog[mid]
    candidates = [k for k in catalog if mid.startswith(k) or k.startswith(mid)]
    if not candidates:
        return None
    best = max(candidates, key=len)
    return catalog[best]


def _fetch_gemini_pricing_catalog() -> dict[str, _PricingRow]:
    catalog: dict[str, _PricingRow] = {}
    try:
        import httpx

        r = httpx.get(_GEMINI_PRICING_URL, follow_redirects=True, timeout=30.0)
        r.raise_for_status()
        text = html.unescape(r.text)
        code_re = re.compile(r'<code translate="no" dir="ltr">(gemini-[a-z0-9.-]+)</code>', re.I)
        for m in code_re.finditer(text):
            mid = m.group(1).lower()
            chunk = text[m.end() : m.end() + 5000]
            std = re.search(
                r'<h3[^>]*data-text="Standard"[^>]*>.*?<table class="pricing-table">(.*?)</table>',
                chunk,
                re.S | re.I,
            )
            if not std:
                std = re.search(r'<table class="pricing-table">(.*?)</table>', chunk, re.S)
            if not std:
                continue
            tbl = std.group(1)
            inp = re.search(r"<td>Input price</td>\s*<td>[^<]*</td>\s*<td>\$?([\d.]+)", tbl)
            out = re.search(r"<td>Output price[^<]*</td>\s*<td>[^<]*</td>\s*<td>\$?([\d.]+)", tbl)
            cached = re.search(r"<td>Context caching price</td>\s*<td>[^<]*</td>\s*<td>\$?([\d.]+)", tbl)
            if not inp or not out:
                continue
            catalog[mid] = (
                float(inp.group(1)),
                float(out.group(1)),
                float(cached.group(1)) if cached else None,
                None,
            )
    except Exception as exc:
        _log.warning("Gemini pricing page unavailable: %s", exc)
    return catalog


def _value_mentions_image(value: Any) -> bool:
    if isinstance(value, str):
        return "image" in value.lower()
    if isinstance(value, list):
        return any(_value_mentions_image(v) for v in value)
    if isinstance(value, dict):
        return any(_value_mentions_image(v) for v in value.values()) or any(
            "image" in str(k).lower() for k in value.keys()
        )
    return False


def _record_mentions_image_modalities(record: dict[str, Any]) -> bool:
    for key, value in record.items():
        kl = str(key).lower()
        if "modality" in kl or "modalities" in kl:
            if _value_mentions_image(value):
                return True
        if isinstance(value, dict) and _record_mentions_image_modalities(value):
            return True
    return False


def _gemini_info_from_model(
    m: Any,
    pricing_catalog: dict[str, _PricingRow] | None = None,
) -> ModelInfo | None:
    name = (getattr(m, "name", None) or "").replace("models/", "").strip()
    if not name:
        return None
    actions = [str(a) for a in (getattr(m, "supported_actions", None) or [])]
    if actions and "generateContent" not in actions:
        return None
    dump: dict[str, Any] = {}
    if hasattr(m, "model_dump"):
        try:
            dump = m.model_dump()
        except Exception:
            dump = {}
    vision = _record_mentions_image_modalities(dump) if dump else False
    ctx = getattr(m, "input_token_limit", None)
    context_limit = int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else None
    tools = "generateContent" in actions if actions else False
    price_in = _float_from_record(dump, "input_price_per_million", "input_cost_per_million")
    price_out = _float_from_record(dump, "output_price_per_million", "output_cost_per_million")
    cached = _float_from_record(dump, "cached_input_price_per_million", "cache_read_price_per_million")
    price_in, price_out, cached, _cache_write = _merge_prices(
        (price_in, price_out, cached, None),
        _resolve_gemini_price(pricing_catalog or {}, name),
    )
    return ModelInfo(
        id=name,
        display_name=str(getattr(m, "display_name", None) or name),
        supports_vision=vision,
        supports_tools=tools,
        context_limit=context_limit,
        price_in=price_in,
        price_out=price_out,
        price_cached_in=cached,
    )


def _fetch_gemini(api_key: str) -> list[ModelInfo]:
    from google import genai

    client = genai.Client(api_key=api_key)
    pricing_catalog = _pricing_catalog()
    models: list[ModelInfo] = []
    for m in client.models.list():
        info = _gemini_info_from_model(m, pricing_catalog)
        if info:
            models.append(info)
    models.sort(key=lambda m: m.id, reverse=True)
    return models

