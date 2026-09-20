"""Google Automations + Pipelines tiles. No BrainRot imports."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

NODES = ("google.complete", "google.image")


def _secret(name: str) -> str:
    from backend.agent.secrets import get_key

    return get_key(name) or ""


def _persist(raw: bytes, *, prompt: str, gateway: str, model: str) -> dict[str, Any]:
    from frontend.ui_web.generated_images import save_generated_image

    return save_generated_image(raw, prompt=prompt, gateway=gateway, model=model)


def _pair(ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = ctx.get("config") if isinstance(ctx.get("config"), dict) else {}
    payload = ctx.get("payload") if isinstance(ctx.get("payload"), dict) else {}
    return cfg, payload


def handle_complete(ctx: dict[str, Any]) -> dict[str, Any]:
    cfg, payload = _pair(ctx)
    prompt = str(cfg.get("prompt") or payload.get("prompt") or payload.get("text") or "")
    model = str(cfg.get("model") or payload.get("model") or "")
    from backend.automations.llm_complete import complete_prompt

    return complete_prompt("gemini", prompt, model)


def _imagen_http(model: str, prompt: str, key: str) -> dict[str, Any]:
    q = urllib.parse.urlencode({"key": key})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict?{q}"
    body = {"instances": [{"prompt": prompt}], "parameters": {"sampleCount": 1}}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def handle_image(ctx: dict[str, Any]) -> dict[str, Any]:
    cfg, payload = _pair(ctx)
    prompt = str(cfg.get("prompt") or payload.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt required"}
    model = str(cfg.get("model") or "imagen-4.0-generate-001")
    key = _secret("gemini")
    if not key:
        return {"ok": False, "error": "Google key required"}
    try:
        data = _imagen_http(model, prompt, key)
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"imagen API {exc.code}"}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    pred = (data.get("predictions") or [{}])[0]
    b64 = str(pred.get("bytesBase64Encoded") or pred.get("bytes") or "")
    if not b64:
        return {"ok": False, "error": "no image bytes"}
    raw = base64.b64decode(b64)
    saved = _persist(raw, prompt=prompt, gateway="google", model=model)
    path = str(saved.get("path") or "")
    return {
        "ok": True,
        "files": [{"path": path, "name": saved.get("name") or "image.png"}],
        "path": path,
        "prompt": prompt,
    }


def register_nodes(api: Any) -> None:
    if not hasattr(api, "register_pipeline_node"):
        return
    api.register_pipeline_node("google.complete", handle_complete)
    api.register_pipeline_node("google.image", handle_image)
