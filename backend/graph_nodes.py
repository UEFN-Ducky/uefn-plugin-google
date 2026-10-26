"""Google Automations + Pipelines tiles. No BrainRot imports."""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

NODES = ("google.complete", "google.image")
# Imagen takes no picture: a reference picture goes to this Gemini image model unless one is picked.
GEMINI_IMAGE = "gemini-2.5-flash-image"


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


def _gemini_http(model: str, body: dict[str, Any], key: str) -> dict[str, Any]:
    q = urllib.parse.urlencode({"key": key})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?{q}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _gemini_body(prompt: str, image: Path | None) -> dict[str, Any]:
    """The prompt, with the reference picture as inline data when there is one."""
    parts: list[dict[str, Any]] = [{"text": prompt}]
    if image:
        mime = mimetypes.guess_type(image.name)[0] or "image/png"
        parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(image.read_bytes()).decode()}})
    return {"contents": [{"parts": parts}], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}


def _gemini_b64(data: dict[str, Any]) -> str:
    for cand in data.get("candidates") or []:
        for part in ((cand or {}).get("content") or {}).get("parts") or []:
            inline = part.get("inline_data") or part.get("inlineData") or {}
            if inline.get("data"):
                return str(inline["data"])
    return ""


def _image_path(value: Any) -> str:
    """A picture's path from a path or a file ref ({path})."""
    raw = value.get("path") if isinstance(value, dict) else value
    return str(raw or "").strip()


def handle_image(ctx: dict[str, Any]) -> dict[str, Any]:
    cfg, payload = _pair(ctx)
    prompt = str(cfg.get("prompt") or payload.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt required"}
    model = str(cfg.get("model") or "imagen-4.0-generate-001")
    image = _image_path(cfg.get("image") or payload.get("image"))
    if image and not Path(image).is_file():
        return {"ok": False, "error": f"reference picture not found: {image}"}
    if image and model.startswith("imagen-"):
        model = GEMINI_IMAGE
    key = _secret("gemini")
    if not key:
        return {"ok": False, "error": "Google key required"}
    gemini = model.startswith("gemini-")
    try:
        if gemini:
            b64 = _gemini_b64(_gemini_http(model, _gemini_body(prompt, Path(image) if image else None), key))
        else:
            pred = (_imagen_http(model, prompt, key).get("predictions") or [{}])[0]
            b64 = str(pred.get("bytesBase64Encoded") or pred.get("bytes") or "")
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"{'gemini' if gemini else 'imagen'} API {exc.code}"}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
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


def generate_image(prompt: str, **options: Any) -> dict[str, Any]:
    """Host media generator (kind 'image'): Imagen picture → ``{path, mime}``."""
    cfg = {"prompt": prompt, "model": options.get("model") or ""}
    out = handle_image({"config": cfg, "payload": {}})
    if not out.get("ok"):
        raise RuntimeError(out.get("error") or "Imagen failed")
    return {"path": out["path"], "mime": "image/png"}


def register_nodes(api: Any) -> None:
    if not hasattr(api, "register_pipeline_node"):
        return
    api.register_pipeline_node("google.complete", handle_complete)
    api.register_pipeline_node("google.image", handle_image)
