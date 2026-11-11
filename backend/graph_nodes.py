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
# The picture model when none is picked, and for a reference picture (Imagen takes none).
GEMINI_IMAGE = "gemini-2.5-flash-image"
# Tried in order when the asked model is gone (404). Nov 11 2026, on a real key: the old
# default imagen-4.0-generate-001 is "not found for API version v1beta"; these three made pictures.
IMAGE_FALLBACKS = (GEMINI_IMAGE, "gemini-3.1-flash-image", "gemini-3-pro-image")


def _secret(name: str) -> str:
    """The Google key this PC would chat with: the picked key account, else the plain saved
    key. Reading only the plain key answered "Google key required" to everyone whose keys are
    key accounts (every key added since accounts)."""
    try:
        from backend.agent import accounts

        key = (accounts.account_key(name, accounts.pick_account(name)) or "").strip()
        if key:
            return key
    except Exception:
        pass
    from backend.agent.secrets import get_key

    return get_key(name) or ""


def _google_reason(exc: urllib.error.HTTPError) -> str:
    """Google's own sentence for a refused call ("API 400" alone told nobody what to fix)."""
    try:
        message = str(json.loads(exc.read().decode("utf-8", "replace"))["error"]["message"]).strip()
    except Exception:
        message = ""
    return f"Google refused the picture ({exc.code}): {message[:300]}" if message else f"Google refused the picture (HTTP {exc.code})"


def _persist(raw: bytes, *, prompt: str, gateway: str, model: str) -> dict[str, Any]:
    from frontend.ui_web import generated_images as gen

    try:
        chat = gen._chat_of_caller()  # the chat that asked: the picture is kept with it
    except Exception:
        chat = ""
    return gen.save_generated_image(raw, prompt=prompt, gateway=gateway, model=model, conv_id=chat)


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
    model = str(cfg.get("model") or GEMINI_IMAGE)
    image = _image_path(cfg.get("image") or payload.get("image"))
    if image and not Path(image).is_file():
        return {"ok": False, "error": f"reference picture not found: {image}"}
    if image and model.startswith("imagen-"):
        model = GEMINI_IMAGE
    key = _secret("gemini")
    if not key:
        return {"ok": False, "error": "Add a Google API key in Settings → LLMs → Google to make pictures."}
    b64, error = "", ""
    # The asked model, then the ones known to make pictures: a model Google retired (404)
    # moves on instead of failing every picture until the plugin updates.
    for candidate in dict.fromkeys((model, *IMAGE_FALLBACKS)):
        try:
            if candidate.startswith("gemini-"):
                b64 = _gemini_b64(_gemini_http(candidate, _gemini_body(prompt, Path(image) if image else None), key))
            else:
                pred = (_imagen_http(candidate, prompt, key).get("predictions") or [{}])[0]
                b64 = str(pred.get("bytesBase64Encoded") or pred.get("bytes") or "")
        except urllib.error.HTTPError as exc:
            error = error or _google_reason(exc)
            if exc.code == 404:
                continue
            return {"ok": False, "error": error}
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        model = candidate
        break
    if not b64:
        return {"ok": False, "error": error or "Google answered without a picture. Try wording the prompt differently."}
    raw = base64.b64decode(b64)
    saved = _persist(raw, prompt=prompt, gateway="google", model=model)
    path = str(saved.get("path") or "")
    return {
        "ok": True,
        "files": [{"path": path, "name": saved.get("name") or "image.png"}],
        "path": path,
        "prompt": prompt,
        "model": model,
        "media_url": str(saved.get("media_url") or ""),
    }


def generate_image(prompt: str, **options: Any) -> dict[str, Any]:
    """Host media generator (kind 'image'): a Gemini picture → ``{path, mime}``."""
    cfg = {"prompt": prompt, "model": options.get("model") or ""}
    out = handle_image({"config": cfg, "payload": {}})
    if not out.get("ok"):
        raise RuntimeError(out.get("error") or "Google made no picture")
    return {"path": out["path"], "mime": "image/png"}


IMAGE_INTENT = r"\b(gemini|google|imagen|nano[- ]?banana|image|picture|draw|illustration|concept art)\b"


def image_models(key: str) -> list[str]:
    """The image models this key may use, newest names first (one free listing)."""
    q = urllib.parse.urlencode({"key": key, "pageSize": 1000})
    url = f"https://generativelanguage.googleapis.com/v1beta/models?{q}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        rows = json.loads(resp.read().decode("utf-8")).get("models") or []
    ids = [str(row.get("name") or "").replace("models/", "") for row in rows if isinstance(row, dict)]
    return sorted((i for i in ids if "image" in i), reverse=True)


def register_tools(api: Any) -> None:
    """Gemini's pictures as tools every ducky can call: the plugin's own, so no agent has to
    script around it and the picture shows in the chat like any other maker's."""

    def has_key() -> bool:  # from disk, never the network: it runs on every tool listing
        return bool(_secret("gemini"))

    def tool(name: str) -> Any:
        try:
            return api.tool(name=name, intent=IMAGE_INTENT, visible=has_key)
        except TypeError:  # a host from before visible=
            return api.tool(name=name, intent=IMAGE_INTENT)

    @tool("gemini_generate_image")
    def gemini_generate_image(prompt: str, model: str = "", image: str = "") -> Any:
        """Make a picture with Google Gemini and show it in this chat. Runs on the person's own
        Google API key (Settings > LLMs > Google): asking for the picture is the yes, no credits.
        prompt: what the picture shows. image: path of a reference picture to edit or restyle
        (optional). model: a Gemini image model (optional; gemini_image_models lists them; the
        default is gemini-2.5-flash-image). Returns the picture, so you see it, and its path."""
        text = str(prompt or "").strip()
        if not text:
            raise ValueError("Say what the picture shows (prompt).")
        out = handle_image({"config": {"prompt": text, "model": str(model or "").strip(),
                                       "image": str(image or "").strip()}, "payload": {}})
        if not out.get("ok"):
            raise ValueError(str(out.get("error") or "Google made no picture."))
        info = json.dumps({"path": str(out["path"]).replace("\\", "/"), "media_url": out.get("media_url") or "",
                           "model": out.get("model") or "", "maker": "Google Gemini", "cost": "Your own API key"})
        try:
            from backend.tools.core.model_picture import model_picture

            return [info, model_picture(str(out["path"]))]
        except Exception:  # an older host: the card still shows the picture from media_url
            return info

    @tool("gemini_image_models")
    def gemini_image_models() -> str:
        """List the Gemini image models the person's Google API key can use (free, no picture made)."""
        key = _secret("gemini")
        if not key:
            raise ValueError("Add a Google API key in Settings → LLMs → Google to make pictures.")
        try:
            return json.dumps({"default": GEMINI_IMAGE, "models": image_models(key)})
        except urllib.error.HTTPError as exc:
            raise ValueError(_google_reason(exc)) from exc
        except OSError as exc:
            raise ValueError(f"Could not reach Google: {exc}") from exc


def register_nodes(api: Any) -> None:
    if not hasattr(api, "register_pipeline_node"):
        return
    api.register_pipeline_node("google.complete", handle_complete)
    api.register_pipeline_node("google.image", handle_image)
