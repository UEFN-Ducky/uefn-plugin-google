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


MEDIA_INTENT = (r"\b(gemini|google|imagen|veo|nano[- ]?banana|image|picture|draw|illustration|concept art"
                r"|video|clip|animation|trailer)\b")
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
# Video on a Gemini API key is Gemini Omni (the Interactions API); some keys also list Veo.
# Nov 11 2026, on a real key: no Veo model at all, three gemini-omni ones.
VIDEO_DEFAULT = "gemini-omni-1.2-flash"
# What to tell the person about each model when asking which to use. A model not here is listed plain.
MODEL_NOTES = {
    "gemini-2.5-flash-image": "fast, the lowest price (Nano Banana)",
    "gemini-3.1-flash-lite-image": "the lightest and quickest",
    "gemini-3.1-flash-image": "newer, fast, better with text in the picture",
    "gemini-3-pro-image": "the best quality, slower, the highest price",
    "gemini-omni-1.2-flash": "3 to 15 second clips with sound, the newest",
    "gemini-omni-1.1-flash": "the earlier Omni model",
}
VIDEO_WAIT_S = 300.0  # one tool call waits this long; gemini_video_status picks a longer one up
_API = "https://generativelanguage.googleapis.com/v1beta"


def _listed(key: str, *words: str) -> list[str]:
    """Model ids this key lists whose name has one of ``words``, newest first (one free
    listing). A "-preview" twin of a released model is left out."""
    q = urllib.parse.urlencode({"key": key, "pageSize": 1000})
    with urllib.request.urlopen(f"{_API}/models?{q}", timeout=30) as resp:
        rows = json.loads(resp.read().decode("utf-8")).get("models") or []
    ids = {str(row.get("name") or "").replace("models/", "") for row in rows if isinstance(row, dict)}
    return sorted((i for i in ids if any(w in i for w in words)
                   and not (i.endswith("-preview") and i[:-8] in ids)), reverse=True)


def image_models(key: str) -> list[str]:
    return _listed(key, "image")


def video_models(key: str) -> list[str]:
    return _listed(key, "omni", "veo")


def _choices(key: str, kind: str) -> list[dict[str, str]]:
    """What the person can pick from, with a few words on each. A picture listing that fails
    still offers the models known to work. Video never names a model the key did not list:
    the first version offered a guessed Veo name as "available" and Google answered 404."""
    try:
        ids = image_models(key) if kind == "picture" else video_models(key)
    except (OSError, ValueError):
        ids = []
    if not ids and kind == "picture":
        ids = list(IMAGE_FALLBACKS)
    return [{"id": i, **({"about": MODEL_NOTES[i]} if i in MODEL_NOTES else {})} for i in ids]


def _default(models: list[dict[str, str]], kind: str) -> str:
    """The model "auto" means: the usual one when this key lists it, else the first it lists."""
    usual = GEMINI_IMAGE if kind == "picture" else VIDEO_DEFAULT
    ids = [row["id"] for row in models]
    if not ids:
        return usual if kind == "picture" else ""
    return usual if usual in ids else ids[0]


def ask_first(key: str, kind: str, tool: str) -> str:
    """The answer to a call the person has not agreed to yet: nothing was made, nothing was
    billed. It names what is available and what it costs, for the ducky to ask the person."""
    models = _choices(key, kind)
    if not models:
        return json.dumps({"made": False, "billed": False, "models": [], "tell_the_person": (
            "This Google API key lists no video models, so no video can be made with it. Video needs a key "
            f"from a Google account with billing turned on. Nothing was made or billed. {PRICING_URL}")})
    billed = ("Google bills each picture to the person's own Google API key (a free-tier key has a daily allowance)."
              if kind == "picture" else
              "Google bills video per second of footage to the person's own Google API key; it needs a key with "
              "billing on, and one clip can cost a few dollars.")
    return json.dumps({
        "made": False,
        "billed": False,
        "ask_the_person": (f"Nothing was made yet. Ask the person with ducky_ask_user before making this {kind}: "
                           f"which model to use (offer the ones in models, with their notes), and that they agree "
                           f"to the cost. Then call {tool} again with model=<their pick> and confirm_spend=true. "
                           f"Never set confirm_spend=true without their yes for this {kind}."),
        "models": models,
        "default": _default(models, kind),
        "cost": f"{billed} Prices: {PRICING_URL}",
    })


def _find_video(node: Any) -> dict[str, Any]:
    """The first video part (its ``data`` or ``uri``) anywhere in an Interaction's steps."""
    if isinstance(node, dict):
        mime = str(node.get("mime_type") or node.get("mimeType") or "")
        if (node.get("type") == "video" or mime.startswith("video/")) and (node.get("data") or node.get("uri")):
            return node
        node = list(node.values())
    if isinstance(node, list):
        for item in node:
            found = _find_video(item)
            if found:
                return found
    return {}


def _omni_video(model: str, prompt: str, key: str, seconds: int, aspect_ratio: str) -> bytes:
    """One Gemini Omni clip through the Interactions API (Google's cookbook Get_started_Omni):
    the call returns when the clip is made, with the video in its steps."""
    fmt: dict[str, Any] = {"type": "video"}
    if aspect_ratio:
        fmt["aspect_ratio"] = str(aspect_ratio)
    if seconds:
        fmt["duration"] = f"{int(seconds)}s"  # Omni: "3s" to "15s"
    # The web call itself, not the Google library: the copy bundled in Ducky answered 404
    # "Model 'gemini-omni-1.2-flash' not found" for the request this call makes a clip from
    # (Nov 11 2026, same key, same minute: 2.7 MB mp4 in steps[].content[]).
    req = urllib.request.Request(f"{_API}/interactions?{urllib.parse.urlencode({'key': key})}",
                                 data=json.dumps({"model": model, "input": prompt, "response_format": fmt}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=900) as resp:
        video = _find_video(json.loads(resp.read().decode("utf-8")))
    data = video.get("data")
    if isinstance(data, str):
        data = base64.b64decode(data)
    if not data and video.get("uri"):
        data = _veo_download(str(video["uri"]), key)
    if not data:
        raise ValueError("Google finished without a video. Try wording the prompt differently.")
    return bytes(data)


def _veo_start(model: str, prompt: str, key: str, image: Path | None, seconds: int, aspect_ratio: str) -> str:
    """Start one Veo clip; returns the operation's name (Google makes it in the background)."""
    instance: dict[str, Any] = {"prompt": prompt}
    if image:
        mime = mimetypes.guess_type(image.name)[0] or "image/png"
        instance["image"] = {"bytesBase64Encoded": base64.b64encode(image.read_bytes()).decode(), "mimeType": mime}
    params = {k: v for k, v in (("durationSeconds", int(seconds or 0)), ("aspectRatio", str(aspect_ratio or ""))) if v}
    body = {"instances": [instance], **({"parameters": params} if params else {})}
    req = urllib.request.Request(f"{_API}/models/{model}:predictLongRunning?{urllib.parse.urlencode({'key': key})}",
                                 data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        name = str(json.loads(resp.read().decode("utf-8")).get("name") or "")
    if not name:
        raise ValueError("Google did not start the video.")
    return name


def _veo_check(operation: str, key: str) -> dict[str, Any]:
    """{} while Google is still making it, else {"uri"} or {"error"}."""
    with urllib.request.urlopen(f"{_API}/{operation}?{urllib.parse.urlencode({'key': key})}", timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("done"):
        return {}
    if data.get("error"):
        return {"error": str((data["error"] or {}).get("message") or "Google could not make the video.")}
    made = ((data.get("response") or {}).get("generateVideoResponse") or {})
    for sample in made.get("generatedSamples") or []:
        uri = str(((sample or {}).get("video") or {}).get("uri") or "")
        if uri:
            return {"uri": uri}
    reasons = "; ".join(str(r) for r in made.get("raiMediaFilteredReasons") or [])
    return {"error": reasons or "Google finished without a video. Try wording the prompt differently."}


def _veo_download(uri: str, key: str) -> bytes:
    req = urllib.request.Request(uri, headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def _persist_video(raw: bytes, *, prompt: str, model: str) -> dict[str, Any]:
    """Keep the clip with the chat that asked (Ducky's generated media folder)."""
    import time

    from frontend.ui_web import generated_images as gen

    try:
        chat = gen._chat_of_caller()
    except Exception:
        chat = ""
    return gen.save_generated_image(raw, prompt=prompt, gateway="google", model=model,
                                    filename=f"veo_{int(time.time())}.mp4", mime="video/mp4", conv_id=chat)


def _video_answer(operation: str, key: str, prompt: str, model: str, wait_s: float) -> str:
    """Wait for the clip up to ``wait_s``, keep it with the chat, and say where it is; a clip
    still being made answers with its operation for gemini_video_status."""
    import time

    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        state = _veo_check(operation, key)
        if state or time.monotonic() >= deadline:
            break
        time.sleep(8)
    if not state:
        return json.dumps({"made": False, "status": "running", "operation": operation,
                           "next": "Google is still making it. Call gemini_video_status with this operation in a minute."})
    if state.get("error"):
        raise ValueError(f"Google made no video: {state['error']}")
    saved = _persist_video(_veo_download(state["uri"], key), prompt=prompt, model=model)
    path = str(saved.get("path") or "")
    return json.dumps({"ok": True, "made": True, "path": path.replace("\\", "/"), "downloaded": [path],
                       "media_url": str(saved.get("media_url") or ""), "model": model, "maker": "Google Veo"})


def register_tools(api: Any) -> None:
    """Gemini's pictures and Veo's videos as tools every ducky can call. Each one costs the
    person money on their own Google key, so each refuses to run until the person has been
    told what is available and what it costs and has said yes (confirm_spend)."""

    def has_key() -> bool:  # from disk, never the network: it runs on every tool listing
        return bool(_secret("gemini"))

    def tool(name: str) -> Any:
        try:
            return api.tool(name=name, intent=MEDIA_INTENT, visible=has_key)
        except TypeError:  # a host from before visible=
            return api.tool(name=name, intent=MEDIA_INTENT)

    def key_or_raise() -> str:
        key = _secret("gemini")
        if not key:
            raise ValueError("Add a Google API key in Settings → LLMs → Google first.")
        return key

    def refused(exc: Exception) -> ValueError:
        if isinstance(exc, urllib.error.HTTPError):
            return ValueError(_google_reason(exc).replace("the picture", "it"))
        return ValueError(f"Could not reach Google: {exc}")

    @tool("gemini_generate_image")
    def gemini_generate_image(prompt: str, model: str = "", image: str = "", confirm_spend: bool = False,
                              wait: bool = True, output_dir: str = "") -> Any:
        """Make a picture with Google Gemini and show it in this chat. It is billed to the
        person's own Google API key, so ASK FIRST: call it with just the prompt and it answers
        with the models available and the cost, and makes nothing. Ask the person which model
        and whether they agree (ducky_ask_user), then call again with model=<their pick> and
        confirm_spend=true. prompt: what the picture shows. image: path of a reference picture
        to edit or restyle (optional). model="auto" is the default model. Returns the picture."""
        _ = wait
        text = str(prompt or "").strip()
        if not text:
            raise ValueError("Say what the picture shows (prompt).")
        key = key_or_raise()
        picked = str(model or "").strip()
        if not confirm_spend or not picked:
            return ask_first(key, "picture", "gemini_generate_image")
        out = handle_image({"config": {"prompt": text, "model": "" if picked.lower() in ("auto", "default") else picked,
                                       "image": str(image or "").strip()}, "payload": {}})
        if not out.get("ok"):
            raise ValueError(str(out.get("error") or "Google made no picture."))
        info = json.dumps({"ok": True, "made": True, "path": str(out["path"]).replace("\\", "/"),
                           "downloaded": [str(out["path"])], "media_url": out.get("media_url") or "",
                           "model": out.get("model") or "", "maker": "Google Gemini"})
        if output_dir:  # a workflow or Ducky's own picture tool called: it reads the JSON
            return info
        try:
            from backend.tools.core.model_picture import model_picture

            return [info, model_picture(str(out["path"]))]
        except Exception:  # an older host: the card still shows the picture from media_url
            return info

    @tool("gemini_generate_video")
    def gemini_generate_video(prompt: str, model: str = "", image: str = "", seconds: int = 0,
                              aspect_ratio: str = "", confirm_spend: bool = False) -> str:
        """Make a short video with Google (Gemini Omni, or Veo where the key has it) and show it
        in this chat. Video is billed per second of footage to the person's own Google API key
        and needs a key with billing on, so ASK FIRST: call
        it with just the prompt and it answers with the models available and the cost, and
        makes nothing. Ask the person which model and whether they agree (ducky_ask_user), then
        call again with model=<their pick> and confirm_spend=true. image: path of a picture the
        video starts from (Veo models only). seconds (Omni: 3 to 15) / aspect_ratio ("16:9",
        "9:16"): optional, the model's own defaults when left out. Takes a few minutes; a Veo
        clip still being made answers with an operation for gemini_video_status."""
        text = str(prompt or "").strip()
        if not text:
            raise ValueError("Say what the video shows (prompt).")
        key = key_or_raise()
        picked = str(model or "").strip()
        if not confirm_spend or not picked:
            return ask_first(key, "video", "gemini_generate_video")
        start = _image_path(image)
        if start and not Path(start).is_file():
            raise ValueError(f"reference picture not found: {start}")
        if picked.lower() in ("auto", "default"):
            picked = _default(_choices(key, "video"), "video")
            if not picked:
                return ask_first(key, "video", "gemini_generate_video")  # says the key has none
        try:
            if picked.startswith("veo"):
                operation = _veo_start(picked, text, key, Path(start) if start else None, seconds, aspect_ratio)
                return _video_answer(operation, key, text, picked, VIDEO_WAIT_S)
            if start:
                # ponytail: Omni takes pictures as input parts; add when someone needs image-to-video.
                raise ValueError("A starting picture works with Veo models only for now. Leave image out.")
            saved = _persist_video(_omni_video(picked, text, key, seconds, aspect_ratio), prompt=text, model=picked)
            path = str(saved.get("path") or "")
            return json.dumps({"ok": True, "made": True, "path": path.replace("\\", "/"), "downloaded": [path],
                               "media_url": str(saved.get("media_url") or ""), "model": picked,
                               "maker": "Google Gemini Omni"})
        except (urllib.error.HTTPError, OSError) as exc:
            raise refused(exc) from exc
        except ValueError:
            raise
        except Exception as exc:  # the Google library's own errors: its sentence says why
            raise ValueError(f"Google made no video: {str(exc)[:300]}") from exc

    @tool("gemini_video_status")
    def gemini_video_status(operation: str, wait: bool = True) -> str:
        """Check a video gemini_generate_video is still making (free: the clip was already
        agreed and started). Keeps the finished clip with this chat and returns its path."""
        name = str(operation or "").strip()
        if "/operations/" not in name:
            raise ValueError("Pass the operation gemini_generate_video returned.")
        key = key_or_raise()
        model = name.split("/operations/")[0].replace("models/", "")
        try:
            return _video_answer(name, key, "", model, VIDEO_WAIT_S if wait else 0.0)
        except (urllib.error.HTTPError, OSError) as exc:
            raise refused(exc) from exc

    @tool("gemini_image_models")
    def gemini_image_models() -> str:
        """List the Gemini image models the person's Google API key can use, with a few words
        on each (free, nothing is made)."""
        models = _choices(key_or_raise(), "picture")
        return json.dumps({"default": _default(models, "picture"), "models": models})

    @tool("gemini_video_models")
    def gemini_video_models() -> str:
        """List the video models (Gemini Omni, Veo) the person's Google API key can use; an empty
        list means the key cannot make video (free, nothing is made)."""
        models = _choices(key_or_raise(), "video")
        return json.dumps({"default": _default(models, "video"), "models": models})


def register_nodes(api: Any) -> None:
    if not hasattr(api, "register_pipeline_node"):
        return
    api.register_pipeline_node("google.complete", handle_complete)
    api.register_pipeline_node("google.image", handle_image)
