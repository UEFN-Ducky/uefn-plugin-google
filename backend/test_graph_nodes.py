from __future__ import annotations

import json
from pathlib import Path


def test_plugin_declares_nodes():
    data = json.loads((Path(__file__).resolve().parent.parent / "plugin.json").read_text(encoding="utf-8"))
    ids = {n["id"] for n in data["contributes"]["automations"]["nodes"]}
    assert ids == {"google.complete", "google.image"}


def _picture(b64: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": "ok"}, {"inlineData": {"data": b64}}]}}]}


def _gone(model: str):
    import io
    import urllib.error

    body = json.dumps({"error": {"message": f"models/{model} is not found for API version v1beta"}}).encode()
    return urllib.error.HTTPError("https://x", 404, "Not Found", {}, io.BytesIO(body))


def test_image_writes_png(tmp_path, monkeypatch):
    from backend import graph_nodes

    png = b"\x89PNG\r\n\x1a\n"
    monkeypatch.setattr(graph_nodes, "_gemini_http",
                        lambda model, body, key: _picture(__import__("base64").b64encode(png).decode()))
    monkeypatch.setattr(graph_nodes, "_secret", lambda *_a, **_k: "g-test")

    def _save(raw, **kw):
        dest = tmp_path / "out.png"
        dest.write_bytes(raw)
        return {"ok": True, "path": str(dest), "name": "out.png"}

    monkeypatch.setattr(graph_nodes, "_persist", _save)
    out = graph_nodes.handle_image({"config": {"prompt": "a duck"}, "payload": {}})
    assert out["ok"] is True
    assert Path(out["path"]).is_file()
    assert "brainrot" not in Path(graph_nodes.__file__).read_text(encoding="utf-8")


def test_image_node_takes_a_reference_picture(tmp_path, monkeypatch):
    import base64

    from backend import graph_nodes

    data = json.loads((Path(__file__).resolve().parent.parent / "plugin.json").read_text(encoding="utf-8"))
    node = next(n for n in data["contributes"]["automations"]["nodes"] if n["id"] == "google.image")
    assert node["accepts_image"] is True  # core lists it on Edit Image
    assert any(f["id"] == "image" for f in node["config_fields"])
    offered = [o["id"] for f in node["config_fields"] if f["id"] == "model" for o in f["options"]]
    assert offered and set(offered) <= set(graph_nodes.IMAGE_FALLBACKS)  # only models that made pictures live

    png = b"\x89PNG\r\n\x1a\n"
    b64 = base64.b64encode(png).decode()
    calls = []
    monkeypatch.setattr(graph_nodes, "_gemini_http", lambda model, body, key: calls.append(("gemini", model, body))
                        or _picture(b64))
    monkeypatch.setattr(graph_nodes, "_secret", lambda *_a, **_k: "g-test")
    monkeypatch.setattr(graph_nodes, "_persist", lambda raw, **kw: {"path": str(tmp_path / "out.png")})
    front = tmp_path / "front.png"
    front.write_bytes(png)

    assert graph_nodes.handle_image({"config": {"prompt": "side view", "image": str(front)}})["ok"] is True
    kind, model, body = calls[-1]
    assert (kind, model) == ("gemini", "gemini-2.5-flash-image")
    assert body["contents"][0]["parts"] == [{"text": "side view"}, {"inline_data": {"mime_type": "image/png", "data": b64}}]
    graph_nodes.handle_image({"config": {"prompt": "side view"}, "payload": {"image": {"path": str(front)}}})
    assert calls[-1][0] == "gemini"
    # No picture and no model picked: the Gemini image model (Imagen's old default is a 404).
    assert graph_nodes.handle_image({"config": {"prompt": "a duck"}})["ok"] is True
    assert calls[-1][:2] == ("gemini", "gemini-2.5-flash-image")
    assert calls[-1][2]["contents"][0]["parts"] == [{"text": "a duck"}]
    missing = graph_nodes.handle_image({"config": {"prompt": "x", "image": str(tmp_path / "nope.png")}})
    assert missing["ok"] is False and "not found" in missing["error"]
    assert len(calls) == 3


def test_a_model_google_retired_moves_on_to_one_that_makes_pictures(tmp_path, monkeypatch):
    """Nov 11 2026: imagen-4.0-generate-001, the default and what saved workflows hold, is a 404."""
    from backend import graph_nodes

    tried = []

    def imagen(model, prompt, key):
        tried.append(model)
        raise _gone(model)

    def gemini(model, body, key):
        tried.append(model)
        if model == "gemini-2.5-flash-image":
            raise _gone(model)
        return _picture("iVBORw==")

    monkeypatch.setattr(graph_nodes, "_imagen_http", imagen)
    monkeypatch.setattr(graph_nodes, "_gemini_http", gemini)
    monkeypatch.setattr(graph_nodes, "_secret", lambda *_a: "g-test")
    saved = {}
    monkeypatch.setattr(graph_nodes, "_persist", lambda raw, **kw: saved.update(kw) or {"path": str(tmp_path / "x.png")})
    out = graph_nodes.handle_image({"config": {"prompt": "a duck", "model": "imagen-4.0-generate-001"}})
    assert out["ok"] is True
    assert tried == ["imagen-4.0-generate-001", "gemini-2.5-flash-image", "gemini-3.1-flash-image"]
    assert saved["model"] == "gemini-3.1-flash-image"  # the model that really made it

    monkeypatch.setattr(graph_nodes, "_gemini_http", lambda model, body, key: (_ for _ in ()).throw(_gone(model)))
    out = graph_nodes.handle_image({"config": {"prompt": "a duck"}})
    assert out["ok"] is False and "is not found for API version" in out["error"]  # Google's own sentence


def test_a_refusal_says_googles_reason(monkeypatch):
    import io
    import urllib.error

    from backend import graph_nodes

    body = json.dumps({"error": {"message": "Quota exceeded for quota metric 'Generate Content API requests'"}}).encode()
    monkeypatch.setattr(graph_nodes, "_secret", lambda *_a: "g-test")
    monkeypatch.setattr(graph_nodes, "_gemini_http", lambda model, b, key: (_ for _ in ()).throw(
        urllib.error.HTTPError("https://x", 429, "Too Many", {}, io.BytesIO(body))))
    out = graph_nodes.handle_image({"config": {"prompt": "a duck"}})
    assert out == {"ok": False, "error": "Google refused the picture (429): Quota exceeded for quota metric "
                                         "'Generate Content API requests'"}


def test_the_key_is_the_picked_key_account(monkeypatch):
    """Keys added since accounts are key accounts: the plain lookup said 'Google key required'."""
    import sys
    import types

    import backend.agent as agent_pkg
    from backend import graph_nodes

    fake = types.SimpleNamespace(pick_account=lambda p: "ed86", account_key=lambda p, a: f"KEY-{a}")
    monkeypatch.setattr(agent_pkg, "accounts", fake, raising=False)
    monkeypatch.setitem(sys.modules, "backend.agent.accounts", fake)
    assert graph_nodes._secret("gemini") == "KEY-ed86"


def test_registers_gemini_media_generator(tmp_path, monkeypatch):
    import pytest

    from backend import graph_nodes, register
    from backend.test_registration import _RecordingApi

    class _Api(_RecordingApi):
        def register_media_generator(self, kind, fn, describe=None):
            self.media = {kind: (fn, describe)}

    api = _Api()
    register(api)
    gen, desc = api.media["image"]
    assert desc == {"label": "Google Gemini", "secret_key": "gemini"}

    seen = {}
    monkeypatch.setattr(graph_nodes, "_secret", lambda *_a: "g-test")
    monkeypatch.setattr(graph_nodes, "_gemini_http",
                        lambda model, body, key: seen.update(model=model) or _picture("iVBORw=="))
    monkeypatch.setattr(graph_nodes, "_persist", lambda raw, **kw: {"path": str(tmp_path / "x.png")})
    assert gen("a duck") == {"path": str(tmp_path / "x.png"), "mime": "image/png"}
    assert seen["model"] == "gemini-2.5-flash-image"
    gen("a duck", model="gemini-3-pro-image")
    assert seen["model"] == "gemini-3-pro-image"

    monkeypatch.setattr(graph_nodes, "_secret", lambda *_a: "")
    with pytest.raises(RuntimeError, match="Add a Google API key"):
        gen("a duck")

    old = _RecordingApi()  # an older host has no register_media_generator
    register(old)
    assert not hasattr(old, "media")
