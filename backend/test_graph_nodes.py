from __future__ import annotations

import json
from pathlib import Path


def test_plugin_declares_nodes():
    data = json.loads((Path(__file__).resolve().parent.parent / "plugin.json").read_text(encoding="utf-8"))
    ids = {n["id"] for n in data["contributes"]["automations"]["nodes"]}
    assert ids == {"google.complete", "google.image"}


def test_image_writes_png(tmp_path, monkeypatch):
    from backend import graph_nodes

    png = b"\x89PNG\r\n\x1a\n"
    monkeypatch.setattr(
        graph_nodes,
        "_imagen_http",
        lambda model, prompt, key: {"predictions": [{"bytesBase64Encoded": __import__("base64").b64encode(png).decode()}]},
    )
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


def test_registers_imagen_media_generator(tmp_path, monkeypatch):
    import pytest

    from backend import graph_nodes, register
    from backend.test_registration import _RecordingApi

    class _Api(_RecordingApi):
        def register_media_generator(self, kind, fn, describe=None):
            self.media = {kind: (fn, describe)}

    api = _Api()
    register(api)
    gen, desc = api.media["image"]
    assert desc == {"label": "Google Imagen", "secret_key": "gemini"}

    seen = {}
    monkeypatch.setattr(graph_nodes, "_secret", lambda *_a: "g-test")
    monkeypatch.setattr(
        graph_nodes,
        "_imagen_http",
        lambda model, prompt, key: seen.update(model=model) or {"predictions": [{"bytesBase64Encoded": "iVBORw=="}]},
    )
    monkeypatch.setattr(graph_nodes, "_persist", lambda raw, **kw: {"path": str(tmp_path / "x.png")})
    assert gen("a duck") == {"path": str(tmp_path / "x.png"), "mime": "image/png"}
    assert seen["model"] == "imagen-4.0-generate-001"
    gen("a duck", model="imagen-3.0-generate-002")
    assert seen["model"] == "imagen-3.0-generate-002"

    monkeypatch.setattr(graph_nodes, "_secret", lambda *_a: "")
    with pytest.raises(RuntimeError, match="key required"):
        gen("a duck")

    old = _RecordingApi()  # an older host has no register_media_generator
    register(old)
    assert not hasattr(old, "media")
