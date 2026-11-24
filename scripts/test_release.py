from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release  # noqa: E402


def test_upload_with_ticket_puts_the_zip_and_returns_its_id(monkeypatch, tmp_path: Path) -> None:
    """release_platforms uploads every build through this before uds_release."""
    zip_path = tmp_path / "x.zip"
    zip_path.write_bytes(b"zipbytes")
    asked, put = [], []

    def fake_mcp(base, key, name, args):
        asked.append((name, args))
        return {"payload": {"uploadId": "u1", "putUrl": "https://store.invalid/put"}}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    def fake_urlopen(req, timeout=0):
        put.append((req.full_url, req.get_method(), req.data))
        return _Resp()

    monkeypatch.setattr(release, "mcp_call", fake_mcp)
    monkeypatch.setattr(release.urllib.request, "urlopen", fake_urlopen)
    assert release.upload_with_ticket("https://b", "k", zip_path) == "u1"
    assert asked == [("uds_upload_ticket", {"size": 8, "sha256": hashlib.sha256(b"zipbytes").hexdigest(), "purpose": "release"})]
    assert put == [("https://store.invalid/put", "PUT", b"zipbytes")]
