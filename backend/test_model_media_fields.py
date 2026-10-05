"""Gemini model records report video/audio/image limits only when the API says so."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from backend.model_fetch import _gemini_info_from_model


def _model(dump: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        name="models/gemini-x",
        display_name="Gemini X",
        supported_actions=["generateContent"],
        input_token_limit=1_000_000,
        model_dump=lambda: dump,
    )


def test_modalities_and_max_images_from_record():
    info = _gemini_info_from_model(
        _model(
            {
                "input_modalities": ["TEXT", "IMAGE", "VIDEO", "AUDIO"],
                "max_images": 3600,
            }
        )
    )
    assert info.supports_vision is True
    assert (info.supports_video, info.supports_audio, info.max_images) == (True, True, 3600)


def test_max_images_camel_case_key():
    assert _gemini_info_from_model(_model({"maxImages": 16, "input_modalities": ["TEXT"]})).max_images == 16


def test_modalities_listed_without_video_audio_are_false():
    info = _gemini_info_from_model(_model({"input_modalities": ["TEXT", "IMAGE"]}))
    assert (info.supports_video, info.supports_audio) == (False, False)


def test_media_fields_unknown_without_modality_data():
    info = _gemini_info_from_model(_model({}))
    assert (info.max_images, info.supports_video, info.supports_audio) == (None, None, None)
