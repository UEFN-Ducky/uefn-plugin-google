"""Gemini model records report video/audio/image limits only when the API says so."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from backend.model_fetch import _gemini_info_from_model


def _model(dump: dict[str, Any], name: str = "gemma-x") -> SimpleNamespace:
    return SimpleNamespace(
        name=f"models/{name}",
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


def test_input_modalities_without_video_audio_are_false():
    info = _gemini_info_from_model(_model({"input_modalities": ["TEXT", "IMAGE"]}, "gemini-3.5-flash"))
    assert (info.supports_video, info.supports_audio) == (False, False)


def test_output_modalities_never_count_as_input():
    info = _gemini_info_from_model(_model({"output_modalities": ["TEXT", "VIDEO", "AUDIO"]}))
    assert (info.supports_video, info.supports_audio) == (None, None)


def test_media_fields_unknown_without_modality_data():
    info = _gemini_info_from_model(_model({}))
    assert (info.max_images, info.supports_video, info.supports_audio) == (None, None, None)


def _sdk_model(name: str) -> SimpleNamespace:
    # Mirrors google-genai types.Model: no modality fields at all.
    return _model({"name": f"models/{name}", "input_token_limit": 1_000_000}, name)


def test_gemini_without_modality_fields_defaults_to_all_media():
    info = _gemini_info_from_model(_sdk_model("gemini-3.5-flash-lite"))
    assert (info.supports_vision, info.supports_video, info.supports_audio) == (True, True, True)


def test_gemini_explicit_input_modalities_win_over_default():
    info = _gemini_info_from_model(_model({"input_modalities": ["TEXT", "IMAGE"]}, "gemini-3.5-flash"))
    assert (info.supports_vision, info.supports_video, info.supports_audio) == (True, False, False)


def test_non_gemini_keeps_unknown_behaviour():
    info = _gemini_info_from_model(_sdk_model("gemma-3-27b-it"))
    assert (info.supports_vision, info.supports_video, info.supports_audio) == (False, None, None)


def test_live_and_omni_models_get_vision_only_default():
    for name in ("gemini-2.5-flash-live", "gemini-omni-flash"):
        info = _gemini_info_from_model(_sdk_model(name))
        assert (info.supports_vision, info.supports_video, info.supports_audio) == (True, None, None)
