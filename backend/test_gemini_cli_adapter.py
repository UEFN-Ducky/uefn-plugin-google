from __future__ import annotations

from backend.gemini_cli_adapter import GeminiCliAdapter, build_gemini_argv


def _argv(**overrides: object) -> list[str]:
    kwargs: dict[str, object] = {
        "binary": "gemini",
        "prompt": "salut",
        "model": "gemini-flash-latest",
        "extra_args": "",
    }
    kwargs.update(overrides)
    return build_gemini_argv(**kwargs)  # type: ignore[arg-type]


def test_picker_is_empty_without_a_live_catalog(monkeypatch) -> None:
    monkeypatch.setattr("backend.gemini_cli_adapter._gemini_cli_model_rows", lambda: [])
    assert GeminiCliAdapter().detect(settings=None).models == []


def test_auto_approve_is_on_by_default() -> None:
    assert "-y" in _argv()


def test_auto_approve_can_be_turned_off() -> None:
    """-y approves file writes and shell commands: it must not be forced on."""
    assert "-y" not in _argv(auto_approve=False)


def test_model_and_extra_args_still_reach_the_cli() -> None:
    argv = _argv(model="gemini-pro-latest", extra_args="--debug")

    assert argv[:2] == ["gemini", "-p"]
    assert "salut" in argv
    assert argv[argv.index("-m") + 1] == "gemini-pro-latest"
    assert "--debug" in argv
