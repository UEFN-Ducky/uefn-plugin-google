from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from google.genai import types

from backend.agent.providers.base import ProviderMessage, StreamEventKind, ToolCallRequest
from backend.gemini_provider import GeminiProvider


def _provider() -> GeminiProvider:
    return GeminiProvider("test-key", "gemini-3.7-flash")


def _tool_loop_messages() -> list[ProviderMessage]:
    """A live tool loop: the model called search_assets and we answered it."""
    return [
        ProviderMessage(role="user", content="trouve un banc"),
        ProviderMessage(
            role="assistant",
            tool_calls=[
                ToolCallRequest(
                    id="gemini_search_assets_0",
                    name="search_assets",
                    arguments={"search": "bench"},
                )
            ],
        ),
        ProviderMessage(
            role="tool",
            tool_call_id="gemini_search_assets_0",
            content='{"assets": ["CP_PrincessCastle_Bench_C"]}',
        ),
    ]


def _in_flight(provider: GeminiProvider, name: str) -> None:
    """Mimic the raw model Content stored during the current tool loop."""
    provider._fc_contents = [
        types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name=name, args={"search": "bench"})
                )
            ],
        )
    ]


def _function_responses(contents: list[types.Content]) -> list[types.FunctionResponse]:
    out: list[types.FunctionResponse] = []
    for content in contents:
        for part in content.parts or []:
            if part.function_response is not None:
                out.append(part.function_response)
    return out


def test_function_response_carries_the_called_function_name() -> None:
    """Gemini pairs a response to its call by name — a constant name breaks the loop."""
    provider = _provider()
    _in_flight(provider, "search_assets")

    contents = provider._to_gemini_contents(_tool_loop_messages())

    responses = _function_responses(contents)
    assert len(responses) == 1
    assert responses[0].name == "search_assets"


def test_function_response_name_falls_back_to_the_call_id() -> None:
    """An unmatched result must still be non-empty: Gemini rejects an empty name."""
    provider = _provider()
    _in_flight(provider, "search_assets")

    messages = _tool_loop_messages()
    messages[2] = ProviderMessage(role="tool", tool_call_id="orphan_id", content="{}")

    responses = _function_responses(provider._to_gemini_contents(messages))
    assert len(responses) == 1
    assert responses[0].name == "orphan_id"


def _chunk(*, parts: list[Any] | None, finish_reason: str = "") -> SimpleNamespace:
    content = None if parts is None else SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content, finish_reason=finish_reason)
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


def _run_turn(provider: GeminiProvider, chunks: list[Any]) -> list[Any]:
    provider._client = lambda: SimpleNamespace(  # type: ignore[method-assign]
        models=SimpleNamespace(generate_content_stream=lambda **kw: iter(chunks))
    )

    async def collect() -> list[Any]:
        return [
            event
            async for event in provider.stream_turn(
                system="sys",
                messages=[ProviderMessage(role="user", content="salut")],
                tools=[],
            )
        ]

    return asyncio.run(collect())


def test_blocked_candidate_without_content_does_not_crash() -> None:
    """A candidate blocked by a safety filter carries no content at all."""
    events = _run_turn(_provider(), [_chunk(parts=None, finish_reason="SAFETY")])

    done = [e for e in events if e.kind is StreamEventKind.DONE]
    assert len(done) == 1


def test_non_stop_finish_reason_is_reported_instead_of_a_silent_empty_reply() -> None:
    """Without this, a filtered or truncated turn looks like a successful empty answer."""
    events = _run_turn(_provider(), [_chunk(parts=None, finish_reason="SAFETY")])

    done = next(e for e in events if e.kind is StreamEventKind.DONE)
    assert done.stop_reason == "SAFETY"


def test_text_parts_still_stream_normally() -> None:
    chunks = [_chunk(parts=[SimpleNamespace(text="bonjour", function_call=None)], finish_reason="STOP")]

    events = _run_turn(_provider(), chunks)

    deltas = [e.text for e in events if e.kind is StreamEventKind.TEXT_DELTA]
    done = next(e for e in events if e.kind is StreamEventKind.DONE)
    assert deltas == ["bonjour"]
    assert done.text == "bonjour"
    assert done.stop_reason == "stop"


_LEAKED_EXECUTE_PYTHON = (
    '[Tool call: mcp__uefn__execute_python({"code": '
    '"import unreal\\nacts = unreal.EditorLevelLibrary.get_all_level_actors()\\nprint(len(acts))\\n"})]'
)


def test_text_channel_tool_call_is_recovered_and_not_shown() -> None:
    """Gemini sometimes verbalizes the call as `[Tool call: …]` instead of a functionCall part."""
    chunks = [_chunk(parts=[SimpleNamespace(text=_LEAKED_EXECUTE_PYTHON, function_call=None)], finish_reason="STOP")]

    events = _run_turn(_provider(), chunks)

    tool_events = [e for e in events if e.kind is StreamEventKind.TOOL_CALLS]
    done = next(e for e in events if e.kind is StreamEventKind.DONE)
    deltas = [e.text for e in events if e.kind is StreamEventKind.TEXT_DELTA]
    assert len(tool_events) == 1
    call = tool_events[0].tool_calls[0]
    assert call.name == "execute_python"
    assert "get_all_level_actors" in call.arguments["code"]
    assert "[Tool call" not in (done.text or "")
    assert not any("[Tool call" in (d or "") for d in deltas)
    assert done.stop_reason == "tool_calls"


def test_narration_around_a_leaked_call_is_kept() -> None:
    text = "Je vérifie.\n" + _LEAKED_EXECUTE_PYTHON + "\n"
    chunks = [_chunk(parts=[SimpleNamespace(text=text, function_call=None)], finish_reason="STOP")]

    events = _run_turn(_provider(), chunks)

    done = next(e for e in events if e.kind is StreamEventKind.DONE)
    assert "Je vérifie." in (done.text or "")
    assert "[Tool call" not in (done.text or "")
    tool_events = [e for e in events if e.kind is StreamEventKind.TOOL_CALLS]
    assert tool_events[0].tool_calls[0].name == "execute_python"


def test_native_function_call_strips_mcp_prefix() -> None:
    fc = SimpleNamespace(name="mcp__uefn__execute_python", args={"code": "print(1)"})
    chunks = [_chunk(parts=[SimpleNamespace(text=None, function_call=fc)], finish_reason="STOP")]

    events = _run_turn(_provider(), chunks)

    call = next(e for e in events if e.kind is StreamEventKind.TOOL_CALLS).tool_calls[0]
    assert call.name == "execute_python"
    assert call.arguments == {"code": "print(1)"}


def test_thought_text_is_not_shown_as_the_reply() -> None:
    thought = SimpleNamespace(text="hmm", function_call=None, thought=True)
    visible = SimpleNamespace(text="ok", function_call=None)
    chunks = [_chunk(parts=[thought, visible], finish_reason="STOP")]

    events = _run_turn(_provider(), chunks)

    thinking = [e.text for e in events if e.kind is StreamEventKind.THINKING]
    deltas = [e.text for e in events if e.kind is StreamEventKind.TEXT_DELTA]
    assert thinking == ["hmm"]
    assert deltas == ["ok"]


def test_malformed_tool_call_text_is_left_alone() -> None:
    chunks = [_chunk(parts=[SimpleNamespace(text="[Tool call: nope]", function_call=None)], finish_reason="STOP")]

    events = _run_turn(_provider(), chunks)

    assert not [e for e in events if e.kind is StreamEventKind.TOOL_CALLS]
    done = next(e for e in events if e.kind is StreamEventKind.DONE)
    assert "[Tool call: nope]" in (done.text or "")


def test_historical_tool_turns_are_not_written_as_live_call_syntax() -> None:
    """`[Called tool name(json)]` in history is what Gemini copies into the text channel."""
    provider = _provider()
    messages = [
        ProviderMessage(role="user", content="go"),
        ProviderMessage(
            role="assistant",
            tool_calls=[
                ToolCallRequest(id="x", name="execute_python", arguments={"code": "print(1)"})
            ],
        ),
        ProviderMessage(role="tool", tool_call_id="x", content="ok"),
        ProviderMessage(role="user", content="suite"),
    ]

    blob = "\n".join(
        (p.text or "")
        for content in provider._to_gemini_contents(messages)
        for p in (content.parts or [])
    )
    assert "[Called tool" not in blob
    assert "[Tool call" not in blob
    assert "execute_python" in blob
