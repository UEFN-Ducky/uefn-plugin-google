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
