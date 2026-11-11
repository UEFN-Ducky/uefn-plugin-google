from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from google.genai import types

from backend.agent.providers.base import ProviderMessage, StreamEventKind, ToolCallRequest
from backend.gemini_provider import GeminiProvider


def _provider() -> GeminiProvider:
    return GeminiProvider("test-key", "gemini-3.7-flash")


class _Models:
    """A key's models: list() answers (or refuses); generate_content must never be called."""

    def __init__(self, error: str = "") -> None:
        self.error, self.generated, self.listed = error, [], []

    def generate_content(self, **kw: Any) -> Any:
        self.generated.append(kw)
        return SimpleNamespace(text="pong")

    def list(self, config: Any = None) -> list[Any]:
        self.listed.append(config)
        if self.error:
            raise RuntimeError(self.error)
        return [SimpleNamespace(name="models/gemini-3.8-flash")]


def _key_test(error: str = "") -> tuple[tuple[bool, str], _Models]:
    p = GeminiProvider("test-key", "gemini-2.5-flash")  # a model Google retired: never asked
    models = _Models(error)
    p._client = lambda: SimpleNamespace(models=models)  # type: ignore[method-assign]
    return asyncio.run(p.test_connection()), models


def test_key_test_is_one_model_listing_not_a_generated_reply() -> None:
    """Saving a key 'took forever' (a thinking model answering 'ping'), and a retired test
    model 404'd a good key."""
    (ok, detail), models = _key_test()
    assert ok and detail == "Gemini OK"
    assert models.generated == [] and models.listed == [{"page_size": 1}]


def test_key_test_bad_key_fails_with_googles_reason() -> None:
    (ok, detail), models = _key_test("400 API_KEY_INVALID")
    assert not ok and "API_KEY_INVALID" in detail and models.generated == []

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


def test_tool_results_survive_the_live_context_note_core_ends_every_step_with() -> None:
    """Seen live: get_status + call_tool seven times, then the loop guard, never a reply.
    Core's last message of every step is a user note; the provider read that as a new turn,
    forgot its tool calls and replayed them as text with the results left out."""
    provider = _provider()
    fc = SimpleNamespace(name="search_assets", args={"search": "bench"})
    _run_turn(provider, [_chunk(parts=[SimpleNamespace(text=None, function_call=fc)], finish_reason="STOP")])
    assert len(provider._fc_contents) == 1

    step_two = _tool_loop_messages() + [ProviderMessage(role="user", content="[live context] UEFN online")]
    seen: list[dict] = []
    provider._client = lambda: SimpleNamespace(  # type: ignore[method-assign]
        models=SimpleNamespace(generate_content_stream=lambda **kw: seen.append(kw) or iter([])))

    async def go() -> None:
        async for _ in provider.stream_turn(system="s", messages=step_two, tools=[]):
            pass

    asyncio.run(go())
    responses = _function_responses(seen[0]["contents"])
    assert [r.name for r in responses] == ["search_assets"]
    assert responses[0].response == {"result": '{"assets": ["CP_PrincessCastle_Bench_C"]}'}
    assert len(provider._fc_contents) == 1  # kept for the next step


def test_two_calls_in_one_step_are_answered_in_one_turn() -> None:
    provider = _provider()
    provider._fc_contents = [types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="a", args={})),
        types.Part(function_call=types.FunctionCall(name="b", args={})),
    ])]
    messages = [
        ProviderMessage(role="user", content="go"),
        ProviderMessage(role="assistant", tool_calls=[ToolCallRequest(id="1", name="a", arguments={}),
                                                      ToolCallRequest(id="2", name="b", arguments={})]),
        ProviderMessage(role="tool", tool_call_id="1", content="A"),
        ProviderMessage(role="tool", tool_call_id="2", content="B"),
    ]
    contents = provider._to_gemini_contents(messages)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert [p.function_response.name for p in contents[-1].parts] == ["a", "b"]


def test_an_earlier_turns_tool_result_is_still_told_to_the_model() -> None:
    """Replayed as text (no live call syntax), result included: it used to be left out."""
    contents = _provider()._to_gemini_contents(_tool_loop_messages() + [ProviderMessage(role="user", content="and now?")])
    texts = [p.text for c in contents for p in (c.parts or []) if p.text]
    assert any("Already ran `search_assets`" in t for t in texts)
    assert any("CP_PrincessCastle_Bench_C" in t and "Result of `search_assets`" in t for t in texts)
    assert not _function_responses(contents)


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


def test_text_reaches_the_chat_chunk_by_chunk() -> None:
    """The reply used to be held until the model finished: a long answer looked stuck."""
    chunks = [
        _chunk(parts=[SimpleNamespace(text="Un ", function_call=None)]),
        _chunk(parts=[SimpleNamespace(text="deux ", function_call=None)]),
        _chunk(parts=[SimpleNamespace(text="trois", function_call=None)], finish_reason="STOP"),
    ]

    events = _run_turn(_provider(), chunks)

    assert [e.text for e in events if e.kind is StreamEventKind.TEXT_DELTA] == ["Un ", "deux ", "trois"]
    assert next(e for e in events if e.kind is StreamEventKind.DONE).text == "Un deux trois"


def test_a_leaked_call_split_across_chunks_is_never_shown() -> None:
    head, tail = _LEAKED_EXECUTE_PYTHON[:7], _LEAKED_EXECUTE_PYTHON[7:]
    chunks = [
        _chunk(parts=[SimpleNamespace(text="Je regarde [1]. " + head, function_call=None)]),
        _chunk(parts=[SimpleNamespace(text=tail, function_call=None)], finish_reason="STOP"),
    ]

    events = _run_turn(_provider(), chunks)

    shown = "".join(e.text for e in events if e.kind is StreamEventKind.TEXT_DELTA)
    assert shown == "Je regarde [1]. "
    assert next(e for e in events if e.kind is StreamEventKind.TOOL_CALLS).tool_calls[0].name == "execute_python"


def test_thoughts_are_asked_for_only_when_thinking_is_on() -> None:
    seen: list[dict] = []

    def run(effort: str) -> dict:
        provider = GeminiProvider("k", "gemini-2.5-flash", thinking_effort=effort)
        provider._client = lambda: SimpleNamespace(  # type: ignore[method-assign]
            models=SimpleNamespace(generate_content_stream=lambda **kw: seen.append(kw) or iter([])))

        async def go() -> None:
            async for _ in provider.stream_turn(system="s", messages=[ProviderMessage(role="user", content="x")], tools=[]):
                pass

        asyncio.run(go())
        return seen[-1]["config"]["thinking_config"]

    assert run("high") == {"thinking_budget": 16384, "include_thoughts": True}
    assert run("off") == {"thinking_budget": 0}


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
