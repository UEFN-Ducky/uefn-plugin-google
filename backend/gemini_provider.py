"""Google Gemini provider."""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

from backend.agent.multimodal_content import build_gemini_user_parts
from backend.agent.prompt_cache import PromptCachePayload
from backend.agent.providers.base import (
    ProviderMessage,
    StreamEvent,
    StreamEventKind,
    ToolCallRequest,
)
from backend.agent.providers.cache_utils import parse_gemini_usage
from backend.agent.thinking_effort import normalize_thinking_effort
from .schema_sanitize import sanitize_gemini_schema

BUDGET = {"low": 2048, "medium": 8192, "high": 16384}


def gemini_supports_thinking(model: str) -> bool:
    mid = (model or "").strip().lower()
    return "gemini-2.5" in mid or "gemini-2-5" in mid or "gemini-3" in mid


def thinking_menu(model: str) -> dict | None:
    if not gemini_supports_thinking(model):
        return None
    mid = (model or "").strip().lower()
    if "gemini-3" in mid:
        return {
            "lo": "Faster",
            "hi": "Smarter",
            "levels": [
                {"id": "off", "label": "Off", "thinking_tokens": 0, "hint": "No extended thinking"},
                {"id": "low", "label": "Low", "thinking_tokens": None, "hint": "thinking_level=low, no token cap"},
                {"id": "medium", "label": "Med", "thinking_tokens": None, "hint": "thinking_level=medium, no token cap"},
                {"id": "high", "label": "High", "thinking_tokens": None, "hint": "thinking_level=high, no token cap"},
            ],
        }
    return {
        "lo": "Faster",
        "hi": "Smarter",
        "levels": [
            {"id": "off", "label": "Off", "thinking_tokens": 0, "hint": "thinking_budget=0"},
            {"id": "low", "label": "Low", "thinking_tokens": BUDGET["low"], "hint": f"{BUDGET['low']} thinking tokens"},
            {"id": "medium", "label": "Med", "thinking_tokens": BUDGET["medium"], "hint": f"{BUDGET['medium']} thinking tokens"},
            {"id": "high", "label": "High", "thinking_tokens": BUDGET["high"], "hint": f"{BUDGET['high']} thinking tokens"},
        ],
    }


def gemini_thinking_config(model: str, thinking_effort: str) -> dict[str, Any] | None:
    """Gemini 3.x uses thinking_level; 2.5 uses thinking_budget."""
    if not gemini_supports_thinking(model):
        return None
    mid = (model or "").strip().lower()
    effort = normalize_thinking_effort(thinking_effort)
    is_3 = "gemini-3" in mid
    if is_3:
        if effort == "off":
            # Gemini 3 can't turn thinking off, and no setting at all means its default: the
            # most thinking, billed as output. Off asks for "low", the least every Gemini 3
            # model takes ("minimal" is a 400 on gemini-3.8-flash, tried live Nov 11 2026).
            return {"thinking_level": "low"}
        level = "high" if ("pro" in mid and effort == "medium") else effort
        return {"thinking_level": level}
    if effort == "off":
        if "pro" in mid:
            return None
        return {"thinking_budget": 0}
    return {"thinking_budget": BUDGET.get(effort, 8192)}

# Where a text-channel tool call starts: live text is shown up to here, the rest is read
# once the turn ends (a call must never flash on screen as text).
_LEAK_START = re.compile(r"\[(?:Tool call:|Called tool)", re.IGNORECASE)
_LEAK_HEAD_MAX = len("[Called tool")


def streamable_end(text: str, shown: int) -> int:
    """How far ``text`` can be shown now: up to a leaked call, or a ``[`` that may start one."""
    hit = _LEAK_START.search(text, shown)
    if hit:
        return hit.start()
    bracket = text.rfind("[", max(shown, len(text) - _LEAK_HEAD_MAX))
    return bracket if bracket >= 0 else len(text)


# Gemini (lite / thinking, especially) sometimes writes the call as text
# instead of a functionCall part — then the turn looks finished and the user
# has to type "continue". Skills also teach Cursor's mcp__server__tool names.
_LEAKED_CALL_HEAD = re.compile(
    r"\[(?:Tool call:|Called tool)\s+([A-Za-z_][A-Za-z0-9_./-]*)\s*\(",
    re.IGNORECASE,
)


def canonical_gemini_tool_name(name: str) -> str:
    n = (name or "").strip()
    if n.startswith("mcp__") and n.count("__") >= 2:
        return n.rsplit("__", 1)[-1]
    return n


def _read_json_object(src: str, start: int) -> tuple[Any, int]:
    """Parse a JSON object at src[start]. Returns (obj, index_after) or (None, start)."""
    if start >= len(src) or src[start] != "{":
        return None, start
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(src)):
        ch = src[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(src[start : i + 1]), i + 1
                except json.JSONDecodeError:
                    return None, start
    return None, start


def extract_leaked_tool_calls(text: str) -> tuple[list[ToolCallRequest], str]:
    """Turn `[Tool call: name({json})]` blobs into real calls; return leftover prose."""
    if not text:
        return [], text
    calls: list[ToolCallRequest] = []
    out: list[str] = []
    pos = 0
    while True:
        match = _LEAKED_CALL_HEAD.search(text, pos)
        if not match:
            out.append(text[pos:])
            break
        out.append(text[pos : match.start()])
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        obj, after = _read_json_object(text, cursor)
        if not isinstance(obj, dict):
            out.append(text[match.start() : match.end()])
            pos = match.end()
            continue
        cursor = after
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor < len(text) and text[cursor] == ")":
            cursor += 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor < len(text) and text[cursor] == "]":
            cursor += 1
        name = canonical_gemini_tool_name(match.group(1))
        calls.append(
            ToolCallRequest(
                id=f"gemini_{name}_{len(calls)}",
                name=name,
                arguments=obj,
            )
        )
        pos = cursor
    return calls, "".join(out).strip()


def _merge_tool_calls(
    existing: list[ToolCallRequest], extra: list[ToolCallRequest]
) -> None:
    seen = {
        (t.name, json.dumps(t.arguments, sort_keys=True, default=str)) for t in existing
    }
    for call in extra:
        key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
        if key not in seen:
            existing.append(call)
            seen.add(key)


class GeminiProvider:
    def __init__(self, api_key: str, model: str, *, thinking_effort: str = "off", **_kw: Any) -> None:
        self._api_key = api_key
        self._model = model
        self._thinking_effort = normalize_thinking_effort(thinking_effort)
        # Raw model Content objects from function_call responses in the current
        # tool-loop turn, preserving thought_signatures for thinking models.
        self._fc_contents: list[Any] = []

    def _client(self):
        from google import genai
        from google.genai import types

        # The "-latest" aliases and preview models live on v1beta only; some SDK builds default to v1.
        return genai.Client(api_key=self._api_key, http_options=types.HttpOptions(api_version="v1beta"))

    def _to_gemini_contents(self, messages: list[ProviderMessage]) -> list[Any]:
        from google.genai import types

        contents: list[Any] = []

        # Identify assistant messages that carry tool_calls so we can tell
        # which ones have matching stored raw Content (current tool-loop turn)
        # vs historical ones (from a previous model or turn).
        fc_indices: list[int] = [
            i for i, m in enumerate(messages)
            if m.role == "assistant" and m.tool_calls
        ]
        n_fc = len(fc_indices)
        # The last stored contents belong to the last fc-assistant messages (this run's tool
        # loop); earlier ones are historical and lack thought_signatures. When core trimmed
        # some of this run's steps away, the newest stored ones are the ones still here.
        stored = self._fc_contents[-n_fc:] if n_fc else []
        raw_map: dict[int, Any] = dict(zip(fc_indices[n_fc - len(stored):], stored))

        # Gemini pairs a functionResponse to its functionCall by NAME (unlike
        # Anthropic/OpenAI, which pair by id), and rejects an empty one. Answering
        # every call with the same placeholder name leaves the model believing its
        # calls went unanswered, so it repeats them until the turn hits max turns.
        call_names: dict[str, str] = {
            tc.id: tc.name
            for m in messages
            if m.role == "assistant" and m.tool_calls
            for tc in m.tool_calls
        }

        # When we flatten a historical assistant+tool_calls to text, the
        # subsequent tool-role messages are orphaned — skip them.
        skip_tool_msgs = False
        answering = False  # the last content is this step's function responses

        for i, m in enumerate(messages):
            if m.role == "tool":
                name = call_names.get(m.tool_call_id) or m.tool_call_id or "tool"
                if skip_tool_msgs:
                    # A step replayed as text (an earlier turn, or this provider was made
                    # anew mid-run): its result goes back as text too. Leaving it out hid
                    # every earlier tool result from the model, which then ran the tool again.
                    part = types.Part.from_text(text=f"Result of `{name}`:\n{m.content}")
                else:
                    part = types.Part.from_function_response(name=name, response={"result": m.content})
                # The answers to one step's calls go back as one turn, as Gemini sent them.
                if answering:
                    contents[-1].parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))
                answering = True
                continue
            answering = False

            # Any non-tool message resets the skip flag.
            skip_tool_msgs = False
            role = "model" if m.role == "assistant" else "user"

            if m.role == "assistant" and m.tool_calls:
                if i in raw_map:
                    # Current-turn: echo raw Content (preserves thought_signatures).
                    contents.append(raw_map[i])
                else:
                    # Historical: flatten to plain text so the API doesn't see
                    # functionCall parts without thought_signatures. Do not
                    # write `[Called tool name({json})]` — Gemini copies that
                    # into the text channel as a fake live call.
                    lines: list[str] = []
                    if m.content:
                        lines.append(m.content)
                    names = ", ".join(f"`{tc.name}`" for tc in m.tool_calls)
                    if names:
                        lines.append(f"Already ran {names}.")
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text="\n".join(lines))],
                        )
                    )
                    skip_tool_msgs = True
                continue

            parts: list[Any] = []
            if m.role == "user" and m.attachments:
                parts.extend(build_gemini_user_parts(m.content, m.attachments))
            elif m.content:
                parts.append(types.Part.from_text(text=m.content))
            if parts:
                contents.append(types.Content(role=role, parts=parts))
        return contents

    def _to_gemini_tools(self, tools: list[dict[str, Any]]) -> list[Any]:
        from google.genai import types

        decls = []
        for t in tools:
            decls.append(
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=sanitize_gemini_schema(
                        t.get("parameters") or {"type": "object", "properties": {}}
                    ),
                )
            )
        return [types.Tool(function_declarations=decls)] if decls else []

    async def stream_turn(
        self,
        *,
        system: str,
        messages: list[ProviderMessage],
        tools: list[dict[str, Any]],
        cancel_event: Any | None = None,
        cache: PromptCachePayload | None = None,
    ) -> AsyncIterator[StreamEvent]:
        # No reset of the stored tool-call contents here. It used to clear them whenever the
        # last message wasn't a tool result, as "a new user turn". Core ends every step's
        # messages with a live-context note from the user, so they were cleared on every
        # step: each tool call was replayed as "Already ran X" with its result left out, the
        # model never saw an answer and called the same tools until the loop guard stopped it
        # (seen live Nov 11 2026: get_status + call_tool seven times, no reply). A provider
        # lives for one run, so the list starts empty by itself.
        client = self._client()
        collected_text = ""
        thought_text = ""
        tool_calls: list[ToolCallRequest] = []
        gemini_tools = self._to_gemini_tools(tools)
        cancelled = False
        usage: dict[str, int] = {}
        # Accumulate raw response Parts so we can echo them back with
        # thought_signatures intact on the next tool-loop iteration.
        raw_response_parts: list[Any] = []
        finish_reason = ""
        native_function_call = False

        config: dict[str, Any] = {
            "system_instruction": system,
            "tools": gemini_tools if gemini_tools else None,
        }
        thinking = gemini_thinking_config(self._model, self._thinking_effort)
        if thinking:
            if self._thinking_effort != "off":
                # Thought summaries, so a long think shows as thinking instead of a stuck chat.
                thinking = {**thinking, "include_thoughts": True}
            config["thinking_config"] = thinking
        shown = 0  # how much of collected_text has been sent to the chat
        response = client.models.generate_content_stream(
            model=self._model,
            contents=self._to_gemini_contents(messages),
            config=config,
        )
        for chunk in response:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                cancelled = True
                break
            usage_metadata = getattr(chunk, "usage_metadata", None)
            if usage_metadata is not None:
                usage = parse_gemini_usage(usage_metadata)
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            reason = getattr(candidate, "finish_reason", None)
            if reason:
                finish_reason = getattr(reason, "name", None) or str(reason)
            # A candidate stopped by a safety filter or the token cap carries no content.
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                raw_response_parts.append(part)
                fc = getattr(part, "function_call", None)
                if fc and fc.name:
                    native_function_call = True
                    name = canonical_gemini_tool_name(fc.name)
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ToolCallRequest(
                            id=f"gemini_{name}_{len(tool_calls)}",
                            name=name,
                            arguments=args,
                        )
                    )
                if getattr(part, "thought", False):
                    if part.text:
                        thought_text += part.text
                        yield StreamEvent(kind=StreamEventKind.THINKING, text=part.text)
                    continue
                if part.text:
                    collected_text += part.text
                    # Live: the reply used to arrive in one piece when the model finished,
                    # so a long answer looked like a chat that never answered.
                    end = streamable_end(collected_text, shown)
                    if end > shown:
                        yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=collected_text[shown:end])
                        shown = end
        if cancelled:
            return
        _, held_back = extract_leaked_tool_calls(collected_text[shown:])
        leaked, collected_text = extract_leaked_tool_calls(collected_text)
        _merge_tool_calls(tool_calls, leaked)
        if not tool_calls:
            thought_leaked, _ = extract_leaked_tool_calls(thought_text)
            _merge_tool_calls(tool_calls, thought_leaked)
        if held_back:
            yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=held_back)
        # Store the raw model Content (with thought_signatures) so the next
        # tool-loop iteration can echo it back without losing signatures.
        if tool_calls:
            self._store_function_call_content(
                raw_response_parts, tool_calls, native=native_function_call
            )
        if tool_calls:
            yield StreamEvent(kind=StreamEventKind.TOOL_CALLS, tool_calls=tool_calls, usage=usage)
            stop_reason = "tool_calls"
        elif finish_reason and finish_reason.upper() != "STOP":
            # Surface a filtered or truncated turn, or the app shows it as a
            # successful empty reply with no hint of why nothing came back.
            stop_reason = finish_reason
        else:
            stop_reason = "stop"
        yield StreamEvent(
            kind=StreamEventKind.DONE,
            text=collected_text,
            stop_reason=stop_reason,
            usage=usage,
        )

    def _store_function_call_content(
        self,
        raw_response_parts: list[Any],
        tool_calls: list[ToolCallRequest],
        *,
        native: bool,
    ) -> None:
        from google.genai import types

        if native and raw_response_parts:
            self._fc_contents.append(
                types.Content(role="model", parts=list(raw_response_parts))
            )
            return
        # Text-channel recovery: synthesize functionCall parts so the next
        # loop can pair functionResponse by name.
        sig = next(
            (
                getattr(part, "thought_signature", None)
                for part in raw_response_parts
                if getattr(part, "thought_signature", None)
            ),
            None,
        )
        parts: list[Any] = []
        for call in tool_calls:
            part = types.Part.from_function_call(name=call.name, args=call.arguments)
            if sig is not None:
                try:
                    part.thought_signature = sig
                except Exception:
                    pass
            parts.append(part)
        if parts:
            self._fc_contents.append(types.Content(role="model", parts=parts))

    async def test_connection(self) -> tuple[bool, str]:
        """Is the key good? One models.list page: a second or less, where a generated
        "ping" on a thinking model took long, and a model Google retired for new keys
        404'd a good key. A bad or blocked key fails here the same way."""
        try:
            client = self._client()
            pager = client.models.list(config={"page_size": 1})
            next(iter(pager), None)
            return True, "Gemini OK"
        except Exception as e:
            return False, str(e)
