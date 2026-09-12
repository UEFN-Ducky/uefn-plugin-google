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
from backend.agent.thinking_effort import EFFORT_BUDGET, normalize_thinking_effort
from .schema_sanitize import sanitize_gemini_schema


def gemini_supports_thinking(model: str) -> bool:
    mid = (model or "").strip().lower()
    return "gemini-2.5" in mid or "gemini-2-5" in mid or "gemini-3" in mid


def gemini_thinking_config(model: str, thinking_effort: str) -> dict[str, Any] | None:
    """Gemini 3.x uses thinking_level; 2.5 uses thinking_budget."""
    if not gemini_supports_thinking(model):
        return None
    mid = (model or "").strip().lower()
    effort = normalize_thinking_effort(thinking_effort)
    is_3 = "gemini-3" in mid
    if is_3:
        if effort == "off":
            return None
        level = "high" if ("pro" in mid and effort == "medium") else effort
        return {"thinking_level": level}
    if effort == "off":
        if "pro" in mid:
            return None
        return {"thinking_budget": 0}
    return {"thinking_budget": EFFORT_BUDGET.get(effort, 8192)}

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

        return genai.Client(api_key=self._api_key)

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
        n_stored = len(self._fc_contents)
        n_fc = len(fc_indices)
        # The last `n_stored` fc-assistant messages come from the current
        # tool-loop; earlier ones are historical and lack thought_signatures.
        raw_map: dict[int, Any] = {}
        if n_stored > 0 and n_fc > 0:
            for offset, msg_i in enumerate(fc_indices[max(0, n_fc - n_stored):]):
                if offset < n_stored:
                    raw_map[msg_i] = self._fc_contents[offset]

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

        for i, m in enumerate(messages):
            if m.role == "tool":
                if skip_tool_msgs:
                    continue
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=call_names.get(m.tool_call_id) or m.tool_call_id or "tool",
                                response={"result": m.content},
                            )
                        ],
                    )
                )
                continue

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
        # Reset stored raw contents on a new user turn (not a tool-loop continuation).
        if messages and messages[-1].role != "tool":
            self._fc_contents = []

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
            config["thinking_config"] = thinking
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
        if cancelled:
            return
        leaked, collected_text = extract_leaked_tool_calls(collected_text)
        _merge_tool_calls(tool_calls, leaked)
        if not tool_calls:
            thought_leaked, _ = extract_leaked_tool_calls(thought_text)
            _merge_tool_calls(tool_calls, thought_leaked)
        if collected_text:
            yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=collected_text)
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
        try:
            client = self._client()
            r = client.models.generate_content(
                model=self._model,
                contents="ping",
            )
            _ = r.text
            return True, "Gemini OK"
        except Exception as e:
            return False, str(e)
