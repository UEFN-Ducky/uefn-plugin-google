"""Google Gemini provider."""

from __future__ import annotations

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
from .schema_sanitize import sanitize_gemini_schema


class GeminiProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        # Raw model Content objects from function_call responses in the current
        # tool-loop turn, preserving thought_signatures for thinking models.
        self._fc_contents: list[Any] = []

    def _client(self):
        from google import genai

        return genai.Client(api_key=self._api_key)

    def _to_gemini_contents(self, messages: list[ProviderMessage]) -> list[Any]:
        from google.genai import types
        import json as _json

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
                    # functionCall parts without thought_signatures.
                    lines: list[str] = []
                    if m.content:
                        lines.append(m.content)
                    for tc in m.tool_calls:
                        args_str = _json.dumps(tc.arguments, ensure_ascii=False)
                        lines.append(f"[Called tool {tc.name}({args_str})]")
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
        tool_calls: list[ToolCallRequest] = []
        gemini_tools = self._to_gemini_tools(tools)
        cancelled = False
        usage: dict[str, int] = {}
        # Accumulate raw response Parts so we can echo them back with
        # thought_signatures intact on the next tool-loop iteration.
        raw_response_parts: list[Any] = []
        finish_reason = ""

        response = client.models.generate_content_stream(
            model=self._model,
            contents=self._to_gemini_contents(messages),
            config={
                "system_instruction": system,
                "tools": gemini_tools if gemini_tools else None,
            },
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
                if part.text:
                    collected_text += part.text
                    yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=part.text)
                fc = getattr(part, "function_call", None)
                if fc and fc.name:
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ToolCallRequest(
                            id=f"gemini_{fc.name}_{len(tool_calls)}",
                            name=fc.name,
                            arguments=args,
                        )
                    )
        if cancelled:
            return
        # Store the raw model Content (with thought_signatures) so the next
        # tool-loop iteration can echo it back without losing signatures.
        if tool_calls and raw_response_parts:
            from google.genai import types

            self._fc_contents.append(
                types.Content(role="model", parts=list(raw_response_parts))
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
