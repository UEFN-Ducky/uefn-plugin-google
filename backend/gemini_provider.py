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


class GeminiProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _client(self):
        from google import genai

        return genai.Client(api_key=self._api_key)

    def _to_gemini_contents(self, messages: list[ProviderMessage]) -> list[Any]:
        from google.genai import types

        contents: list[Any] = []
        for m in messages:
            if m.role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name="tool",
                                response={"result": m.content},
                            )
                        ],
                    )
                )
                continue
            role = "model" if m.role == "assistant" else "user"
            parts: list[Any] = []
            if m.role == "user" and m.attachments:
                parts.extend(build_gemini_user_parts(m.content, m.attachments))
            elif m.content:
                parts.append(types.Part.from_text(text=m.content))
            if m.tool_calls:
                for tc in m.tool_calls:
                    parts.append(
                        types.Part.from_function_call(name=tc.name, args=tc.arguments)
                    )
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
                    parameters=t.get("parameters") or {"type": "object", "properties": {}},
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
        client = self._client()
        collected_text = ""
        tool_calls: list[ToolCallRequest] = []
        gemini_tools = self._to_gemini_tools(tools)
        cancelled = False
        usage: dict[str, int] = {}

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
            for part in chunk.candidates[0].content.parts or []:
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
        if tool_calls:
            yield StreamEvent(kind=StreamEventKind.TOOL_CALLS, tool_calls=tool_calls, usage=usage)
        yield StreamEvent(
            kind=StreamEventKind.DONE,
            text=collected_text,
            stop_reason="tool_calls" if tool_calls else "stop",
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
