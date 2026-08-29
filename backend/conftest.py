"""Stand-ins for the Ducky host packages so this plugin's tests run standalone.

This package is loaded by the app as ``backend``, and the app's own code lives
under ``backend.agent.*`` — a namespace that only exists inside the running app.
These stubs mirror just the shapes the plugin actually touches, so provider and
adapter modules can be imported and exercised without the app.
"""

from __future__ import annotations

import enum
import sys
import types as _pytypes
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderMessage:
    role: str
    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_call_id: str = ""
    attachments: list[Any] = field(default_factory=list)
    thinking: str = ""
    thinking_blocks: list[dict[str, Any]] = field(default_factory=list)


class StreamEventKind(enum.Enum):
    TEXT_DELTA = "text_delta"
    THINKING = "thinking"
    TOOL_CALLS = "tool_calls"
    DONE = "done"


@dataclass
class StreamEvent:
    kind: StreamEventKind
    text: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class PromptCachePayload:
    system: str = ""


@dataclass
class ModelInfo:
    id: str
    display_name: str = ""
    supports_vision: bool = False
    supports_tools: bool = False
    context_limit: int | None = None
    price_in: float | None = None
    price_out: float | None = None
    price_cached_in: float | None = None


@dataclass
class CodingAgentCapabilities:
    terminal_agent: bool = False
    chat_api: bool = False
    a2a: bool = False
    mcp_inject: bool = False
    needs_api_key: bool = False
    needs_cli: bool = False
    resume: bool = False


@dataclass
class CodingAgentInfo:
    id: str
    label: str
    enabled: bool = False
    available: bool = False
    status: str = ""
    cli_path: str = ""
    default_args: str = ""
    capabilities: Any = None
    models: list[dict[str, str]] = field(default_factory=list)


@dataclass
class CodingAgentLaunchResult:
    ok: bool = True
    error: str = ""
    status: str = ""
    reply: str = ""


def _package(name: str) -> _pytypes.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = _pytypes.ModuleType(name)
        mod.__path__ = []  # a package, so dotted submodule imports resolve
        sys.modules[name] = mod
    parent, _, leaf = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], leaf, mod)
    return mod


def _module(name: str, **attrs: Any) -> _pytypes.ModuleType:
    mod = _package(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _install_host_stubs() -> None:
    # `backend` itself is this plugin's real package — never replace it.
    for pkg in (
        "backend.agent",
        "backend.agent.providers",
        "backend.agent.coding_agents",
    ):
        _package(pkg)

    _module(
        "backend.agent.providers.base",
        ProviderMessage=ProviderMessage,
        StreamEvent=StreamEvent,
        StreamEventKind=StreamEventKind,
        ToolCallRequest=ToolCallRequest,
    )
    _module("backend.agent.providers.cache_utils", parse_gemini_usage=lambda meta: {})
    _module("backend.agent.prompt_cache", PromptCachePayload=PromptCachePayload)
    _module(
        "backend.agent.multimodal_content",
        build_gemini_user_parts=lambda text, attachments: [],
    )
    _module(
        "backend.agent.model_fetch",
        ModelInfo=ModelInfo,
        _PricingRow=tuple,
        _cache_put=lambda cache, key, value: cache.__setitem__(key, value),
        _float_from_record=lambda record, *keys: None,
        _merge_prices=lambda primary, fallback: fallback or primary,
    )
    _module("backend.agent.secrets", get_key=lambda name: "", has_key=lambda name: False)
    _module(
        "backend.agent.coding_agents.base",
        CodingAgentCapabilities=CodingAgentCapabilities,
        CodingAgentInfo=CodingAgentInfo,
        CodingAgentLaunchResult=CodingAgentLaunchResult,
        which_cli=lambda name, override="": override or "",
    )
    _module(
        "backend.agent.coding_agents.cli_shared",
        finalize_cli_turn=lambda **kw: CodingAgentLaunchResult(),
        truncate_tool_result=lambda text: text,
    )
    _module("backend.agent.coding_agents.proc_exec", run_streaming_process=lambda **kw: None)
    _module(
        "backend.agent.coding_agents.settings_helpers",
        coding_agent_cfg=lambda settings, agent_id: {},
    )


_install_host_stubs()
