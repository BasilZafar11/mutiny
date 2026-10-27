"""Observe local PydanticAI function-tool executions through SDK events."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.messages import (
    NativeToolCallPart,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    OutputToolCallEvent,
    ToolCallPart,
)
from pydantic_ai.usage import UsageLimits
from pydantic_core import to_jsonable_python

from mutiny_core.adapter import TargetAdapter, ToolsNotObservableError
from mutiny_core.trace import AdapterTurnResult, ToolCall


class PydanticAIAdapter(TargetAdapter):
    """Run an Agent locally, retaining separate message history per session.

    Only validated function-tool calls are evidence. Output tools, rejected
    arguments, and skipped calls are not executed function tools. Provider-side
    built-in tools are unsupported and fail closed rather than hiding evidence.
    """

    def __init__(
        self,
        agent: Agent[Any, Any],
        *,
        deps: Any = None,
        context: dict[str, Any] | None = None,
        context_provider: Callable[[], dict[str, Any]] | None = None,
        on_reset: Callable[[], None] | None = None,
        request_limit: int = 10,
    ) -> None:
        if not isinstance(agent, Agent):
            raise TypeError("PydanticAIAdapter requires a PydanticAI Agent")
        if request_limit < 1:
            raise ValueError("request_limit must be positive")
        self._agent = agent
        self._deps = deps
        self._context = copy.deepcopy(context or {})
        self._context_provider = context_provider
        self._on_reset = on_reset
        self._request_limit = request_limit
        self._sessions: dict[str, list[ModelMessage]] = {}

    def reset(self, session_id: str) -> None:
        if self._on_reset is not None:
            self._on_reset()
        self._sessions[session_id] = []

    def step(self, session_id: str, user_message: str) -> AdapterTurnResult:
        if session_id not in self._sessions:
            self.reset(session_id)
        calls: list[ToolCall] = []
        outputs: list[dict[str, Any]] = []
        observed_call_ids: set[str] = set()
        observed = False

        async def observe(_context: Any, events: Any) -> None:
            nonlocal observed
            observed = True
            async for event in events:
                if isinstance(event, (FunctionToolCallEvent, OutputToolCallEvent)):
                    observed_call_ids.add(event.tool_call_id)
                if (
                    isinstance(event, FunctionToolCallEvent)
                    and event.args_valid is True
                ):
                    part = event.part
                    if not part.tool_call_id or not part.tool_name:
                        raise ToolsNotObservableError(
                            "PydanticAI tool identity is missing"
                        )
                    calls.append(
                        ToolCall(
                            id=part.tool_call_id,
                            name=part.tool_name,
                            arguments=part.args_as_dict(),
                        )
                    )
                elif isinstance(event, FunctionToolResultEvent):
                    # Skipped/invalid calls have results too; pair only actual calls.
                    if any(call.id == event.tool_call_id for call in calls):
                        outputs.append(
                            {
                                "call_id": event.tool_call_id,
                                "output": to_jsonable_python(event.part.content),
                            }
                        )

        result = self._agent.run_sync(
            user_message,
            message_history=copy.deepcopy(self._sessions[session_id]),
            deps=self._deps,
            usage_limits=UsageLimits(request_limit=self._request_limit),
            event_stream_handler=observe,
        )
        if not observed or not callable(getattr(result, "new_messages", None)):
            raise ToolsNotObservableError(
                "PydanticAI run did not expose tool observation"
            )
        messages = result.new_messages()
        if isinstance(result.output, DeferredToolRequests):
            raise ToolsNotObservableError(
                "PydanticAI deferred tools require external execution"
            )
        if any(
            isinstance(part, NativeToolCallPart)
            for msg in messages
            for part in msg.parts
        ):
            raise ToolsNotObservableError(
                "PydanticAI provider-side built-in tools are unsupported"
            )
        if any(
            isinstance(part, ToolCallPart)
            and part.tool_call_id not in observed_call_ids
            for msg in messages
            for part in msg.parts
        ):
            raise ToolsNotObservableError(
                "PydanticAI tool call is missing its observation event"
            )
        if any(not any(out["call_id"] == call.id for out in outputs) for call in calls):
            raise ToolsNotObservableError(
                "PydanticAI tool call has no observable result"
            )
        turn = AdapterTurnResult(
            assistant_message=str(result.output) if result.output is not None else None,
            tool_calls=calls,
            tool_results=outputs,
            raw={"adapter": "pydantic_ai", "session_id": session_id},
        )
        self._sessions[session_id].extend(copy.deepcopy(messages))
        return turn

    def context(self, session_id: str | None = None) -> dict[str, Any]:
        _ = session_id
        facts = self._context_provider() if self._context_provider else self._context
        return copy.deepcopy(facts)
