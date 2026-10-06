"""Single-conversation runner: messages → adapter → ExecutionTrace."""

from __future__ import annotations

import uuid
import queue
import threading

from mutiny_core.adapter.port import TargetAdapter, ToolsNotObservableError
from mutiny_core.trace.models import ExecutionTrace, TraceTurn


def execute_conversation(
    adapter: TargetAdapter,
    messages: list[str],
    *,
    candidate_id: str,
    session_id: str | None = None,
    step_timeout_seconds: float = 60.0,
) -> ExecutionTrace:
    """Reset adapter, step each user message, aggregate an ExecutionTrace.

    Pure orchestration over the ``TargetAdapter`` port. No persistence, no LLM.
    """
    sid = session_id or str(uuid.uuid4())
    trace = ExecutionTrace(
        candidate_id=candidate_id,
        session_id=sid,
        status="executing",
    )

    try:
        adapter.reset(sid)
        for user_message in messages:
            try:
                result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

                def run_step() -> None:
                    try:
                        result_queue.put((True, adapter.step(sid, user_message)))
                    except BaseException as exc:  # propagate adapter exceptions
                        result_queue.put((False, exc))

                threading.Thread(target=run_step, daemon=True).start()
                try:
                    succeeded, result = result_queue.get(timeout=step_timeout_seconds)
                except queue.Empty:
                    trace.status = "error"
                    trace.error = f"adapter_step_timeout: exceeded {step_timeout_seconds:g}s"
                    return trace
                if not succeeded:
                    if isinstance(result, ToolsNotObservableError):
                        raise result
                    if isinstance(result, BaseException):
                        raise result
                    raise RuntimeError("adapter step failed")
            except ToolsNotObservableError:
                raise
            turn = TraceTurn(
                user_message=user_message,
                assistant_message=result.assistant_message,
                tool_calls=list(result.tool_calls),
                tool_results=list(result.tool_results),
                raw=dict(result.raw) if result.raw else None,
            )
            trace.turns.append(turn)
            trace.all_tool_calls.extend(turn.tool_calls)
        trace.status = "scored"
        return trace
    except ToolsNotObservableError as exc:
        trace.status = "error"
        trace.error = str(exc)
        raise
