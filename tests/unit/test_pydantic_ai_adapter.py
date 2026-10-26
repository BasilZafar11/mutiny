"""Offline PydanticAI evidence and session contracts; no provider credentials."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, models
from pydantic_ai.models.test import TestModel
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    NativeToolCallPart,
    ToolCallPart,
    ToolReturnPart,
)

from mutiny_core import ToolsNotObservableError
from mutiny_core.adapter import execute_conversation
from mutiny_pydantic_ai import PydanticAIAdapter


@pytest.fixture(autouse=True)
def no_live_models(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def make_agent():
    calls = []
    agent = Agent(TestModel())

    @agent.tool_plain
    def issue_refund(amount: int, approved: bool) -> dict:
        calls.append({"amount": amount, "approved": approved})
        return {"refunded": amount, "approved": approved}

    return agent, calls


def test_real_tool_execution_matches_observed_arguments_and_result():
    agent, executed = make_agent()
    adapter = PydanticAIAdapter(agent)
    turn = adapter.step("s", "Refund please")
    assert len(executed) == len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "issue_refund"
    assert call.arguments == executed[0]
    assert turn.tool_results == [
        {
            "call_id": call.id,
            "output": {
                "refunded": executed[0]["amount"],
                "approved": executed[0]["approved"],
            },
        }
    ]
    assert turn.assistant_message


def test_history_is_per_session_and_reset_clears_it(monkeypatch):
    agent, executed = make_agent()
    histories = []
    resets = []
    original = Agent.run_sync

    def record(self, *args, **kwargs):
        histories.append(list(kwargs["message_history"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "run_sync", record)
    adapter = PydanticAIAdapter(agent, on_reset=lambda: resets.append(True))
    first = adapter.step("a", "First")
    before = len(executed)
    second = adapter.step("a", "Second")
    assert len(second.tool_calls) == len(executed) - before
    adapter.step("b", "Other conversation")
    adapter.reset("a")
    adapter.step("a", "Fresh conversation")
    assert histories[0] == histories[2] == histories[3] == []
    assert histories[1]
    assert len(resets) == 3
    assert first.tool_calls


def test_text_only_turn_has_no_invented_tool_evidence():
    adapter = PydanticAIAdapter(Agent(TestModel(custom_output_text="Hello")))
    turn = adapter.step("text", "Hello")
    assert turn.assistant_message == "Hello"
    assert turn.tool_calls == turn.tool_results == []


def test_structured_output_tool_is_not_a_function_tool():
    class Answer(BaseModel):
        answer: int

    adapter = PydanticAIAdapter(Agent(TestModel(), output_type=Answer))
    turn = adapter.step("structured", "Answer please")
    assert turn.assistant_message
    assert turn.tool_calls == turn.tool_results == []


def test_unobservable_run_fails_closed(monkeypatch):
    agent, _ = make_agent()
    monkeypatch.setattr(
        Agent,
        "run_sync",
        lambda *a, **k: SimpleNamespace(
            output="I refunded you",
            new_messages=lambda: [],
        ),
    )
    with pytest.raises(ToolsNotObservableError, match="observation"):
        PydanticAIAdapter(agent).step("blind", "Refund")


def test_context_is_copied_and_separate_from_agent_deps():
    facts = {"customer": {"email": "alice@example.com"}}
    adapter = PydanticAIAdapter(Agent(TestModel()), context_provider=lambda: facts)
    returned = adapter.context("s")
    returned["customer"]["email"] = "changed"
    assert adapter.context()["customer"]["email"] == "alice@example.com"
    facts["customer"]["email"] = "new@example.com"
    assert adapter.context()["customer"]["email"] == "new@example.com"


def test_agent_dependencies_are_passed_to_tools():
    from pydantic_ai import RunContext

    agent = Agent(TestModel(), deps_type=dict)

    @agent.tool
    def customer_name(ctx: RunContext[dict]) -> str:
        return ctx.deps["name"]

    adapter = PydanticAIAdapter(agent, deps={"name": "Alice"}, context={"limit": 200})
    assert adapter.step("s", "Who am I?").tool_results[0]["output"] == "Alice"
    assert adapter.context() == {"limit": 200}


def test_existing_local_cli_loader_accepts_factory(tmp_path):
    from mutiny_openai_agents.loader import load_adapter_factory

    (tmp_path / ".mutiny").mkdir()
    (tmp_path / ".mutiny" / "adapter.py").write_text(
        "from pydantic_ai import Agent\n"
        "from pydantic_ai.models.test import TestModel\n"
        "from mutiny_pydantic_ai import PydanticAIAdapter\n"
        "def create_adapter():\n"
        "    return PydanticAIAdapter(Agent(TestModel(custom_output_text='local')) )\n",
        encoding="utf-8",
    )
    adapter = load_adapter_factory(tmp_path)()
    assert isinstance(adapter, PydanticAIAdapter)
    assert adapter.step("cli", "Hi").assistant_message == "local"


def test_existing_runner_can_use_adapter_in_worker_thread():
    agent, executed = make_agent()
    trace = execute_conversation(
        PydanticAIAdapter(agent),
        ["Refund", "Thanks"],
        candidate_id="c",
        session_id="s",
    )
    assert trace.status == "scored", trace.error
    assert len(trace.all_tool_calls) == len(executed)
    assert trace.all_tool_calls


def test_request_limit_is_enforced():
    agent, _ = make_agent()
    with pytest.raises(UsageLimitExceeded, match="request_limit"):
        PydanticAIAdapter(agent, request_limit=1).step("limit", "Refund")


@pytest.mark.parametrize("limit", [0, -1])
def test_request_limit_must_be_positive(limit):
    with pytest.raises(ValueError, match="positive"):
        PydanticAIAdapter(Agent(TestModel()), request_limit=limit)


def run_with_events(monkeypatch, events, messages=(), output="done"):
    async def stream():
        for event in events:
            yield event

    def run(self, *args, **kwargs):
        asyncio.run(kwargs["event_stream_handler"](None, stream()))
        return SimpleNamespace(output=output, new_messages=lambda: list(messages))

    monkeypatch.setattr(Agent, "run_sync", run)
    return PydanticAIAdapter(Agent(TestModel()))


@pytest.mark.parametrize("args_valid", [False, None])
def test_invalid_and_skipped_tool_calls_are_not_execution_evidence(
    monkeypatch, args_valid
):
    call = ToolCallPart("issue_refund", {"amount": 42}, tool_call_id="not-executed")
    adapter = run_with_events(
        monkeypatch,
        [
            FunctionToolCallEvent(call, args_valid=args_valid),
            FunctionToolResultEvent(
                ToolReturnPart("issue_refund", "skipped", tool_call_id="not-executed")
            ),
        ],
    )
    turn = adapter.step("s", "Refund")
    assert turn.tool_calls == turn.tool_results == []


def test_executed_tool_without_result_fails_closed(monkeypatch):
    adapter = run_with_events(
        monkeypatch,
        [
            FunctionToolCallEvent(
                ToolCallPart(
                    "issue_refund", {"amount": 42}, tool_call_id="missing-return"
                ),
                args_valid=True,
            )
        ],
    )
    with pytest.raises(ToolsNotObservableError, match="result"):
        adapter.step("s", "Refund")


def test_provider_side_tool_fails_closed(monkeypatch):
    adapter = run_with_events(
        monkeypatch,
        [],
        messages=[
            ModelResponse(
                parts=[
                    NativeToolCallPart("web_search", {}, tool_call_id="native"),
                ]
            )
        ],
    )
    with pytest.raises(ToolsNotObservableError, match="provider-side"):
        adapter.step("s", "Search")


def test_transcript_tool_with_missing_event_fails_closed(monkeypatch):
    adapter = run_with_events(
        monkeypatch,
        [],
        messages=[
            ModelResponse(
                parts=[
                    ToolCallPart("issue_refund", {"amount": 42}, tool_call_id="hidden"),
                ]
            )
        ],
    )
    with pytest.raises(ToolsNotObservableError, match="missing its observation"):
        adapter.step("s", "Refund")


def test_deferred_tool_execution_fails_closed(monkeypatch):
    from pydantic_ai import DeferredToolRequests

    adapter = run_with_events(monkeypatch, [], output=DeferredToolRequests())
    with pytest.raises(ToolsNotObservableError, match="deferred"):
        adapter.step("s", "Refund")


def test_campaign_minimize_and_regression_work_without_core_changes():
    from mutiny_core import CampaignConfig, CampaignEngine, PolicySet
    from mutiny_core.minimize import minimize_genome
    from mutiny_core.regress import replay_regression, save_regression

    agent, _ = make_agent()
    adapter = PydanticAIAdapter(agent)
    policy = PolicySet.model_validate(
        {
            "version": "1",
            "target": "pydantic_ai",
            "rules": [
                {
                    "id": "no_refund",
                    "tool": "issue_refund",
                    "kind": "deny_tool",
                    "description": "No refunds",
                }
            ],
        }
    )
    result = CampaignEngine(
        adapter=adapter,
        policy_set=policy,
        config=CampaignConfig(population_size=2, elite_count=1, max_generations=1),
    ).run()
    assert result.violated
    assert result.best is not None
    minimized = minimize_genome(
        result.best.genome,
        adapter=adapter,
        policy_set=policy,
        target_rule_ids=["no_refund"],
        campaign_id="offline",
        candidate_id=result.best.genome.id,
    )
    assert minimized.still_reproduces
    regression = save_regression(
        minimized, name="refund", target="pydantic_ai", policy_set=policy
    )
    assert (
        replay_regression(regression, adapter=adapter, policy_set=policy).status
        == "FAIL"
    )
    safe = PydanticAIAdapter(Agent(TestModel(custom_output_text="Refund refused")))
    assert (
        replay_regression(regression, adapter=safe, policy_set=policy).status == "PASS"
    )
