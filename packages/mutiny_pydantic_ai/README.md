# Mutiny PydanticAI adapter (unreleased source package)

`PydanticAIAdapter` implements Mutiny's `TargetAdapter` for **local** PydanticAI
agents. Core is framework-independent. Hosted remains observe-only: customer
agents run on your machine; `mutiny run --hosted` uploads redacted lineage.

## Install from this checkout

```bash
uv sync --group dev
```

For an existing environment, install the local Core and adapter packages:

```bash
pip install -e packages/mutiny_core -e packages/mutiny_pydantic_ai
```

This package uses `pydantic-ai-slim>=2.61.0,<3` (tested with 2.61.0). Install your
chosen PydanticAI provider extra separately for live models. The offline example
below needs no provider, credentials, or network access. This package is not yet
advertised as available on PyPI.

## Wire a local project

Create `.mutiny/adapter.py` in the project containing your PydanticAI `Agent`:

```python
from agent import support_agent
from mutiny_pydantic_ai import PydanticAIAdapter

def create_adapter():
    return PydanticAIAdapter(
        support_agent,
        context={"customer": {"email": "alice@example.com"}},
    )
```

Run `mutiny init` if needed, then edit `policy.yaml` for the actual tools your
agent exposes. Keep the custom adapter factory when scaffolding. Run
`mutiny run`, minimize findings through the existing local campaign flow, and
replay saved findings with `mutiny test`. No CLI target enum or Hosted execution
path is added; the CLI already loads any local `TargetAdapter` factory.

## Offline tool observation

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from mutiny_pydantic_ai import PydanticAIAdapter

agent = Agent(TestModel())

@agent.tool_plain
def issue_refund(amount: int) -> dict:
    return {"refunded": amount}  # Mock only: no real refund.

adapter = PydanticAIAdapter(agent)
turn = adapter.step("demo", "Please issue a refund")
assert turn.tool_calls[0].name == "issue_refund"
assert turn.tool_results[0]["call_id"] == turn.tool_calls[0].id
```

The adapter consumes PydanticAI's public function-tool events. Only validated
calls that execute are mapped to tool evidence; output-schema tools, argument
validation failures, and skipped calls do not become function-tool invocations.
No tool events on a text-only turn is valid. Missing observation surfaces,
unpaired executed calls, or provider-side built-in tool calls raise
`ToolsNotObservableError`. Tool evidence is never inferred from assistant text.

History is isolated per session. `reset(session_id)` clears that conversation;
use `on_reset` to reset mock tools or other external state. `deps` is passed to
the agent separately from the deterministic policy facts returned by `context`.
`context_provider` can supply fresh policy facts; returned facts are copied.
`request_limit` bounds model requests per turn (default 10). SDK failures
propagate to the existing candidate error handling. The SDK runs synchronously
inside the local runner; share neither an adapter nor stateful tools between
concurrent campaigns.

Public SDK references: [agent execution](https://pydantic.dev/docs/ai/api/pydantic-ai/agent/),
[tool events](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/), and
[offline testing](https://pydantic.dev/docs/ai/guides/testing/).
