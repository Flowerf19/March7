from contextlib import contextmanager

import pytest

from twin.shared.a2a.server import A2AServer
from twin.shared.a2a.types import A2AMessage, AgentCard, Part, TaskStatus


@pytest.mark.asyncio
async def test_execute_handler_runs_inside_langsmith_parent_context(monkeypatch):
    seen_parent = []

    @contextmanager
    def fake_tracing_context(parent):
        seen_parent.append(parent)
        yield

    monkeypatch.setattr(
        "twin.shared.a2a.server.tracing_context_from_parent",
        fake_tracing_context,
    )

    async def handler(params):
        yield A2AMessage(role="agent", parts=[Part(type="text", text=params["text"])])

    server = A2AServer(
        agent_card=AgentCard(name="test", description="", url="", version="1"),
        skill_handlers={"chat": handler},
    )
    task = server._store.create(
        task_id="t1", session_id="s1", skill="chat", creator_peer="march7"
    )
    assert task is not None
    parent = {"langsmith-trace": "trace-id", "baggage": "langsmith-project=march7-bot"}

    await server._execute_handler("t1", handler, {"text": "done"}, parent)

    assert seen_parent == [parent]
    assert task.status == TaskStatus.COMPLETED
    assert server._store.snapshot("t1")[0].parts[0].text == "done"
