from typing import AsyncIterator

import pytest

from twin.shared.a2a.client import A2AClient
from twin.shared.a2a.types import A2AMessage, A2ATask, Part, TaskStatus


class FakeA2AClient(A2AClient):
    def __init__(self, messages: list[A2AMessage], status: TaskStatus = TaskStatus.COMPLETED):
        super().__init__("http://a2a.test", actor="march7", secret="test-secret")
        self.messages = messages
        self.status = status
        self.sent_params = []

    async def send_task(self, params: dict) -> A2ATask:
        self.sent_params.append(params)
        return A2ATask(
            id=params["id"],
            session_id=params.get("sessionId"),
            skill=params.get("skill"),
            status=TaskStatus.IN_PROGRESS,
        )

    async def get_task(self, task_id: str) -> A2ATask:
        return A2ATask(id=task_id, status=self.status)

    async def subscribe_stream(self, task_id: str) -> AsyncIterator[A2AMessage]:
        for message in self.messages:
            yield message


@pytest.mark.asyncio
async def test_send_text_task_waits_for_stream_result():
    client = FakeA2AClient([
        A2AMessage(role="agent", parts=[Part(type="text", text="...")]),
        A2AMessage(role="agent", parts=[Part(type="text", text="done")]),
    ])

    result = await client.send_text_task(
        skill="chat",
        session_id="u1",
        text="hello",
    )

    assert result == "done"


@pytest.mark.asyncio
async def test_send_data_task_returns_last_data_part():
    client = FakeA2AClient([
        A2AMessage(role="agent", parts=[Part(type="data", data={"snapshot": [{"role": "user"}]})]),
    ])

    data = await client.send_data_task(skill="get_snapshot", session_id="u1")

    assert data == {"snapshot": [{"role": "user"}]}


@pytest.mark.asyncio
async def test_send_data_task_moves_langsmith_parent_without_mutating_payload():
    client = FakeA2AClient([
        A2AMessage(role="agent", parts=[Part(type="data", data={"status": "ok"})]),
    ])
    parent = {"langsmith-trace": "trace-id"}
    payload = {"scope": "user", "_langsmith_parent": parent}

    data = await client.send_data_task(
        skill="consolidate_discussion",
        session_id="u1",
        params={"payload": payload},
    )

    assert data == {"status": "ok"}
    assert payload["_langsmith_parent"] == parent
    assert client.sent_params[0]["_langsmith_parent"] == parent
    assert "_langsmith_parent" not in client.sent_params[0]["payload"]


@pytest.mark.asyncio
async def test_send_task_and_wait_raises_on_failed_task():
    client = FakeA2AClient(
        [A2AMessage(role="agent", parts=[Part(type="text", text="boom")])],
        status=TaskStatus.FAILED,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await client.send_text_task(skill="chat", session_id="u1", text="hello")
