"""A2A fail-closed stream contract: truncation and unknown status never succeed.

Wire contract (controlled):
- Normal output: `data: {message}\\n\\n` (unchanged shape).
- Truncation: single `event: error\\ndata: {"error":"stream_truncated",...}\\n\\n`
  for evicted slow queues (`reason=evicted`) or rolled history
  (`reason=history_truncated`). Client raises A2AStreamError.
- Final status: only COMPLETED succeeds; None/IN_PROGRESS/PENDING/FAILED/
  CANCELLED raise. FAILED/CANCELLED messages preserved.

All streaming tests use real A2AClient over local TestServer (no outside net).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from twin.shared.a2a.auth import make_a2a_headers
from twin.shared.a2a.client import A2AClient, A2AStreamError
from twin.shared.a2a.server import A2AServer
from twin.shared.a2a.tasks import TaskStore, _message_size, _SubscriberQueue
from twin.shared.a2a.types import A2AMessage, A2ATask, AgentCard, Part, TaskStatus

SECRET = "test-a2a-shared-secret"


def _card() -> AgentCard:
    return AgentCard(name="Evernight", description="", url="", version="1")


def _msg(text: str) -> A2AMessage:
    return A2AMessage(role="agent", parts=[Part(type="text", text=text)])


async def _echo_handler(params):
    yield A2AMessage(role="agent", parts=[Part(type="text", text="hi")])


async def _data_handler(params):
    yield A2AMessage(role="agent", parts=[Part(type="data", data={"ok": True})])


async def _fail_handler(params):
    yield _msg("before-fail")
    raise RuntimeError("boom-handler")
    yield _msg("unreached")  # pragma: no cover


def _server(store: TaskStore | None = None, handlers: dict | None = None) -> A2AServer:
    return A2AServer(
        agent_card=_card(),
        skill_handlers=handlers or {"chat": _echo_handler, "consolidate": _data_handler},
        agent_name="evernight",
        shared_secret=SECRET,
        task_store=store or TaskStore(),
    )


@asynccontextmanager
async def _test_client(app):
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


def _base(raw: TestClient) -> str:
    return str(raw.make_url("/")).rstrip("/")


async def _wait_for(cond, timeout: float = 2.0):
    loop = asyncio.get_event_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if cond():
            return True
        await asyncio.sleep(0.01)
    return cond()


class _StatusClient(A2AClient):
    """Real send_task_and_wait logic with controlled stream + status."""

    def __init__(self, messages: list[A2AMessage], status: A2ATask | None):
        super().__init__("http://a2a.test", actor="march7", secret="test-secret")
        self._messages = messages
        self._status = status

    async def send_task(self, params: dict) -> A2ATask:
        return A2ATask(id=params.get("id", "t1"), status=TaskStatus.IN_PROGRESS)

    async def subscribe_stream(self, task_id: str) -> AsyncIterator[A2AMessage]:
        for message in self._messages:
            yield message

    async def get_task(self, task_id: str):
        return self._status


async def test_eviction_overflow_raises_despite_completed_real_client():
    store = TaskStore(
        max_subscriber_messages=1,
        max_subscriber_bytes=10**6,
        max_task_buffer_messages=100,
        max_task_buffer_bytes=512 * 1024,
    )
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="march7")
    store.publish("t1", _msg("m0"))
    srv = _server(store)
    async with _test_client(srv.build_app()) as raw:
        client = A2AClient(base_url=_base(raw), actor="march7", secret=SECRET)
        try:
            collected: list[A2AMessage] = []

            async def collect():
                async for message in client.subscribe_stream("t1"):
                    collected.append(message)

            task = asyncio.create_task(collect())
            assert await _wait_for(lambda: store.subscriber_count("t1") == 1)
            # Synchronous burst: no await, drain cannot interleave -> evict.
            store.publish("t1", _msg("m1"))
            store.publish("t1", _msg("m2"))
            assert store.subscriber_count("t1") == 0
            store.complete("t1")
            store.broadcast("t1", None)
            assert store.get("t1").status == TaskStatus.COMPLETED
            with pytest.raises(A2AStreamError, match="truncated"):
                await asyncio.wait_for(task, timeout=5.0)
            # Partial replay arrived before the error; never counted success.
            assert [m.parts[0].text for m in collected] == ["m0"]
            assert store.subscriber_count("t1") == 0
            # Fast re-subscribe still sees the full retained buffer cleanly.
            ok: list[A2AMessage] = []
            async for message in client.subscribe_stream("t1"):
                ok.append(message)
            assert [m.parts[0].text for m in ok] == ["m0", "m1", "m2"]
        finally:
            await client.close()


async def test_rolled_snapshot_cannot_pass_as_complete_real_client():
    store = TaskStore(
        max_task_buffer_messages=2,
        max_task_buffer_bytes=64 * 1024,
        max_subscriber_messages=64,
        max_subscriber_bytes=64 * 1024,
    )
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="march7")
    for text in ("m0", "m1", "m2", "m3"):
        store.publish("t1", _msg(text))
    assert [m.parts[0].text for m in store.snapshot("t1")] == ["m2", "m3"]
    store.complete("t1")
    store.broadcast("t1", None)
    srv = _server(store)
    async with _test_client(srv.build_app()) as raw:
        client = A2AClient(base_url=_base(raw), actor="march7", secret=SECRET)
        try:
            with pytest.raises(A2AStreamError, match="truncated"):
                async for _ in client.subscribe_stream("t1"):
                    pass
            # send_task_and_wait on the same rolled task also raises.
            client.send_task = AsyncMock(  # type: ignore[method-assign]
                return_value=A2ATask(id="t1", status=TaskStatus.IN_PROGRESS)
            )
            with pytest.raises(A2AStreamError, match="truncated"):
                await client.send_task_and_wait({"id": "t1"})
        finally:
            await client.close()


async def test_unknown_and_noncompleted_status_raise_clean_succeeds():
    good = [A2AMessage(role="agent", parts=[Part(type="data", data={"ok": True})])]
    # None (auth/timeout on status fetch) must raise, never empty success.
    with pytest.raises(RuntimeError, match="status unknown"):
        await _StatusClient(good, None).send_task_and_wait({"id": "t1"})
    for status in (TaskStatus.IN_PROGRESS, TaskStatus.PENDING):
        with pytest.raises(RuntimeError, match="incomplete"):
            await _StatusClient(
                good, A2ATask(id="t1", status=status)
            ).send_task_and_wait({"id": "t1"})
    # FAILED/CANCELLED preserved.
    with pytest.raises(RuntimeError, match="boom"):
        await _StatusClient(
            [_msg("boom")], A2ATask(id="t1", status=TaskStatus.FAILED)
        ).send_text_task(skill="chat", session_id="1", text="hi")
    with pytest.raises(RuntimeError, match="cancelled"):
        await _StatusClient(
            good, A2ATask(id="t1", status=TaskStatus.CANCELLED)
        ).send_task_and_wait({"id": "t1"})
    # Clean COMPLETED returns all messages/data.
    out = await _StatusClient(
        good, A2ATask(id="t1", status=TaskStatus.COMPLETED)
    ).send_task_and_wait({"id": "t1"})
    assert out == good
    data = await _StatusClient(
        good, A2ATask(id="t1", status=TaskStatus.COMPLETED)
    ).send_data_task(skill="consolidate", session_id="1")
    assert data == {"ok": True}
    # Error statuses never return success empty data.
    with pytest.raises(RuntimeError):
        await _StatusClient([], None).send_data_task(skill="consolidate", session_id="1")


async def test_failed_handler_real_http_raises_no_success_data():
    store = TaskStore()
    srv = _server(store, {"chat": _fail_handler})
    async with _test_client(srv.build_app()) as raw:
        client = A2AClient(base_url=_base(raw), actor="march7", secret=SECRET)
        try:
            with pytest.raises(RuntimeError, match="boom-handler|before-fail|failed"):
                await client.send_text_task(skill="chat", session_id="1", text="hi")
        finally:
            await client.close()


async def test_clean_completed_real_http_returns_all_once_no_dup():
    store = TaskStore()
    srv = _server(store)
    async with _test_client(srv.build_app()) as raw:
        client = A2AClient(base_url=_base(raw), actor="march7", secret=SECRET)
        try:
            assert await client.send_text_task(
                skill="chat", session_id="1", text="hi"
            ) == "hi"
            assert await client.send_data_task(
                skill="consolidate", session_id="1"
            ) == {"ok": True}
        finally:
            await client.close()
    # Live fanout: each message exactly once, single close, no hang.
    live = TaskStore()
    live.create(task_id="t1", skill="chat", session_id="1", creator_peer="march7")
    live.publish("t1", _msg("a"))
    srv2 = _server(live)
    async with _test_client(srv2.build_app()) as raw:
        client = A2AClient(base_url=_base(raw), actor="march7", secret=SECRET)
        try:
            async def collect():
                out = []
                async for message in client.subscribe_stream("t1"):
                    out.append(message.parts[0].text)
                return out

            task = asyncio.create_task(collect())
            assert await _wait_for(lambda: live.subscriber_count("t1") == 1)
            live.publish("t1", _msg("b"))
            live.publish("t1", _msg("c"))
            live.complete("t1")
            live.broadcast("t1", None)
            assert await asyncio.wait_for(task, timeout=5.0) == ["a", "b", "c"]
            assert live.subscriber_count("t1") == 0
        finally:
            await client.close()


def test_utf8_accounting_close_marker_zero_and_evicted_reason():
    ascii_msg = A2AMessage(role="agent", parts=[Part(type="text", text="e" * 100)])
    emoji_msg = A2AMessage(role="agent", parts=[Part(type="text", text="😀" * 100)])
    assert _message_size(emoji_msg) - _message_size(ascii_msg) == 300
    queue: _SubscriberQueue = _SubscriberQueue(64, 64 * 1024)
    queue.put_nowait(emoji_msg)
    assert queue.buffered_bytes == _message_size(emoji_msg)
    queue.put_nowait(None)  # close marker carries no byte charge
    assert queue.buffered_bytes == _message_size(emoji_msg)
    assert queue.get_nowait() is emoji_msg
    assert queue.buffered_bytes == 0
    assert queue.get_nowait() is None
    assert queue.buffered_bytes == 0

    store = TaskStore(max_subscriber_messages=1, max_subscriber_bytes=10**6)
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    slow = store.subscribe("t1")
    assert isinstance(slow, _SubscriberQueue)
    store.publish("t1", _msg("m0"))
    store.publish("t1", _msg("m1"))
    store.publish("t1", _msg("m2"))  # overflow -> evict
    assert store.subscriber_count("t1") == 0
    assert slow.get_nowait() is None  # close signal preserved for API compat
    assert slow.buffered_bytes == 0
    assert getattr(slow, "close_reason", None) == "evicted"


async def test_stream_cap_429_and_disconnect_release_no_hang():
    store = TaskStore(max_subscribers_per_task=16)
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="march7")
    store.publish("t1", _msg("seed"))
    srv = _server(store)
    async with _test_client(srv.build_app()) as raw:
        opened = []
        for _ in range(16):
            headers = make_a2a_headers(
                "march7", "GET", "/tasks/t1/stream", b"", secret=SECRET
            )
            resp = await raw.get("/tasks/t1/stream", headers=headers)
            assert resp.status == 200
            opened.append(resp)
        assert await _wait_for(lambda: store.subscriber_count("t1") == 16)
        headers = make_a2a_headers(
            "march7", "GET", "/tasks/t1/stream", b"", secret=SECRET
        )
        async with raw.get("/tasks/t1/stream", headers=headers) as denied:
            assert denied.status == 429
        assert store.subscriber_count("t1") == 16
        # Disconnect one slow reader: server releases its slot, rest unaffected.
        await opened.pop(0).release()
        assert await _wait_for(lambda: store.subscriber_count("t1") == 15)
        store.complete("t1")
        store.broadcast("t1", None)
        for resp in opened:
            body = await asyncio.wait_for(resp.read(), timeout=5.0)
            text = body.decode("utf-8")
            assert text.count('"seed"') == 1  # exactly once, no duplicate final
            assert "stream_truncated" not in text  # clean close, no error
            resp.close()
        assert await _wait_for(lambda: store.subscriber_count("t1") == 0)
