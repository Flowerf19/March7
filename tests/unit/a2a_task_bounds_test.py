"""Task lifecycle/stream resource bounds (finding #15 + cancellation).

Covers: slow-subscriber eviction without stalling worker/fast peers, max
subscription denial (store + HTTP 429), multibyte byte accounting,
subscribed/idle TTL release with stream close, noncooperative cancel still
counting toward concurrency, bounded shutdown, oversized fail-explicit,
replay/auth no-regression, and SSE write timeout.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import pytest
from aiohttp.test_utils import TestClient, TestServer

from twin.shared.a2a.auth import make_a2a_headers
from twin.shared.a2a.client import A2AClient
from twin.shared.a2a.server import A2AServer
from twin.shared.a2a.tasks import (
    SubscriptionLimitExceeded,
    TaskStore,
    _message_size,
)
from twin.shared.a2a.types import A2AMessage, AgentCard, Part, TaskStatus
from twin.shared.a2a.wire import _write_bytes

SECRET = "test-a2a-shared-secret"


def _card(name: str) -> AgentCard:
    return AgentCard(name=name, description="", url="", version="1")


async def _echo_handler(params):
    yield A2AMessage(role="agent", parts=[Part(type="text", text="hi")])


def _evernight_server(**kwargs) -> A2AServer:
    kwargs.setdefault("agent_name", "evernight")
    kwargs.setdefault("shared_secret", SECRET)
    return A2AServer(
        agent_card=_card("Evernight"),
        skill_handlers={"chat": _echo_handler},
        **kwargs,
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


def _signed(actor: str, method: str, path: str, body: bytes) -> dict[str, str]:
    return make_a2a_headers(actor, method, path, body, secret=SECRET)


async def test_slow_no_reading_evicted_fast_gets_final_bounded():
    store = TaskStore(
        max_subscriber_messages=16,
        max_subscriber_bytes=8 * 1024,
        max_subscribers_per_task=4,
        max_task_buffer_bytes=512 * 1024,
        max_task_buffer_messages=5000,
    )
    assert store.create(
        task_id="t1", skill="chat", session_id="1", creator_peer="m"
    )
    slow = store.subscribe("t1")
    fast = store.subscribe("t1")

    async def publish_all():
        for i in range(2000):
            store.publish(
                "t1",
                A2AMessage(
                    role="agent", parts=[Part(type="text", text=f"m-{i}")]
                ),
            )
            if i % 5 == 0:
                await asyncio.sleep(0)
        store.broadcast("t1", None)

    async def read_fast():
        out = []
        while True:
            msg = await fast.get()
            if msg is None:
                break
            out.append(msg)
        return out

    pub_task = asyncio.create_task(publish_all())
    fast_task = asyncio.create_task(read_fast())
    await pub_task
    fast_msgs = await asyncio.wait_for(fast_task, timeout=5.0)

    # Worker finished all publishes without raising; fast got every message
    # including the final consolidation result.
    assert len(fast_msgs) == 2000
    assert fast_msgs[-1].parts[0].text == "m-1999"
    # Slow never read: evicted, drained, closed with None, memory bounded.
    assert store.subscriber_count("t1") == 1  # only fast remains pre-unsub
    assert slow.qsize() == 1
    assert getattr(slow, "buffered_bytes", 0) == 0
    assert slow.get_nowait() is None
    # Replay buffer still retains the final for re-subscribe (not dropped).
    snap = store.snapshot("t1")
    assert snap and snap[-1].parts[0].text == "m-1999"
    store.unsubscribe("t1", fast)
    assert store.subscriber_count("t1") == 0


async def test_max_subscriptions_denied_store_and_http():
    store = TaskStore(max_subscribers_per_task=2)
    store.create(task_id="s1", skill="chat", session_id="1", creator_peer="march7")
    q1 = store.subscribe("s1")
    q2 = store.subscribe("s1")
    with pytest.raises(SubscriptionLimitExceeded):
        store.subscribe("s1")
    assert store.subscriber_count("s1") == 2
    # Existing streams unaffected by the denial.
    store.publish("s1", A2AMessage(role="agent", parts=[Part(type="text", text="x")]))
    assert q1.get_nowait().parts[0].text == "x"
    assert q2.get_nowait().parts[0].text == "x"

    server = _evernight_server(task_store=store)
    async with _test_client(server.build_app()) as client:
        headers = _signed("march7", "GET", "/tasks/s1/stream", b"")
        async with client.get("/tasks/s1/stream", headers=headers) as resp:
            assert resp.status == 429
    store.unsubscribe("s1", q1)
    store.unsubscribe("s1", q2)


def test_message_size_counts_utf8_bytes():
    ascii_msg = A2AMessage(role="agent", parts=[Part(type="text", text="e" * 100)])
    emoji_msg = A2AMessage(role="agent", parts=[Part(type="text", text="😀" * 100)])
    # 100 x U+1F600 = 400B vs 100B ascii -> 300B delta, not 0 chars delta.
    assert _message_size(emoji_msg) - _message_size(ascii_msg) == 300

    store = TaskStore(max_task_buffer_bytes=500, max_task_buffer_messages=1000)
    store.create(task_id="b1", skill="chat", session_id="1", creator_peer="m")
    for _ in range(10):
        store.publish(
            "b1",
            A2AMessage(role="agent", parts=[Part(type="text", text="😀" * 100)]),
        )
    buffered = store.snapshot("b1")
    # Each ~469B; byte budget holds 1, char counting would hold 2-3.
    assert len(buffered) == 1
    total = sum(_message_size(m) for m in buffered)
    assert total <= 500


async def test_subscriber_byte_budget_evicts_on_bytes_not_count():
    store = TaskStore(
        max_subscriber_messages=1000,
        max_subscriber_bytes=500,
        max_subscribers_per_task=4,
    )
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    slow = store.subscribe("t1")
    store.publish(
        "t1", A2AMessage(role="agent", parts=[Part(type="text", text="😀" * 100)]),
    )
    assert store.subscriber_count("t1") == 1
    store.publish(
        "t1", A2AMessage(role="agent", parts=[Part(type="text", text="😀" * 100)]),
    )
    # 2 x ~469B > 500B -> evicted even though count 2 << 1000.
    assert store.subscriber_count("t1") == 0
    assert slow.get_nowait() is None
    assert len(store.snapshot("t1")) == 2  # replay still holds both


def test_completed_subscribed_ttl_closes_and_releases_data():
    now = [2000.0]
    store = TaskStore(completed_ttl_seconds=60.0, clock=lambda: now[0])
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    store.publish("t1", A2AMessage(role="agent", parts=[Part(type="text", text="priv")]))
    store.complete("t1")
    queue = store.subscribe("t1")
    assert store.snapshot("t1")
    now[0] += 61.0
    assert store.purge_expired() == 1
    assert store.get("t1") is None
    assert store.snapshot("t1") == []
    assert queue.get_nowait() is None
    assert store.subscriber_count("t1") == 0


async def test_background_purge_releases_idle_and_shutdown_stops_it():
    store = TaskStore(completed_ttl_seconds=0.05)
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    store.complete("t1")
    store.start_background_purge(interval=0.02)
    try:
        for _ in range(100):
            if store.get("t1") is None:
                break
            await asyncio.sleep(0.02)
        assert store.get("t1") is None
    finally:
        await store.shutdown(timeout=0.2)
    assert store._purge_task is None or store._purge_task.done()


async def test_noncooperative_cancel_still_counts_blocks_spawn():
    async def bad():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(30)

    store = TaskStore(max_active_tasks=1)
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    worker = asyncio.create_task(bad())
    store.set_worker("t1", worker)
    await asyncio.sleep(0.02)
    start = time.monotonic()
    task = await store.cancel_and_wait("t1", timeout=0.1)
    elapsed = time.monotonic() - start
    try:
        assert elapsed < 2.0
        assert task is not None and task.status == TaskStatus.CANCELLED
        assert not worker.done()  # swallowed cancel, still running
        assert store.active_count() == 1
        assert store.create(
            task_id="t2", skill="chat", session_id="1", creator_peer="m"
        ) is None
        # Record/handle preserved, not evicted while worker lives.
        assert store.purge_expired() == 0
        assert store.get("t1") is not None
        assert store._records["t1"].worker is worker
    finally:
        worker.cancel()
        await asyncio.wait([worker], timeout=1.0)
    assert worker.done()
    assert store.active_count() == 0
    assert store.create(
        task_id="t2", skill="chat", session_id="1", creator_peer="m"
    ) is not None


async def test_shutdown_returns_bounded_with_swallowing_worker():
    async def bad():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(30)

    store = TaskStore()
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    worker = asyncio.create_task(bad())
    store.set_worker("t1", worker)
    await asyncio.sleep(0.02)
    store.start_background_purge(interval=0.05)
    start = time.monotonic()
    await asyncio.wait_for(store.shutdown(timeout=0.2), timeout=2.0)
    elapsed = time.monotonic() - start
    try:
        assert elapsed < 2.0
        assert store.get("t1") is not None
        assert store.get("t1").status == TaskStatus.CANCELLED
        assert store._purge_task is None or store._purge_task.done()
    finally:
        worker.cancel()
        await asyncio.wait([worker], timeout=1.0)


async def test_oversized_single_output_fails_explicitly():
    store = TaskStore(max_task_buffer_bytes=500)
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    queue = store.subscribe("t1")
    store.publish(
        "t1",
        A2AMessage(role="agent", parts=[Part(type="text", text="x" * 10000)]),
    )
    assert store.get("t1").status == TaskStatus.FAILED
    assert store.snapshot("t1") == []
    assert queue.get_nowait() is None
    assert store.subscriber_count("t1") == 0

    async def big_handler(params):
        yield A2AMessage(role="agent", parts=[Part(type="text", text="y" * 10000)])

    server = _evernight_server(task_store=store)
    store.create(task_id="t2", skill="chat", session_id="1", creator_peer="m")
    await server._execute_handler("t2", big_handler, {})
    assert store.get("t2").status == TaskStatus.FAILED


async def test_replay_exactly_once_and_client_roundtrip_no_regression():
    store = TaskStore()
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    m1 = A2AMessage(role="agent", parts=[Part(type="text", text="a")])
    m2 = A2AMessage(role="agent", parts=[Part(type="text", text="b")])
    store.publish("t1", m1)
    store.publish("t1", m2)
    assert [m.parts[0].text for m in store.snapshot("t1")] == ["a", "b"]
    queue = store.subscribe("t1")
    m3 = A2AMessage(role="agent", parts=[Part(type="text", text="c")])
    store.publish("t1", m3)
    assert queue.get_nowait() is m3  # only new, no replay duplication
    assert [m.parts[0].text for m in store.snapshot("t1")] == ["a", "b", "c"]
    store.unsubscribe("t1", queue)

    async with _test_client(_evernight_server().build_app()) as raw:
        base = str(raw.make_url("/")).rstrip("/")
        client = A2AClient(base_url=base, actor="march7", secret=SECRET)
        try:
            assert await client.send_text_task(
                skill="chat", session_id="1", text="hello"
            ) == "hi"
        finally:
            await client.close()


async def test_sse_write_timeout_returns_bounded():
    class SlowResp:
        async def write(self, data: bytes):
            await asyncio.sleep(10)

    class FastResp:
        async def write(self, data: bytes):
            return None

    await _write_bytes(FastResp(), b"x", timeout=0.2)  # type: ignore[arg-type]
    start = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await _write_bytes(SlowResp(), b"x", timeout=0.05)  # type: ignore[arg-type]
    assert time.monotonic() - start < 2.0
