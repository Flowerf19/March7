"""A2A stream follow-up regressions: accounting, cap, ring, FAILED+live."""
from __future__ import annotations

import asyncio
import json
import time

import pytest
from aiohttp import web

from twin.shared.a2a import wire as wire_mod
from twin.shared.a2a.server import A2AServer
from twin.shared.a2a.tasks import (
    TaskStore,
    _message_size,
    _SubscriberQueue,
)
from twin.shared.a2a.types import A2AMessage, AgentCard, Part, TaskStatus
from twin.shared.a2a.wire import serve_task_stream


def _msg(text: str) -> A2AMessage:
    return A2AMessage(role="agent", parts=[Part(type="text", text=text)])


def _queued_bytes(queue: _SubscriberQueue) -> int:
    return sum(
        _message_size(m) for m in list(queue._queue) if m is not None  # type: ignore[attr-defined]
    )


def _sse_texts(written: list[bytes]) -> list[str]:
    out = []
    for chunk in written:
        line = chunk.decode("utf-8")
        if line.startswith("event: error"):
            continue  # fail-closed truncation marker; see _sse_errors
        if line.startswith("event: complete"):
            continue  # completion marker; see _sse_completes
        assert line.startswith("data: ")
        payload = json.loads(line[len("data: "):])
        parts = payload.get("parts", [])
        out.append(parts[0].get("text", "") if parts else "")
    return out


def _sse_completes(written: list[bytes]) -> list[dict]:
    out = []
    for chunk in written:
        line = chunk.decode("utf-8")
        if not line.startswith("event: complete"):
            continue
        for sub in line.split("\n"):
            if sub.startswith("data:"):
                out.append(json.loads(sub[5:].lstrip()))
    return out


def _sse_errors(written: list[bytes]) -> list[dict]:
    out = []
    for chunk in written:
        line = chunk.decode("utf-8")
        if not line.startswith("event: error"):
            continue
        for sub in line.split("\n"):
            if sub.startswith("data:"):
                raw = sub[5:].lstrip()
                out.append(json.loads(raw))
    return out


async def test_mixed_reads_exact_accounting_and_backlog_bound():
    queue: _SubscriberQueue = _SubscriberQueue(64, 64 * 1024)
    msgs = [_msg(f"m-{i}-" + "x" * (i * 10)) for i in range(6)]
    for message in msgs:
        queue.put_nowait(message)
    assert queue.buffered_bytes == sum(_message_size(m) for m in msgs)

    first = await queue.get()
    assert first is msgs[0]
    assert queue.buffered_bytes == _queued_bytes(queue)
    second = queue.get_nowait()
    assert second is msgs[1]
    assert queue.buffered_bytes == _queued_bytes(queue)
    assert queue.buffered_bytes >= 0

    # Drain remainder via async gets; accounting stays exact, never negative.
    while queue.qsize():
        await queue.get()
        assert queue.buffered_bytes == _queued_bytes(queue)
    assert queue.buffered_bytes == 0

    # Slow backlog cannot exceed configured bytes: evicted, drained, closed.
    store = TaskStore(
        max_subscriber_messages=1000,
        max_subscriber_bytes=500,
        max_subscribers_per_task=4,
    )
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    slow = store.subscribe("t1")
    assert isinstance(slow, _SubscriberQueue)
    big = _msg("😀" * 100)
    store.publish("t1", big)
    assert store.subscriber_count("t1") == 1
    assert slow.buffered_bytes == _message_size(big) <= 500
    store.publish("t1", big)
    assert store.subscriber_count("t1") == 0
    assert slow.get_nowait() is None
    assert slow.buffered_bytes == 0


class _FakeStream:
    """Minimal StreamResponse stand-in capturing SSE writes."""

    def __init__(self, *args, **kwargs) -> None:
        self.written: list[bytes] = []
        self.prepared = False
        self.eof = False

    async def prepare(self, request) -> "_FakeStream":
        self.prepared = True
        return self

    async def write(self, data: bytes) -> None:
        self.written.append(data)

    async def write_eof(self) -> None:
        self.eof = True


async def test_blocked_replay_holds_cap_429_and_prepare_releases(
    monkeypatch: pytest.MonkeyPatch,
):
    store = TaskStore(max_subscribers_per_task=16)
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    store.publish("t1", _msg("seed"))

    gate = asyncio.Event()
    instances: list[_FakeStream] = []

    class _Blocking(_FakeStream):
        async def write(self, data: bytes) -> None:
            instances.append(self) if self not in instances else None
            await gate.wait()
            self.written.append(data)

    monkeypatch.setattr(wire_mod.web, "StreamResponse", _Blocking)
    streams = [
        asyncio.create_task(serve_task_stream(store, "t1", object(), write_timeout=5.0))  # type: ignore[arg-type]
        for _ in range(16)
    ]
    try:
        for _ in range(100):
            if store.subscriber_count("t1") == 16:
                break
            await asyncio.sleep(0.02)
        assert store.subscriber_count("t1") == 16

        # 17th stream is denied at the cap before any header/write.
        denied = await asyncio.wait_for(
            serve_task_stream(store, "t1", object(), write_timeout=1.0),  # type: ignore[arg-type]
            timeout=2.0,
        )
        assert isinstance(denied, web.Response)
        assert denied.status == 429
        assert store.subscriber_count("t1") == 16
    finally:
        gate.set()
        # Live streams block for the close signal after replay; complete so
        # all 16 finish and release their slots.
        try:
            store.complete("t1")
        except Exception:
            pass
        try:
            store.broadcast("t1", None)
        except Exception:
            pass
        results = await asyncio.wait_for(asyncio.gather(*streams), timeout=5.0)
        assert all(isinstance(r, _Blocking) for r in results)
        assert store.subscriber_count("t1") == 0

    class _PrepareFails(_FakeStream):
        async def prepare(self, request):
            raise ConnectionError("client gone")

    monkeypatch.setattr(wire_mod.web, "StreamResponse", _PrepareFails)
    with pytest.raises(ConnectionError):
        await serve_task_stream(store, "t1", object(), write_timeout=0.5)  # type: ignore[arg-type]
    assert store.subscriber_count("t1") == 0


async def test_ring_roll_mid_replay_delivers_once_incl_final(
    monkeypatch: pytest.MonkeyPatch,
):
    store = TaskStore(
        max_task_buffer_bytes=64 * 1024,
        max_task_buffer_messages=2,
        max_subscriber_messages=64,
        max_subscriber_bytes=64 * 1024,
    )
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    store.publish("t1", _msg("m0"))

    started = asyncio.Event()
    release = asyncio.Event()
    holder: dict[str, _FakeStream] = {}

    class _PauseFirst(_FakeStream):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._calls = 0

        async def write(self, data: bytes) -> None:
            self._calls += 1
            holder.setdefault("resp", self)
            if self._calls == 1:
                started.set()
                await release.wait()
            self.written.append(data)

    monkeypatch.setattr(wire_mod.web, "StreamResponse", _PauseFirst)
    stream = asyncio.create_task(
        serve_task_stream(store, "t1", object(), write_timeout=5.0)  # type: ignore[arg-type]
    )
    await asyncio.wait_for(started.wait(), timeout=2.0)
    # Ring rolls while the first replay write awaits; task then completes.
    store.publish("t1", _msg("m1"))
    store.publish("t1", _msg("m2"))
    store.publish("t1", _msg("m3"))
    assert [m.parts[0].text for m in store.snapshot("t1")] == ["m2", "m3"]
    store.complete("t1")
    store.broadcast("t1", None)
    release.set()

    response = await asyncio.wait_for(stream, timeout=5.0)
    assert isinstance(response, _PauseFirst)
    assert _sse_texts(response.written) == ["m0", "m1", "m2", "m3"]
    assert _sse_errors(response.written) == []  # live queue got all; no loss
    assert _sse_completes(response.written) == [{"message_count": 4}]
    assert store.subscriber_count("t1") == 0
    # Final retained for re-subscribe despite the roll.
    assert store.snapshot("t1")[-1].parts[0].text == "m3"


async def test_initially_terminal_never_hangs_and_eviction_truncates(
    monkeypatch: pytest.MonkeyPatch,
):
    store = TaskStore()
    store.create(task_id="done", skill="chat", session_id="1", creator_peer="m")
    store.publish("done", _msg("a"))
    store.publish("done", _msg("b"))
    store.complete("done")
    store.broadcast("done", None)

    monkeypatch.setattr(wire_mod.web, "StreamResponse", _FakeStream)
    response = await asyncio.wait_for(
        serve_task_stream(store, "done", object(), write_timeout=1.0),  # type: ignore[arg-type]
        timeout=2.0,
    )
    assert isinstance(response, _FakeStream)
    assert _sse_texts(response.written) == ["a", "b"]
    assert _sse_errors(response.written) == []
    assert _sse_completes(response.written) == [{"message_count": 2}]
    assert store.subscriber_count("done") == 0

    # Slow subscriber evicted while replay writes ends truncated, not full.
    tiny = TaskStore(max_subscriber_messages=1, max_subscriber_bytes=10**6)
    tiny.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    tiny.publish("t1", _msg("m0"))
    started = asyncio.Event()
    release = asyncio.Event()

    class _PauseFirst(_FakeStream):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._calls = 0

        async def write(self, data: bytes) -> None:
            self._calls += 1
            if self._calls == 1:
                started.set()
                await release.wait()
            self.written.append(data)

    monkeypatch.setattr(wire_mod.web, "StreamResponse", _PauseFirst)
    stream = asyncio.create_task(
        serve_task_stream(tiny, "t1", object(), write_timeout=5.0)  # type: ignore[arg-type]
    )
    await asyncio.wait_for(started.wait(), timeout=2.0)
    tiny.publish("t1", _msg("m1"))
    tiny.publish("t1", _msg("m2"))  # overflows the 1-message queue -> evict
    assert tiny.subscriber_count("t1") == 0
    tiny.complete("t1")
    tiny.broadcast("t1", None)
    release.set()
    evicted = await asyncio.wait_for(stream, timeout=5.0)
    assert isinstance(evicted, _PauseFirst)
    # Only the pre-eviction snapshot replay; queued future was dropped.
    assert _sse_texts(evicted.written) == ["m0"]
    # Fail-closed: eviction emits a distinguishable truncation error, never
    # clean EOF as full success.
    errors = _sse_errors(evicted.written)
    assert len(errors) == 1
    assert errors[0].get("error") == "stream_truncated"
    assert _sse_completes(evicted.written) == []  # truncated never completes
    # Final still retained for a fast re-subscribe.
    assert tiny.snapshot("t1")[-1].parts[0].text == "m2"


async def test_failed_live_worker_counted_protected_then_released():
    async def handler(params):
        yield A2AMessage(role="agent", parts=[Part(type="text", text="x" * 10000)])
        await asyncio.sleep(0.3)
        yield A2AMessage(role="agent", parts=[Part(type="text", text="small")])

    store = TaskStore(
        max_task_buffer_bytes=500, max_active_tasks=1, completed_ttl_seconds=60.0
    )
    card = AgentCard(name="Evernight", description="", url="", version="1")
    server = A2AServer(
        agent_card=card,
        skill_handlers={"chat": handler},
        agent_name="evernight",
        shared_secret="s",
        task_store=store,
    )
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    worker = asyncio.create_task(server._execute_handler("t1", handler, {}))
    store.set_worker("t1", worker)
    try:
        await asyncio.sleep(0.1)
        assert store.get("t1") is not None
        assert store.get("t1").status == TaskStatus.FAILED
        assert not worker.done()
        assert store.active_count() == 1
        assert store.purge_expired(now=time.time() + 1000) == 0
        assert store.get("t1") is not None
        assert (
            store.create(task_id="t2", skill="chat", session_id="1", creator_peer="m")
            is None
        )
        await asyncio.wait_for(worker, timeout=2.0)
        assert store.active_count() == 0
        # Handle not retained forever once the worker is done.
        assert store.purge_expired(now=time.time() + 1000) == 1
        assert store.get("t1") is None
    finally:
        if not worker.done():
            worker.cancel()
            await asyncio.wait([worker], timeout=1.0)

    # Bounded shutdown still holds with a swallowing worker.
    async def bad():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(30)

    store2 = TaskStore()
    store2.create(task_id="s1", skill="chat", session_id="1", creator_peer="m")
    stuck = asyncio.create_task(bad())
    store2.set_worker("s1", stuck)
    await asyncio.sleep(0.02)
    try:
        start = time.monotonic()
        await asyncio.wait_for(store2.shutdown(timeout=0.2), timeout=2.0)
        assert time.monotonic() - start < 2.0
        assert store2.get("s1").status == TaskStatus.CANCELLED
    finally:
        stuck.cancel()
        await asyncio.wait([stuck], timeout=1.0)
