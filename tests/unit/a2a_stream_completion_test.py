"""A2A stream completion contract: marker required, truncation never succeeds.

Wire: clean close appends `event: complete` with count of normal frames
actually written; truncation/eviction/timeout omits it. Client requires a
single valid marker, exact count match, no post-marker data, no leftover.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from twin.shared.a2a import wire as wire_mod
from twin.shared.a2a.client import A2AClient, A2AStreamError
from twin.shared.a2a.server import A2AServer
from twin.shared.a2a.sse import SseParser
from twin.shared.a2a.tasks import TaskStore
from twin.shared.a2a.types import A2AMessage, A2ATask, AgentCard, Part, TaskStatus
from twin.shared.a2a.auth import make_a2a_headers

SECRET = "test-a2a-shared-secret"


def _card() -> AgentCard:
    return AgentCard(name="Evernight", description="", url="", version="1")


def _msg(text: str) -> A2AMessage:
    return A2AMessage(role="agent", parts=[Part(type="text", text=text)])


def _server(store: TaskStore | None = None, handlers: dict | None = None) -> A2AServer:
    return A2AServer(
        agent_card=_card(),
        skill_handlers=handlers or {"chat": _echo_two},
        agent_name="evernight",
        shared_secret=SECRET,
        task_store=store or TaskStore(),
    )


async def _echo_two(params):
    yield _msg("one")
    yield _msg("two")


async def _empty_handler(params):
    if False:
        yield _msg("unreached")


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


def _data(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


def _complete(count: int) -> bytes:
    return f"event: complete\ndata: {json.dumps({'message_count': count})}\n\n".encode()


def _complete_raw(raw: str) -> bytes:
    return f"event: complete\ndata: {raw}\n\n".encode()


def _fake_app(payload: bytes):
    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(payload)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/tasks/{task_id}/stream", handler)
    return app


async def _subscribe_texts(client: A2AClient, task_id: str = "t1") -> list[str]:
    out = []
    async for m in client.subscribe_stream(task_id):
        out.append(m.parts[0].text or "")
    return out


def test_stream_error_reexported():
    from twin.shared.a2a.sse import A2AStreamError as FromSse

    assert A2AStreamError is FromSse


async def test_parent_proof_second_write_timeout_rejects_despite_completed():
    store = TaskStore()
    store.create(task_id="proof", skill="chat", session_id="1", creator_peer="march7")
    for text in ("partial-preview", "required-final-result"):
        store.publish("proof", _msg(text))
    store.complete("proof")
    srv = _server(store)
    async with _test_client(srv.build_app()) as raw:
        original = wire_mod._write_bytes
        count = 0

        async def interrupted(response, data, timeout):
            nonlocal count
            count += 1
            if count == 2:
                raise asyncio.TimeoutError("synthetic SSE write deadline")
            await original(response, data, timeout)

        wire_mod._write_bytes = interrupted  # type: ignore[assignment]
        client = A2AClient(_base(raw), actor="march7", secret=SECRET)
        client.send_task = AsyncMock(return_value=store.get("proof"))  # type: ignore[method-assign]
        try:
            with pytest.raises(A2AStreamError, match="truncated"):
                await client.send_task_and_wait({"id": "proof"})
            assert (await client.get_task("proof")).status == TaskStatus.COMPLETED
        finally:
            wire_mod._write_bytes = original  # type: ignore[assignment]
            await client.close()


async def test_valid_two_messages_roundtrip():
    store = TaskStore()
    srv = _server(store)
    async with _test_client(srv.build_app()) as raw:
        client = A2AClient(_base(raw), actor="march7", secret=SECRET)
        try:
            out = await client.send_task_and_wait(
                {"skill": "chat", "sessionId": "1", "id": "v2"}
            )
            assert [m.parts[0].text for m in out] == ["one", "two"]
            headers = make_a2a_headers(
                "march7", "GET", "/tasks/v2/stream", b"", secret=SECRET
            )
            async with raw.get("/tasks/v2/stream", headers=headers) as resp:
                body = await resp.read()
            assert b"event: complete" in body
            assert json.loads(body.split(b"event: complete")[1].split(b"data:")[1].split(b"\n")[0]) == {"message_count": 2}
        finally:
            await client.close()


async def test_valid_zero_output_requires_count_zero():
    store = TaskStore()
    srv = _server(store, {"chat": _empty_handler})
    async with _test_client(srv.build_app()) as raw:
        client = A2AClient(_base(raw), actor="march7", secret=SECRET)
        try:
            assert await client.send_task_and_wait(
                {"skill": "chat", "sessionId": "1", "id": "z0"}
            ) == []
            headers = make_a2a_headers(
                "march7", "GET", "/tasks/z0/stream", b"", secret=SECRET
            )
            async with raw.get("/tasks/z0/stream", headers=headers) as resp:
                body = (await resp.read()).decode()
            assert body.count("event: complete") == 1
            assert '"message_count":0' in body.replace(" ", "")
        finally:
            await client.close()


async def test_missing_complete_fails_closed():
    good = {"role": "agent", "parts": [{"type": "text", "text": "hi"}]}
    for payload in (_data(good), b"", _data(good) + _data(good)):
        async with _test_client(_fake_app(payload)) as raw:
            client = A2AClient(_base(raw), actor="march7", secret=SECRET)
            try:
                with pytest.raises(A2AStreamError, match="truncated"):
                    await _subscribe_texts(client)
            finally:
                await client.close()


async def test_unterminated_event_and_json_fail_closed():
    good = {"role": "agent", "parts": [{"type": "text", "text": "hi"}]}
    cases = [
        b'data: {"role": "agent", "parts": [',
        b'data: {"role": "agent"}\n',
        _data(good) + b'data: {"incomplete": true',
        _data(good) + b'event: complete\ndata: {"message_count": 1}\n',
    ]
    for payload in cases:
        async with _test_client(_fake_app(payload)) as raw:
            client = A2AClient(_base(raw), actor="march7", secret=SECRET)
            try:
                with pytest.raises(A2AStreamError, match="truncated"):
                    await _subscribe_texts(client)
            finally:
                await client.close()


async def test_malformed_count_fails_closed():
    good = {"role": "agent", "parts": [{"type": "text", "text": "hi"}]}
    bad_payloads = [
        _data(good) + _complete_raw('{"nope": 1}'),
        _data(good) + _complete_raw('"1"'),
        _data(good) + _complete_raw('{"message_count": "1"}'),
        _data(good) + _complete_raw('{"message_count": 1.0}'),
        _data(good) + _complete_raw('{"message_count": true}'),
        _data(good) + _complete_raw('{"message_count": false}'),
        _data(good) + _complete_raw('{"message_count": -1}'),
        _data(good) + _complete_raw('{"message_count": null}'),
        _data(good) + _complete_raw('not-json'),
        _data(good) + b"event: complete\n\n",
        _data(good) + _complete(0),
        _data(good) + _complete(2),
        _complete(1),
        _data(good) + _complete(1) + _complete(1),
    ]
    for payload in bad_payloads:
        async with _test_client(_fake_app(payload)) as raw:
            client = A2AClient(_base(raw), actor="march7", secret=SECRET)
            try:
                with pytest.raises(A2AStreamError, match="truncated"):
                    await _subscribe_texts(client)
            finally:
                await client.close()


async def test_data_after_complete_fails_closed():
    good = {"role": "agent", "parts": [{"type": "text", "text": "hi"}]}
    payload = _data(good) + _complete(1) + _data(good)
    async with _test_client(_fake_app(payload)) as raw:
        client = A2AClient(_base(raw), actor="march7", secret=SECRET)
        try:
            with pytest.raises(A2AStreamError, match="truncated"):
                await _subscribe_texts(client)
        finally:
            await client.close()


async def test_invalid_json_not_silent_continue():
    payload = b"data: not-json\n\n" + _complete(0)
    async with _test_client(_fake_app(payload)) as raw:
        client = A2AClient(_base(raw), actor="march7", secret=SECRET)
        try:
            with pytest.raises(A2AStreamError, match="truncated"):
                await _subscribe_texts(client)
        finally:
            await client.close()


async def test_invalid_utf8_fails_closed():
    good = {"role": "agent", "parts": [{"type": "text", "text": "hi"}]}
    payload = _data(good) + b"\xff\xfe\n\n" + _complete(1)
    async with _test_client(_fake_app(payload)) as raw:
        client = A2AClient(_base(raw), actor="march7", secret=SECRET)
        try:
            with pytest.raises(A2AStreamError, match="truncated"):
                await _subscribe_texts(client)
        finally:
            await client.close()
    parser = SseParser()
    with pytest.raises(A2AStreamError, match="truncated"):
        parser.feed(b"\xff")


async def test_early_eof_marker_timeout_rejects_despite_completed():
    store = TaskStore()
    store.create(task_id="e1", skill="chat", session_id="1", creator_peer="march7")
    store.publish("e1", _msg("a"))
    store.publish("e1", _msg("b"))
    store.complete("e1")
    store.broadcast("e1", None)
    srv = _server(store)
    async with _test_client(srv.build_app()) as raw:
        original = wire_mod._write_bytes
        count = 0

        async def fail_marker(response, data, timeout):
            nonlocal count
            count += 1
            if count == 3:
                raise asyncio.TimeoutError("marker deadline")
            await original(response, data, timeout)

        wire_mod._write_bytes = fail_marker  # type: ignore[assignment]
        client = A2AClient(_base(raw), actor="march7", secret=SECRET)
        try:
            with pytest.raises(A2AStreamError, match="truncated"):
                await _subscribe_texts(client, "e1")
            assert store.get("e1").status == TaskStatus.COMPLETED
        finally:
            wire_mod._write_bytes = original  # type: ignore[assignment]
            await client.close()


async def test_utf8_split_boundaries_roundtrip():
    parser = SseParser()
    raw_text = "😀" * 10
    raw_blob = (
        f"data: {json.dumps({'role': 'agent', 'parts': [{'type': 'text', 'text': raw_text}]}, ensure_ascii=False)}\n\n"
    ).encode("utf-8") + _complete(1)
    idx = raw_blob.index("😀".encode("utf-8")) + 2
    first = parser.feed(raw_blob[:idx])
    second = parser.feed(raw_blob[idx:])
    assert first == [] and len(second) == 1
    parser.finish()

    store = TaskStore()
    store.create(task_id="u1", skill="chat", session_id="1", creator_peer="march7")
    store.publish("u1", _msg("pad-" + "😀" * 600))
    store.complete("u1")
    store.broadcast("u1", None)
    srv = _server(store)
    async with _test_client(_fake_app(b"")) as _:
        pass
    async with _test_client(srv.build_app()) as raw:
        client = A2AClient(_base(raw), actor="march7", secret=SECRET)
        try:
            texts = await _subscribe_texts(client, "u1")
            assert texts == ["pad-" + "😀" * 600]
        finally:
            await client.close()


async def test_failed_transport_marker_ok_status_still_raises():
    store = TaskStore()
    store.create(task_id="f1", skill="chat", session_id="1", creator_peer="march7")
    store.publish("f1", _msg("before-fail"))
    store.fail("f1")
    store.broadcast("f1", None)
    srv = _server(store)
    async with _test_client(srv.build_app()) as raw:
        client = A2AClient(_base(raw), actor="march7", secret=SECRET)
        client.send_task = AsyncMock(  # type: ignore[method-assign]
            return_value=A2ATask(id="f1", status=TaskStatus.IN_PROGRESS)
        )
        try:
            assert await _subscribe_texts(client, "f1") == ["before-fail"]
            with pytest.raises(RuntimeError, match="before-fail|failed"):
                await client.send_task_and_wait({"id": "f1"})
        finally:
            await client.close()
