"""A2A authentication, authorization, lifecycle, and retention tests.

Covers findings #1 (unsigned cross-user memory access), #3 (port exposure is
a compose-hunk change, asserted in review), #9 (tool scope binding lives in
a2a_tools_test.py), #14 (cancel must stop workers), and #15 (bounded tasks).
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from twin.shared.a2a.auth import (
    A2AAuthError,
    NonceStore,
    TIMESTAMP_SKEW_SECONDS,
    make_a2a_headers,
    verify_a2a_request,
)
from twin.shared.a2a.client import A2AClient
from twin.shared.a2a.server import A2AServer
from twin.shared.a2a.tasks import TaskStore
from twin.shared.a2a.types import A2AMessage, AgentCard, Part
from twin.shared.config.settings import Config
from twin.shared.memory.consolidation_client import ConsolidationClient
from twin.shared.tools.modules.a2a.march7_snapshot_tool import March7SnapshotTool
from gateway.core.evernight_client import EvernightClient

SECRET = "test-a2a-shared-secret"


def _card(name: str) -> AgentCard:
    return AgentCard(name=name, description="", url="", version="1")


async def _echo_handler(params):
    yield A2AMessage(role="agent", parts=[Part(type="text", text="hi")])


async def _data_handler(params):
    yield A2AMessage(role="agent", parts=[Part(type="data", data={"ok": True})])


def _march7_server(**kwargs) -> A2AServer:
    from twin.march7.server.a2a_server import validate_march7_session

    kwargs.setdefault("agent_name", "march7")
    kwargs.setdefault("shared_secret", SECRET)
    kwargs.setdefault("session_validator", validate_march7_session)
    return A2AServer(
        agent_card=_card("March7"),
        skill_handlers={
            "chat": _echo_handler,
            "get_snapshot": _data_handler,
            "clear_session": _data_handler,
        },
        **kwargs,
    )


def _evernight_server(**kwargs) -> A2AServer:
    kwargs.setdefault("agent_name", "evernight")
    kwargs.setdefault("shared_secret", SECRET)
    return A2AServer(
        agent_card=_card("Evernight"),
        skill_handlers={
            "chat": _echo_handler,
            "consolidate": _data_handler,
            "consolidate_discussion": _data_handler,
        },
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


def _rpc_body(method: str, params: dict, rpc_id: str = "1") -> bytes:
    return json.dumps(
        {"jsonrpc": "2.0", "method": method, "params": params, "id": rpc_id}
    ).encode("utf-8")


async def _post_rpc(client, body: bytes, headers: dict | None = None):
    all_headers = {"Content-Type": "application/json"}
    if headers:
        all_headers.update(headers)
    async with client.post("/", data=body, headers=all_headers) as resp:
        return resp.status, await resp.json()


def _signed(
    actor: str, method: str, path: str, body: bytes, secret: str = SECRET, **kwargs
) -> dict[str, str]:
    return make_a2a_headers(actor, method, path, body, secret=secret, **kwargs)


def _base_url(client: TestClient) -> str:
    return str(client.make_url("/")).rstrip("/")


# ---------------------------------------------------------------------------
# Header signing / verification unit tests
# ---------------------------------------------------------------------------


def test_headers_verify_roundtrip():
    body = b'{"hello":"world"}'
    headers = _signed("march7", "POST", "/", body)
    store = NonceStore()
    assert (
        verify_a2a_request(
            headers=headers,
            method="POST",
            path="/",
            body=body,
            secret=SECRET,
            nonce_store=store,
        )
        == "march7"
    )


def test_verify_rejects_tampered_body():
    headers = _signed("march7", "POST", "/", b'{"a":1}')
    with pytest.raises(A2AAuthError, match="invalid signature"):
        verify_a2a_request(
            headers=headers,
            method="POST",
            path="/",
            body=b'{"a":2}',
            secret=SECRET,
            nonce_store=NonceStore(),
        )


def test_verify_rejects_retargeted_method_and_path():
    body = b""
    headers = _signed("march7", "POST", "/dm", body)
    store = NonceStore()
    with pytest.raises(A2AAuthError, match="invalid signature"):
        verify_a2a_request(
            headers=headers, method="GET", path="/dm", body=body,
            secret=SECRET, nonce_store=store,
        )
    with pytest.raises(A2AAuthError, match="invalid signature"):
        verify_a2a_request(
            headers=headers, method="POST", path="/", body=body,
            secret=SECRET, nonce_store=store,
        )


def test_verify_rejects_wrong_secret():
    body = b"{}"
    headers = _signed("march7", "POST", "/", body, secret="other")
    with pytest.raises(A2AAuthError, match="invalid signature"):
        verify_a2a_request(
            headers=headers, method="POST", path="/", body=body,
            secret=SECRET, nonce_store=NonceStore(),
        )


def test_verify_rejects_expired_and_future_timestamps():
    body = b"{}"
    old = _signed("march7", "POST", "/", body, timestamp=1)
    with pytest.raises(A2AAuthError, match="timestamp"):
        verify_a2a_request(
            headers=old, method="POST", path="/", body=body,
            secret=SECRET, nonce_store=NonceStore(), now=10**10,
        )
    future = _signed(
        "march7", "POST", "/", body, timestamp=10**10 + TIMESTAMP_SKEW_SECONDS + 1
    )
    with pytest.raises(A2AAuthError, match="timestamp"):
        verify_a2a_request(
            headers=future, method="POST", path="/", body=body,
            secret=SECRET, nonce_store=NonceStore(), now=10**10,
        )


def test_verify_rejects_replayed_nonce():
    body = b"{}"
    headers = _signed("march7", "POST", "/", body)
    store = NonceStore()
    verify_a2a_request(
        headers=headers, method="POST", path="/", body=body,
        secret=SECRET, nonce_store=store,
    )
    with pytest.raises(A2AAuthError, match="replay"):
        verify_a2a_request(
            headers=headers, method="POST", path="/", body=body,
            secret=SECRET, nonce_store=store,
        )


def test_verify_rejects_missing_headers_and_secret():
    with pytest.raises(A2AAuthError, match="missing auth headers"):
        verify_a2a_request(
            headers={}, method="POST", path="/", body=b"{}",
            secret=SECRET, nonce_store=NonceStore(),
        )
    headers = _signed("march7", "POST", "/", b"{}")
    with pytest.raises(A2AAuthError, match="missing .* secret"):
        verify_a2a_request(
            headers=headers, method="POST", path="/", body=b"{}",
            secret=None, nonce_store=NonceStore(),
        )


def test_make_headers_rejects_unknown_actor_and_missing_secret(monkeypatch):
    with pytest.raises(A2AAuthError, match="unknown A2A actor"):
        make_a2a_headers("intruder", "POST", "/", b"{}", secret=SECRET)
    monkeypatch.setattr(Config, "A2A_SHARED_SECRET", None)
    with pytest.raises(A2AAuthError, match="missing .* secret"):
        make_a2a_headers("march7", "POST", "/", b"{}")


def test_nonce_store_bounds_and_purges():
    store = NonceStore(ttl_seconds=10, max_entries=3, clock=lambda: 100.0)
    assert store.check_and_add("a")
    assert store.check_and_add("b")
    assert store.check_and_add("c")
    assert not store.check_and_add("b")
    assert store.check_and_add("d")
    assert len(store) == 3
    # Oldest ("a") was evicted to make room.
    assert store.check_and_add("a")
    assert store.purge(now=1000.0) == 3
    assert len(store) == 0


# ---------------------------------------------------------------------------
# Public routes vs protected routes
# ---------------------------------------------------------------------------


async def test_health_and_agent_card_are_public():
    async with _test_client(_march7_server().build_app()) as client:
        async with client.get("/health") as resp:
            assert resp.status == 200
        async with client.get("/.well-known/agent.json") as resp:
            assert resp.status == 200


async def test_unsigned_rpc_is_denied():
    async with _test_client(_march7_server().build_app()) as client:
        body = _rpc_body("tasks/send", {"skill": "chat", "sessionId": "1"})
        status, _ = await _post_rpc(client, body)
        assert status == 401


async def test_unsigned_clear_session_of_another_user_is_denied():
    """Regression for finding #1: no auth, no cross-user memory mutation."""
    async with _test_client(_march7_server().build_app()) as client:
        body = _rpc_body(
            "tasks/send", {"skill": "clear_session", "sessionId": "999"}
        )
        status, _ = await _post_rpc(client, body)
        assert status == 401
        body = _rpc_body(
            "tasks/send", {"skill": "get_snapshot", "sessionId": "999"}
        )
        headers = _signed("march7", "POST", "/", body)
        status, data = await _post_rpc(client, body, headers)
        assert status == 200
        assert data["error"]["code"] == -32003


async def test_tampered_body_and_replay_are_denied_over_http():
    async with _test_client(_march7_server().build_app()) as client:
        body = _rpc_body("tasks/send", {"skill": "chat", "sessionId": "1"})
        headers = _signed("owner", "POST", "/", body)
        tampered = _rpc_body("tasks/send", {"skill": "chat", "sessionId": "2"})
        status, _ = await _post_rpc(client, tampered, headers)
        assert status == 401
        status, _ = await _post_rpc(client, body, headers)
        assert status == 200
        status, _ = await _post_rpc(client, body, headers)
        assert status == 401


async def test_missing_secret_denies_even_signed_requests(monkeypatch):
    monkeypatch.setattr(Config, "A2A_SHARED_SECRET", None)
    server = _march7_server(shared_secret=None)
    async with _test_client(server.build_app()) as client:
        body = _rpc_body("tasks/send", {"skill": "chat", "sessionId": "1"})
        headers = _signed("owner", "POST", "/", body, secret="anything")
        status, _ = await _post_rpc(client, body, headers)
        assert status == 401


# ---------------------------------------------------------------------------
# Per-peer skill policy
# ---------------------------------------------------------------------------


async def test_march7_skill_policy():
    async with _test_client(_march7_server().build_app()) as client:
        snap = _rpc_body("tasks/send", {"skill": "get_snapshot", "sessionId": "123"})
        status, data = await _post_rpc(
            client, snap, _signed("evernight", "POST", "/", snap)
        )
        assert status == 200 and "result" in data

        status, data = await _post_rpc(
            client, snap, _signed("owner", "POST", "/", snap)
        )
        assert status == 200 and "result" in data

        status, data = await _post_rpc(
            client, snap, _signed("march7", "POST", "/", snap)
        )
        assert data["error"]["code"] == -32003

        chat = _rpc_body("tasks/send", {"skill": "chat", "sessionId": "123"})
        status, data = await _post_rpc(
            client, chat, _signed("evernight", "POST", "/", chat)
        )
        assert data["error"]["code"] == -32003
        status, data = await _post_rpc(
            client, chat, _signed("owner", "POST", "/", chat)
        )
        assert status == 200 and "result" in data


async def test_evernight_skill_policy():
    async with _test_client(_evernight_server().build_app()) as client:
        chat = _rpc_body("tasks/send", {"skill": "chat", "sessionId": "123"})
        status, data = await _post_rpc(
            client, chat, _signed("march7", "POST", "/", chat)
        )
        assert status == 200 and "result" in data
        status, data = await _post_rpc(
            client, chat, _signed("evernight", "POST", "/", chat)
        )
        assert data["error"]["code"] == -32003

        unknown = _rpc_body("tasks/send", {"skill": "get_snapshot", "sessionId": "1"})
        status, data = await _post_rpc(
            client, unknown, _signed("march7", "POST", "/", unknown)
        )
        assert data["error"]["code"] == -32003


async def test_default_policy_derives_from_agent_card():
    server = A2AServer(
        agent_card=_card("Evernight"),
        skill_handlers={"chat": _echo_handler},
        shared_secret=SECRET,
    )
    assert server.agent_name == "evernight"
    async with _test_client(server.build_app()) as client:
        chat = _rpc_body("tasks/send", {"skill": "chat", "sessionId": "1"})
        status, data = await _post_rpc(
            client, chat, _signed("march7", "POST", "/", chat)
        )
        assert status == 200 and "result" in data

    server = A2AServer(
        agent_card=_card("mystery"),
        skill_handlers={"chat": _echo_handler},
        shared_secret=SECRET,
    )
    async with _test_client(server.build_app()) as client:
        chat = _rpc_body("tasks/send", {"skill": "chat", "sessionId": "1"})
        status, data = await _post_rpc(
            client, chat, _signed("owner", "POST", "/", chat)
        )
        assert data["error"]["code"] == -32003


async def test_march7_session_validation():
    async with _test_client(_march7_server().build_app()) as client:
        bad = _rpc_body("tasks/send", {"skill": "get_snapshot", "sessionId": "abc"})
        status, data = await _post_rpc(
            client, bad, _signed("evernight", "POST", "/", bad)
        )
        assert data["error"]["code"] == -32602

        missing = _rpc_body("tasks/send", {"skill": "clear_session"})
        status, data = await _post_rpc(
            client, missing, _signed("evernight", "POST", "/", missing)
        )
        assert data["error"]["code"] == -32602

        missing_chat = _rpc_body("tasks/send", {"skill": "chat"})
        status, data = await _post_rpc(
            client, missing_chat, _signed("owner", "POST", "/", missing_chat)
        )
        assert data["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# Task ownership
# ---------------------------------------------------------------------------


async def test_task_read_cancel_stream_bound_to_creator():
    async with _test_client(_evernight_server().build_app()) as client:
        send = _rpc_body(
            "tasks/send", {"id": "owned-1", "skill": "chat", "sessionId": "1"}
        )
        status, data = await _post_rpc(
            client, send, _signed("march7", "POST", "/", send)
        )
        assert status == 200 and data["result"]["id"] == "owned-1"

        get = _rpc_body("tasks/get", {"id": "owned-1"})
        status, data = await _post_rpc(
            client, get, _signed("owner", "POST", "/", get)
        )
        assert data["error"]["code"] == -32003
        status, data = await _post_rpc(
            client, get, _signed("march7", "POST", "/", get)
        )
        assert data["result"]["id"] == "owned-1"

        missing = _rpc_body("tasks/get", {"id": "nope"})
        status, data = await _post_rpc(
            client, missing, _signed("march7", "POST", "/", missing)
        )
        assert data["result"] == {}

        cancel = _rpc_body("tasks/cancel", {"id": "owned-1"})
        status, data = await _post_rpc(
            client, cancel, _signed("owner", "POST", "/", cancel)
        )
        assert data["error"]["code"] == -32003

        stream_headers = _signed(
            "owner", "GET", "/tasks/owned-1/stream", b""
        )
        async with client.get("/tasks/owned-1/stream", headers=stream_headers) as resp:
            assert resp.status == 403
        async with client.get("/tasks/nope/stream", headers=_signed(
            "march7", "GET", "/tasks/nope/stream", b""
        )) as resp:
            assert resp.status == 404


async def test_duplicate_task_id_same_peer_returns_existing():
    calls = []

    async def counting(params):
        calls.append(1)
        yield A2AMessage(role="agent", parts=[Part(type="text", text="x")])

    server = _evernight_server()
    server.skill_handlers["chat"] = counting
    async with _test_client(server.build_app()) as client:
        send = _rpc_body(
            "tasks/send", {"id": "dup-1", "skill": "chat", "sessionId": "1"}
        )
        status, first = await _post_rpc(
            client, send, _signed("march7", "POST", "/", send)
        )
        assert status == 200
        status, second = await _post_rpc(
            client, send, _signed("march7", "POST", "/", send)
        )
        assert status == 200
        assert second["result"]["id"] == "dup-1"
        await asyncio.sleep(0.2)
        assert len(calls) == 1

        status, data = await _post_rpc(
            client, send, _signed("owner", "POST", "/", send)
        )
        assert data["error"]["code"] == -32003


# ---------------------------------------------------------------------------
# Signed client end-to-end
# ---------------------------------------------------------------------------


async def test_signed_client_roundtrip_and_wrong_secret():
    async with _test_client(_evernight_server().build_app()) as raw:
        base_url = _base_url(raw)
        client = A2AClient(base_url=base_url, actor="march7", secret=SECRET)
        try:
            assert (
                await client.send_text_task(
                    skill="chat", session_id="1", text="hello"
                )
                == "hi"
            )
            assert await client.send_data_task(
                skill="consolidate", session_id="1"
            ) == {"ok": True}
            task = await client.send_task(
                {"id": "c1", "skill": "chat", "sessionId": "1"}
            )
            assert task.id == "c1"
            assert (await client.get_task("c1")).id == "c1"
        finally:
            await client.close()

        bad = A2AClient(base_url=base_url, actor="march7", secret="wrong")
        try:
            with pytest.raises(RuntimeError):
                await bad.send_text_task(skill="chat", session_id="1", text="hi")
        finally:
            await bad.close()


async def test_client_without_secret_fails_closed(monkeypatch):
    monkeypatch.setattr(Config, "A2A_SHARED_SECRET", None)
    client = A2AClient(base_url="http://127.0.0.1:9", actor="march7")
    try:
        with pytest.raises(A2AAuthError):
            await client.send_text_task(skill="chat", session_id="1", text="hi")
    finally:
        await client.close()


async def test_callers_use_explicit_actors_and_passthrough_secret():
    assert ConsolidationClient("http://x:8001").a2a_client.actor == "march7"
    assert EvernightClient("http://x:8001")._client.actor == "march7"
    assert March7SnapshotTool(march7_url="http://x:8000")._actor == "evernight"
    assert (
        ConsolidationClient("http://x:8001", secret="s").a2a_client._secret == "s"
    )
    assert EvernightClient("http://x:8001", secret="s")._client._secret == "s"
    assert (
        March7SnapshotTool(march7_url="http://x:8000", secret="s")._secret == "s"
    )


async def test_consolidation_client_end_to_end():
    async with _test_client(_evernight_server().build_app()) as raw:
        base_url = _base_url(raw)
        client = ConsolidationClient(base_url, secret=SECRET)
        try:
            result = await client.consolidate_scope("user", "u1", entries=[])
            assert result["ok"] is True
        finally:
            await client.close()


async def test_evernight_client_end_to_end():
    async with _test_client(_evernight_server().build_app()) as raw:
        base_url = _base_url(raw)
        client = EvernightClient(base_url, secret=SECRET)
        try:
            assert await client.send_chat("u1", "hello") == "hi"
            result = await client.request_consolidation(
                {"scope": "user", "scope_id": "u1"}
            )
            assert result["ok"] is True
        finally:
            await client.close()


async def test_snapshot_tool_end_to_end_signed():
    from twin.shared.tools.approval_context import (
        ApprovalRequestContext,
        clear_current_approval_context,
        set_current_approval_context,
    )

    async with _test_client(_march7_server().build_app()) as raw:
        base_url = _base_url(raw)
        set_current_approval_context(
            ApprovalRequestContext(platform="test", user_id="123")
        )
        try:
            tool = March7SnapshotTool(march7_url=base_url, secret=SECRET)
            out = await tool.execute(user_id="123")
            assert "trống" in out or "T1 của March7" in out
        finally:
            clear_current_approval_context()


# ---------------------------------------------------------------------------
# Cancellation (finding #14)
# ---------------------------------------------------------------------------


async def test_cancel_stops_blocking_worker_without_late_completion():
    release = asyncio.Event()
    effects: list[str] = []

    async def blocking(params):
        await release.wait()
        effects.append("ran-after-cancel")
        yield A2AMessage(role="agent", parts=[Part(type="text", text="late")])

    store = TaskStore()
    server = _evernight_server(task_store=store)
    server.skill_handlers["chat"] = blocking
    async with _test_client(server.build_app()) as raw:
        base_url = _base_url(raw)
        client = A2AClient(base_url=base_url, actor="march7", secret=SECRET)
        try:
            task = await client.send_task(
                {"id": "block-1", "skill": "chat", "sessionId": "1"}
            )
            assert task.status.value == "in_progress"
            for _ in range(100):
                rec_worker = store._records["block-1"].worker
                if rec_worker is not None:
                    break
                await asyncio.sleep(0.01)
            cancelled = await client.cancel_task("block-1")
            assert cancelled.status.value == "cancelled"
            worker = store._records["block-1"].worker
            assert worker is not None and worker.done()
            release.set()
            await asyncio.sleep(0.2)
            final = await client.get_task("block-1")
            assert final.status.value == "cancelled"
            assert effects == []
        finally:
            await client.close()


async def test_store_shutdown_cancels_live_workers():
    release = asyncio.Event()
    started = asyncio.Event()

    async def worker_body():
        started.set()
        await release.wait()

    store = TaskStore()
    store.create(
        task_id="w1", skill="chat", session_id="1", creator_peer="march7"
    )
    worker = asyncio.create_task(worker_body())
    store.set_worker("w1", worker)
    await started.wait()
    await store.shutdown(timeout=1.0)
    assert worker.done()
    assert store.get("w1").status.value == "cancelled"


# ---------------------------------------------------------------------------
# Bounded retention (finding #15)
# ---------------------------------------------------------------------------


async def test_active_task_bound_rejects_overflow():
    release = asyncio.Event()

    async def blocking(params):
        await release.wait()
        yield A2AMessage(role="agent", parts=[Part(type="text", text="x")])

    server = _evernight_server(task_store=TaskStore(max_active_tasks=1))
    server.skill_handlers["chat"] = blocking
    async with _test_client(server.build_app()) as client:
        first = _rpc_body(
            "tasks/send", {"id": "a1", "skill": "chat", "sessionId": "1"}
        )
        status, data = await _post_rpc(
            client, first, _signed("march7", "POST", "/", first)
        )
        assert status == 200 and "result" in data
        second = _rpc_body(
            "tasks/send", {"id": "a2", "skill": "chat", "sessionId": "1"}
        )
        status, data = await _post_rpc(
            client, second, _signed("march7", "POST", "/", second)
        )
        assert data["error"]["code"] == -32004
        release.set()


async def test_completed_tasks_evict_oldest_without_touching_active():
    store = TaskStore(max_completed_tasks=2)
    server = _evernight_server(task_store=store)
    async with _test_client(server.build_app()) as raw:
        base_url = _base_url(raw)
        client = A2AClient(base_url=base_url, actor="march7", secret=SECRET)
        try:
            live = store.create(
                task_id="live", skill="chat", session_id="1",
                creator_peer="march7",
            )
            assert live is not None
            for i in range(3):
                await client.send_text_task(
                    skill="chat", session_id="1", text="hi", task_id=f"q{i}"
                )
            for _ in range(100):
                done = store.get("q2")
                if done is not None and done.status.value in (
                    "completed",
                    "failed",
                ):
                    break
                await asyncio.sleep(0.02)
            assert store.get("q0") is None
            assert store.get("q1") is not None
            assert store.get("q2") is not None
            assert store.get("live") is not None
        finally:
            await client.close()


def test_completed_ttl_evicts_even_when_subscribed():
    """Finding #15: TTL bounds private data even if subscribed/idle."""
    now = [1000.0]
    store = TaskStore(completed_ttl_seconds=60.0, clock=lambda: now[0])
    store.create(task_id="t1", skill="chat", session_id="1", creator_peer="m")
    store.complete("t1")
    store.create(task_id="t2", skill="chat", session_id="1", creator_peer="m")
    store.complete("t2")
    queue = store.subscribe("t2")
    now[0] += 61.0
    assert store.purge_expired() == 2
    assert store.get("t1") is None
    assert store.get("t2") is None
    # Expired stream was closed explicitly so SSE unblocks and releases.
    assert queue.get_nowait() is None
    assert store.subscriber_count("t2") == 0


def test_output_buffer_bytes_are_bounded():
    store = TaskStore(max_task_buffer_bytes=500, max_task_buffer_messages=1000)
    store.create(task_id="b1", skill="chat", session_id="1", creator_peer="m")
    for _ in range(50):
        store.publish(
            "b1", A2AMessage(role="agent", parts=[Part(type="text", text="x" * 100)])
        )
    buffered = store.snapshot("b1")
    assert len(buffered) < 50
    total = sum(len(p.text or "") for m in buffered for p in m.parts)
    assert total <= 500 + 100


# ---------------------------------------------------------------------------
# DM routes inherit middleware protection
# ---------------------------------------------------------------------------


async def test_appended_dm_routes_require_allowed_peer():
    async def fake_dm(request: web.Request) -> web.Response:
        return web.json_response({"success": True})

    app = _evernight_server().build_app()
    app.router.add_post("/dm", fake_dm)
    app.router.add_post("/approval_dm", fake_dm)
    async with _test_client(app) as client:
        body = json.dumps({"user_id": 1, "content": "hi"}).encode()
        async with client.post("/dm", data=body) as resp:
            assert resp.status == 401
        async with client.post(
            "/dm", data=body, headers=_signed("march7", "POST", "/dm", body)
        ) as resp:
            assert resp.status == 200
        async with client.post(
            "/dm", data=body, headers=_signed("owner", "POST", "/dm", body)
        ) as resp:
            assert resp.status == 200
        async with client.post(
            "/dm", data=body, headers=_signed("evernight", "POST", "/dm", body)
        ) as resp:
            assert resp.status == 403
        async with client.post(
            "/approval_dm", data=body,
            headers=_signed("march7", "POST", "/approval_dm", body),
        ) as resp:
            assert resp.status == 200
