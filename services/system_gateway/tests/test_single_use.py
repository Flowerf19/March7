"""Durable single-use + credential separation at the server boundary.

All ledger paths are private tmp files passed explicitly; no test touches the
real host state. Handlers are exercised via auth_middleware without binding a
socket; shell execution is faked.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest import mock

import pytest
from aiohttp.streams import StreamReader
from aiohttp.test_utils import make_mocked_request

INTEGRATION_ROOT = Path(__file__).resolve().parents[3]
if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from twin.shared.system_gateway.auth import (
    canonical_approval_action,
    headers_from_signed,
    mint_approval_token,
    sign_request,
)

import system_gateway.server as server_module
from system_gateway.config import GatewayConfig
from system_gateway.server import auth_middleware, create_app, run_shell, self_update
from system_gateway.state import SERVICE_VERSION

REQUEST_SECRET = "test-request-secret-durable"
APPROVAL_SECRET = "test-approval-secret-durable-different"


def _config(ledger_path: Path, **overrides):
    kwargs = dict(
        raw_shell_enabled=True,
        shared_secret=REQUEST_SECRET,
        approval_secret=APPROVAL_SECRET,
        approval_ledger_file=ledger_path,
    )
    kwargs.update(overrides)
    return GatewayConfig(**kwargs)


def _signed(method: str, path: str, body: bytes, *, secret=REQUEST_SECRET, actor="march7"):
    return headers_from_signed(
        sign_request(secret=secret, method=method, path=path, actor=actor, body=body)
    )


def _shell_token(payload: dict, *, actor="march7", secret=APPROVAL_SECRET, **kwargs):
    return mint_approval_token(
        secret=secret,
        action=canonical_approval_action("shell", payload),
        actor=actor,
        **kwargs,
    )


def _build(app, method: str, path: str, body: bytes, headers):
    loop = asyncio.get_running_loop()
    payload = StreamReader(mock.Mock(), limit=2**16, loop=loop)
    if body:
        payload.feed_data(body)
    payload.feed_eof()
    request = make_mocked_request(method, path, headers=headers, payload=payload)
    request._app = app
    return request


class _FakeAdapter:
    def __init__(self):
        self.calls: list[dict] = []

    async def run_shell(self, command, **kwargs):
        self.calls.append({"command": command, **kwargs})
        return {"ok": True, "output": "mocked-ok", "error": None, "exit_code": 0, "data": {}}


@pytest.mark.asyncio
async def test_normal_grant_executes_once(monkeypatch, tmp_path: Path):
    fake = _FakeAdapter()
    monkeypatch.setattr(server_module, "select_adapter", lambda *a, **k: fake)
    app = create_app(_config(tmp_path / "ledger.db"))
    body = json.dumps(
        {"command": "echo hello", "approval_id": _shell_token({"command": "echo hello"})}
    ).encode()
    req = _build(app, "POST", "/shell/run", body, _signed("POST", "/shell/run", body))
    resp = await auth_middleware(req, run_shell)
    assert resp.status == 200
    assert fake.calls and fake.calls[0]["command"] == "echo hello"


@pytest.mark.asyncio
async def test_restart_replay_denied(tmp_path: Path, monkeypatch):
    fake = _FakeAdapter()
    monkeypatch.setattr(server_module, "select_adapter", lambda *a, **k: fake)
    ledger = tmp_path / "ledger.db"
    token = _shell_token({"command": "echo restart"})
    body = json.dumps({"command": "echo restart", "approval_id": token}).encode()

    app1 = create_app(_config(ledger))
    req1 = _build(app1, "POST", "/shell/run", body, _signed("POST", "/shell/run", body))
    assert (await auth_middleware(req1, run_shell)).status == 200

    # Simulate restart: new app instance sharing the same durable ledger.
    app2 = create_app(_config(ledger))
    req2 = _build(app2, "POST", "/shell/run", body, _signed("POST", "/shell/run", body))
    resp2 = await auth_middleware(req2, run_shell)
    assert resp2.status == 403
    assert json.loads(resp2.text)["error"] == "approval_replayed"
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_self_update_restart_replay_denied(tmp_path: Path):
    ledger = tmp_path / "ledger.db"
    exec_payload = {"from_version": SERVICE_VERSION}
    token = mint_approval_token(
        secret=APPROVAL_SECRET,
        action=canonical_approval_action("self.update", exec_payload),
        actor="owner-cli",
    )
    body = json.dumps(dict(exec_payload, approval_id=token)).encode()

    app1 = create_app(_config(ledger))
    req1 = _build(
        app1,
        "POST",
        "/self/update",
        body,
        _signed("POST", "/self/update", body, actor="owner-cli"),
    )
    assert (await auth_middleware(req1, self_update)).status == 200

    app2 = create_app(_config(ledger))
    req2 = _build(
        app2,
        "POST",
        "/self/update",
        body,
        _signed("POST", "/self/update", body, actor="owner-cli"),
    )
    resp2 = await auth_middleware(req2, self_update)
    assert resp2.status == 403
    assert json.loads(resp2.text)["error"] == "approval_replayed"


@pytest.mark.asyncio
async def test_ledger_failure_prevents_adapter_call(tmp_path: Path, monkeypatch):
    fake = _FakeAdapter()
    monkeypatch.setattr(server_module, "select_adapter", lambda *a, **k: fake)
    blocker = tmp_path / "blocker"
    blocker.write_text("not-a-dir", encoding="utf-8")
    app = create_app(_config(blocker / "ledger.db"))
    body = json.dumps(
        {"command": "echo hi", "approval_id": _shell_token({"command": "echo hi"})}
    ).encode()
    req = _build(app, "POST", "/shell/run", body, _signed("POST", "/shell/run", body))
    resp = await auth_middleware(req, run_shell)
    assert resp.status == 403
    assert json.loads(resp.text)["error"] == "approval_invalid"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_injected_equal_keys_rejected_before_handler(tmp_path: Path):
    with pytest.raises(ValueError):
        _config(tmp_path / "ledger.db", shared_secret="same", approval_secret="same")


@pytest.mark.asyncio
async def test_equal_keys_denied_when_state_injected(tmp_path: Path, monkeypatch):
    # Defense in depth: even a hand-built state with equal credentials denies.
    from system_gateway.state import GatewayState

    fake = _FakeAdapter()
    monkeypatch.setattr(server_module, "select_adapter", lambda *a, **k: fake)
    from system_gateway.approval_ledger import ApprovalLedger

    ledger = ApprovalLedger(tmp_path / "ledger.db")
    app = create_app(_config(tmp_path / "other.db"))
    state = app[server_module.STATE_KEY]
    tampered = GatewayState(
        raw_shell_enabled=True,
        shared_secret="same-cred",
        approval_secret="same-cred",
        approval_ledger=ledger,
    )
    app[server_module.STATE_KEY] = tampered
    exec_payload = {"command": "echo hi"}
    token = mint_approval_token(
        secret="same-cred",
        action=canonical_approval_action("shell", exec_payload),
        actor="march7",
    )
    body = json.dumps(dict(exec_payload, approval_id=token)).encode()
    req = _build(
        app,
        "POST",
        "/shell/run",
        body,
        _signed("POST", "/shell/run", body, secret="same-cred"),
    )
    resp = await auth_middleware(req, run_shell)
    assert resp.status == 403
    assert json.loads(resp.text)["error"] == "approval_invalid"
    assert fake.calls == []
    assert "same-cred" not in resp.text


@pytest.mark.asyncio
async def test_missing_owner_key_denies(tmp_path: Path, monkeypatch):
    fake = _FakeAdapter()
    monkeypatch.setattr(server_module, "select_adapter", lambda *a, **k: fake)
    app = create_app(_config(tmp_path / "ledger.db", approval_secret=None))
    body = json.dumps(
        {"command": "echo hi", "approval_id": _shell_token({"command": "echo hi"})}
    ).encode()
    req = _build(app, "POST", "/shell/run", body, _signed("POST", "/shell/run", body))
    resp = await auth_middleware(req, run_shell)
    assert resp.status == 403
    assert json.loads(resp.text)["error"] == "approval_invalid"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_expired_token_denied_without_claim(tmp_path: Path, monkeypatch):
    fake = _FakeAdapter()
    monkeypatch.setattr(server_module, "select_adapter", lambda *a, **k: fake)
    app = create_app(_config(tmp_path / "ledger.db"))
    expired = _shell_token(
        {"command": "echo hi"}, ttl_seconds=120, now=time.time() - 10_000
    )
    body = json.dumps({"command": "echo hi", "approval_id": expired}).encode()
    req = _build(app, "POST", "/shell/run", body, _signed("POST", "/shell/run", body))
    resp = await auth_middleware(req, run_shell)
    assert resp.status == 403
    assert fake.calls == []
