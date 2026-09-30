"""Integration tests for the native System Gateway aiohttp app.

These tests exercise the middleware chain directly without binding a real
socket so they run inside restrictive sandboxes. We call handlers via the
auth_middleware using aiohttp's make_mocked_request with a real StreamReader
payload (the public body contract for aiohttp 3.14.3).

The gateway exposes one generic shell-exec path: ``POST /shell/run``. Owner
approval (single-use, canonical-action-bound, actor-bound, verified with the
separate owner approval key) is the security boundary; the approved command
runs verbatim on the platform shell. Shell execution is mocked for
determinism; no test depends on host binaries such as ``uptime``.
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

# Ensure the native gateway package is importable from the repo root.
SERVICES_ROOT = Path(__file__).resolve().parents[2] / "services"
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))

from twin.shared.system_gateway.auth import (
    canonical_approval_action,
    headers_from_signed,
    mint_approval_token,
    sign_request,
)

from system_gateway import server as server_module
from system_gateway.config import GatewayConfig
from system_gateway.server import (
    STATE_KEY,
    capabilities,
    create_app,
    health,
    run_shell,
    self_update,
    auth_middleware,
)
from system_gateway.state import SERVICE_VERSION


REQUEST_SECRET = "test-request-secret"
APPROVAL_SECRET = "test-approval-secret-different"


def _config(**overrides):
    kwargs = dict(
        raw_shell_enabled=True,
        shared_secret=REQUEST_SECRET,
        approval_secret=APPROVAL_SECRET,
    )
    kwargs.update(overrides)
    return GatewayConfig(**kwargs)


def _signed_headers(
    method: str,
    path: str,
    body: bytes,
    *,
    secret: str = REQUEST_SECRET,
    actor: str = "march7",
):
    signed = sign_request(
        secret=secret,
        method=method,
        path=path,
        actor=actor,
        body=body,
    )
    return headers_from_signed(signed)


def _shell_token(
    payload: dict,
    *,
    actor: str = "march7",
    secret: str = APPROVAL_SECRET,
    **kwargs,
):
    action = canonical_approval_action("shell", payload)
    return mint_approval_token(secret=secret, action=action, actor=actor, **kwargs)


def _update_token(
    payload: dict,
    *,
    actor: str = "owner-cli",
    secret: str = APPROVAL_SECRET,
    **kwargs,
):
    action = canonical_approval_action("self.update", payload)
    return mint_approval_token(secret=secret, action=action, actor=actor, **kwargs)


def _build_request(app, method: str, path: str, body: bytes = b"", extra_headers=None):
    """Create a mocked request bound to ``app`` with a real StreamReader body."""

    headers = dict(extra_headers or {})
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - tests always run in a loop
        loop = asyncio.new_event_loop()
    payload = StreamReader(mock.Mock(), limit=2**16, loop=loop)
    if body:
        payload.feed_data(body)
    payload.feed_eof()
    request = make_mocked_request(method, path, headers=headers, payload=payload)
    request._app = app
    return request


class _FakeAdapter:
    """Deterministic shell adapter; records calls, never spawns subprocesses."""

    def __init__(self, result: dict | None = None):
        self.calls: list[dict] = []
        self._result = result or {
            "ok": True,
            "output": "mocked-ok",
            "error": None,
            "exit_code": 0,
            "data": {"shell": "/bin/sh"},
        }

    async def run_shell(
        self,
        command: str,
        *,
        shell=None,
        cwd=None,
        timeout: int = 30,
        max_output_chars: int = 8000,
    ):
        self.calls.append(
            {
                "command": command,
                "shell": shell,
                "cwd": cwd,
                "timeout": timeout,
                "max_output_chars": max_output_chars,
            }
        )
        return dict(self._result)


def _mock_adapter(monkeypatch, result: dict | None = None) -> _FakeAdapter:
    fake = _FakeAdapter(result)
    monkeypatch.setattr(server_module, "select_adapter", lambda *a, **k: fake)
    return fake


def _shell_body(command="echo hello", approval_id="tok", **extra) -> bytes:
    payload = {"command": command, "approval_id": approval_id}
    payload.update(extra)
    return json.dumps(payload).encode()


# --- Public endpoints --------------------------------------------------------


async def test_health_endpoint_is_public():
    app = create_app(_config())
    request = _build_request(app, "GET", "/health")
    response = await health(request)
    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["status"] == "ok"
    assert payload["service"] == "system_gateway"
    assert payload["version"]
    assert payload["platform"]
    assert payload["uptime"] >= 0


async def test_capabilities_endpoint_is_public():
    app = create_app(_config())
    request = _build_request(app, "GET", "/capabilities")
    response = await capabilities(request)
    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["platform"]
    assert payload["raw_shell"] is True
    assert payload["shells"]
    assert payload["structured_actions"] == []
    assert payload["action_details"] == []


async def test_routes_have_no_actions_run():
    app = create_app(_config())
    canonicals = [route.resource.canonical for route in app.router.routes()]
    removed_route = "/actions" + "/run"
    assert removed_route not in canonicals
    assert "/shell/run" in canonicals
    assert "/self/update" in canonicals
    assert not hasattr(server_module, "run_action")


# --- Auth layer --------------------------------------------------------------


async def test_shell_endpoint_rejects_unsigned_request():
    app = create_app(_config())
    request = _build_request(app, "POST", "/shell/run", body=b"{}")
    response = await auth_middleware(request, run_shell)
    assert response.status == 401


async def test_shell_endpoint_rejects_invalid_signature():
    app = create_app(_config())
    body = json.dumps({"command": "echo hello", "approval_id": "fresh"}).encode()
    bad_headers = _signed_headers("POST", "/shell/run", body, secret="wrong-secret")
    request = _build_request(
        app, "POST", "/shell/run", body=body, extra_headers=bad_headers
    )
    response = await auth_middleware(request, run_shell)
    assert response.status == 401


async def test_shell_endpoint_rejects_stale_timestamp():
    app = create_app(_config())
    body = b"{}"
    signed = sign_request(
        secret=REQUEST_SECRET,
        method="POST",
        path="/shell/run",
        actor="march7",
        body=body,
        timestamp=str(int(time.time()) - 10_000),
        nonce="stale-nonce",
    )
    request = _build_request(
        app, "POST", "/shell/run", body=body, extra_headers=headers_from_signed(signed)
    )
    response = await auth_middleware(request, run_shell)
    assert response.status == 401
    payload = json.loads(response.text)
    assert "timestamp" in payload["error"]


async def test_shell_endpoint_rejects_replayed_nonce(monkeypatch):
    app = create_app(_config())
    _mock_adapter(monkeypatch)
    exec_payload: dict = {"command": "echo hello"}
    body = _shell_body(approval_id=_shell_token(exec_payload))
    headers = _signed_headers("POST", "/shell/run", body)

    req1 = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    resp1 = await auth_middleware(req1, run_shell)
    assert resp1.status == 200

    # Same signed headers => same transport nonce => replayed at the auth layer.
    req2 = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    resp2 = await auth_middleware(req2, run_shell)
    assert resp2.status == 401
    payload = json.loads(resp2.text)
    assert "nonce" in payload["error"]


async def test_mutating_path_refuses_when_secret_unset():
    app = create_app(GatewayConfig(shared_secret=None))
    request = _build_request(app, "POST", "/shell/run", body=b"{}")
    response = await auth_middleware(request, run_shell)
    assert response.status == 503


async def test_health_and_capabilities_bypass_auth_middleware():
    """Health and capabilities must not require a signature."""

    app = create_app(GatewayConfig(shared_secret=None))
    request = _build_request(app, "GET", "/health")
    response = await auth_middleware(request, health)
    assert response.status == 200


# --- Policy + approval token -------------------------------------------------


async def test_shell_endpoint_denies_missing_approval_id():
    app = create_app(_config())
    body = json.dumps({"command": "echo hello"}).encode()
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "approval_required"


async def test_shell_endpoint_denies_replayed_approval_id(monkeypatch):
    app = create_app(_config())
    _mock_adapter(monkeypatch)
    exec_payload: dict = {"command": "echo hello"}
    token = _shell_token(exec_payload)
    body = _shell_body(approval_id=token)
    headers1 = _signed_headers("POST", "/shell/run", body)
    req1 = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers1)
    resp1 = await auth_middleware(req1, run_shell)
    assert resp1.status == 200

    # Fresh transport signature but the SAME approval token => replayed token.
    headers2 = _signed_headers("POST", "/shell/run", body)
    req2 = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers2)
    resp2 = await auth_middleware(req2, run_shell)
    assert resp2.status == 403
    assert json.loads(resp2.text)["error"] == "approval_replayed"


async def test_shell_endpoint_rejects_token_minted_with_request_key(monkeypatch):
    # The shared request credential cannot mint a valid owner approval.
    app = create_app(_config())
    _mock_adapter(monkeypatch)
    exec_payload: dict = {"command": "echo hello"}
    forged = _shell_token(exec_payload, secret=REQUEST_SECRET)
    body = _shell_body(approval_id=forged)
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "approval_invalid"


async def test_shell_endpoint_rejects_forged_token():
    app = create_app(_config())
    exec_payload: dict = {"command": "echo hello"}
    forged = _shell_token(exec_payload, secret="wrong-secret")
    body = _shell_body(approval_id=forged)
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "approval_invalid"


async def test_shell_endpoint_rejects_expired_token():
    app = create_app(_config())
    exec_payload: dict = {"command": "echo hello"}
    expired = _shell_token(exec_payload, ttl_seconds=120, now=time.time() - 10_000)
    body = _shell_body(approval_id=expired)
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "approval_invalid"


async def test_shell_endpoint_rejects_actor_mismatched_token():
    app = create_app(_config())
    exec_payload: dict = {"command": "echo hello"}
    token = _shell_token(exec_payload, actor="evernight")
    body = _shell_body(approval_id=token)
    # Request signed as march7 but approval bound to evernight.
    headers = _signed_headers("POST", "/shell/run", body, actor="march7")
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "approval_invalid"


@pytest.mark.parametrize(
    "swapped",
    [
        {"command": "echo swapped"},
        {"shell": "/bin/bash"},
        {"cwd": "/var"},
        {"timeout": 99},
    ],
)
async def test_shell_endpoint_rejects_swapped_execution_fields(swapped):
    app = create_app(_config())
    approved = {"command": "echo hello", "shell": "/bin/sh", "cwd": "/tmp", "timeout": 30}
    token = _shell_token(approved)
    # Same token, but one execution field swapped on the wire.
    mutated = dict(approved, **swapped)
    body = json.dumps(dict(mutated, approval_id=token)).encode()
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "approval_invalid"


async def test_shell_endpoint_denied_when_raw_shell_disabled():
    app = create_app(_config(raw_shell_enabled=False))
    exec_payload: dict = {"command": "echo hello"}
    body = _shell_body(approval_id=_shell_token(exec_payload))
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "raw_shell_disabled"


async def test_shell_endpoint_denies_when_approval_key_missing():
    app = create_app(_config(approval_secret=None))
    exec_payload: dict = {"command": "echo hello"}
    # Even a well-formed token cannot verify without the owner approval key.
    token = _shell_token(exec_payload)
    body = _shell_body(approval_id=token)
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "approval_invalid"


async def test_raw_shell_defaults_to_disabled():
    app = create_app(GatewayConfig(shared_secret=REQUEST_SECRET, approval_secret=APPROVAL_SECRET))
    assert app[STATE_KEY].raw_shell_enabled is False


# --- Execution (mocked adapter, deterministic) -------------------------------


async def test_shell_endpoint_executes_with_mocked_adapter(monkeypatch):
    app = create_app(_config())
    fake = _mock_adapter(monkeypatch)
    exec_payload: dict = {"command": "echo hello"}
    token = _shell_token(exec_payload)
    body = _shell_body(approval_id=token)
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["ok"] is True
    assert payload["output"] == "mocked-ok"
    assert payload["exit_code"] == 0
    assert payload["data"]["shell"]
    assert fake.calls[0]["command"] == "echo hello"


async def test_shell_endpoint_reports_command_failed_on_nonzero_exit(monkeypatch):
    app = create_app(_config())
    _mock_adapter(
        monkeypatch,
        {"ok": False, "output": "", "error": "command_failed", "exit_code": 1, "data": {}},
    )
    exec_payload: dict = {"command": "echo hello"}
    token = _shell_token(exec_payload)
    body = _shell_body(approval_id=token)
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 500
    payload = json.loads(response.text)
    assert payload["ok"] is False
    assert payload["error"] == "command_failed"
    assert payload["exit_code"] == 1


async def test_shell_endpoint_caps_timeout_to_maximum(monkeypatch):
    app = create_app(_config())
    fake = _mock_adapter(monkeypatch)
    exec_payload: dict = {"command": "echo hello", "timeout": 99_999}
    token = _shell_token(exec_payload)
    body = json.dumps(
        {"command": "echo hello", "approval_id": token, "timeout": 99_999}
    ).encode()
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 200
    assert fake.calls[0]["timeout"] == 300
    started = [
        event for event in app[STATE_KEY].audit_log
        if event.get("event") == "action.started"
    ]
    assert started[-1]["details"]["timeout"] == 300


async def test_shell_endpoint_passes_output_limit_to_adapter(monkeypatch):
    app = create_app(_config())
    fake = _mock_adapter(monkeypatch)
    exec_payload: dict = {"command": "echo hello"}
    token = _shell_token(exec_payload)
    # max_output_chars is output-shaping only: changing it does not invalidate
    # the approval, but the server forwards it to the adapter.
    body = json.dumps(
        {"command": "echo hello", "approval_id": token, "max_output_chars": 256}
    ).encode()
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 200
    assert fake.calls[0]["max_output_chars"] == 256


async def test_shell_endpoint_records_audit_event(monkeypatch):
    app = create_app(_config())
    _mock_adapter(monkeypatch)
    exec_payload: dict = {"command": "echo hello"}
    token = _shell_token(exec_payload)
    body = _shell_body(approval_id=token)
    headers = _signed_headers("POST", "/shell/run", body)
    request = _build_request(app, "POST", "/shell/run", body=body, extra_headers=headers)
    response = await auth_middleware(request, run_shell)
    assert response.status == 200
    state = app[STATE_KEY]
    events = {event.get("event") for event in state.audit_log}
    assert "approval.resolved" in events
    assert "action.started" in events
    assert "action.completed" in events
    started = [
        event for event in state.audit_log
        if event.get("event") == "action.started" and event.get("subject") == "shell"
    ]
    assert started, state.audit_log
    event = started[-1]
    assert event["actor"] == "march7"
    assert event["approval_id"] == token


# --- /self/update ------------------------------------------------------------


async def test_self_update_accepts_valid_owner_request():
    app = create_app(_config())
    exec_payload = {"from_version": SERVICE_VERSION, "to_version": "0.2.0"}
    token = _update_token(exec_payload)
    body = json.dumps(dict(exec_payload, approval_id=token)).encode()
    headers = _signed_headers(
        "POST", "/self/update", body, secret=REQUEST_SECRET, actor="owner-cli"
    )
    request = _build_request(app, "POST", "/self/update", body=body, extra_headers=headers)
    response = await auth_middleware(request, self_update)
    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["ok"] is True
    assert "update accepted" in payload["message"]


async def test_self_update_rejects_unsigned_request():
    app = create_app(_config())
    body = json.dumps({"from_version": SERVICE_VERSION}).encode()
    request = _build_request(app, "POST", "/self/update", body=body)
    response = await auth_middleware(request, self_update)
    assert response.status == 401


async def test_self_update_rejects_swapped_target_version():
    app = create_app(_config())
    approved = {"from_version": SERVICE_VERSION, "to_version": "0.2.0"}
    token = _update_token(approved)
    swapped = {"from_version": SERVICE_VERSION, "to_version": "9.9.9", "approval_id": token}
    body = json.dumps(swapped).encode()
    headers = _signed_headers(
        "POST", "/self/update", body, secret=REQUEST_SECRET, actor="owner-cli"
    )
    request = _build_request(app, "POST", "/self/update", body=body, extra_headers=headers)
    response = await auth_middleware(request, self_update)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "approval_invalid"


async def test_self_update_rejects_request_key_minted_token():
    app = create_app(_config())
    exec_payload = {"from_version": SERVICE_VERSION, "to_version": "0.2.0"}
    forged = _update_token(exec_payload, secret=REQUEST_SECRET)
    body = json.dumps(dict(exec_payload, approval_id=forged)).encode()
    headers = _signed_headers(
        "POST", "/self/update", body, secret=REQUEST_SECRET, actor="owner-cli"
    )
    request = _build_request(app, "POST", "/self/update", body=body, extra_headers=headers)
    response = await auth_middleware(request, self_update)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "approval_invalid"


async def test_self_update_denies_when_approval_key_missing():
    app = create_app(_config(approval_secret=None))
    exec_payload = {"from_version": SERVICE_VERSION}
    token = _update_token(exec_payload)
    body = json.dumps(dict(exec_payload, approval_id=token)).encode()
    headers = _signed_headers(
        "POST", "/self/update", body, secret=REQUEST_SECRET, actor="owner-cli"
    )
    request = _build_request(app, "POST", "/self/update", body=body, extra_headers=headers)
    response = await auth_middleware(request, self_update)
    assert response.status == 403
    assert json.loads(response.text)["error"] == "approval_invalid"


async def test_self_update_rejects_version_mismatch():
    app = create_app(_config())
    exec_payload = {"from_version": "not-the-real-version"}
    token = _update_token(exec_payload)
    body = json.dumps(dict(exec_payload, approval_id=token)).encode()
    headers = _signed_headers(
        "POST", "/self/update", body, secret=REQUEST_SECRET, actor="owner-cli"
    )
    request = _build_request(app, "POST", "/self/update", body=body, extra_headers=headers)
    response = await auth_middleware(request, self_update)
    assert response.status == 409
    payload = json.loads(response.text)
    assert payload["error"] == "version_mismatch"


async def test_self_update_rejects_replayed_token():
    app = create_app(_config())
    exec_payload = {"from_version": SERVICE_VERSION}
    token = _update_token(exec_payload)
    body = json.dumps(dict(exec_payload, approval_id=token)).encode()
    headers1 = _signed_headers(
        "POST", "/self/update", body, secret=REQUEST_SECRET, actor="owner-cli"
    )
    req1 = _build_request(app, "POST", "/self/update", body=body, extra_headers=headers1)
    resp1 = await auth_middleware(req1, self_update)
    assert resp1.status == 200

    headers2 = _signed_headers(
        "POST", "/self/update", body, secret=REQUEST_SECRET, actor="owner-cli"
    )
    req2 = _build_request(app, "POST", "/self/update", body=body, extra_headers=headers2)
    resp2 = await auth_middleware(req2, self_update)
    assert resp2.status == 403
    assert json.loads(resp2.text)["error"] == "approval_replayed"
