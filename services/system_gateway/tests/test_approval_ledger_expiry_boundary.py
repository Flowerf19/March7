"""Boundary regression: expiry rechecked at durable claim serialization point.

Covers parent probe (iat90/exp100, verify at 99.99, claim at 100.01 must not
yield [True, True]), exact-expiry, float now, busy-lock wait crossing expiry,
two instances sharing one SQLite file, concurrent initial creation, and
server adapter call-count for crossing expiry. Private tmp paths only, no
sleep, deterministic clock mocks.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

INTEGRATION_ROOT = Path(__file__).resolve().parents[3]
if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from system_gateway.approval_ledger import (
    ApprovalLedger,
    LedgerUnavailable,
    approval_key_fingerprint,
)
from twin.shared.system_gateway.auth import (
    mint_approval_token,
    verify_approval_token,
)

AFP = approval_key_fingerprint("synthetic-owner-key")
SYN_KEY = "synthetic-owner-key"
SYN_ACTION = "synthetic-bound-action"


def _token(*, now: float, ttl: int, nonce: str) -> str:
    return mint_approval_token(
        secret=SYN_KEY,
        action=SYN_ACTION,
        actor="march7",
        now=now,
        ttl_seconds=ttl,
        nonce=nonce,
    )


def test_parent_probe_expired_claims_never_double_win(tmp_path: Path):
    path = tmp_path / "ledger.sqlite"
    first, second = ApprovalLedger(path), ApprovalLedger(path)
    token = _token(now=90, ttl=10, nonce="synthetic-nonce")
    assert verify_approval_token(
        secret=SYN_KEY, token=token, action=SYN_ACTION, actor="march7", now=99.99
    ).valid is True
    assert verify_approval_token(
        secret=SYN_KEY, token=token, action=SYN_ACTION, actor="march7", now=99.99
    ).valid is True
    with patch(
        "system_gateway.approval_ledger.time.time", return_value=100.01
    ):
        claims = [
            x.claim(nonce="synthetic-nonce", expiry=100, key_fp=AFP)
            for x in (first, second)
        ]
    assert claims == [False, False]


def test_exact_expiry_rejected_consistent_with_verify(tmp_path: Path):
    ledger = ApprovalLedger(tmp_path / "ledger.db")
    with patch("system_gateway.approval_ledger.time.time", return_value=100.0):
        assert ledger.claim(nonce="exact", expiry=100, key_fp=AFP) is False
    with patch("system_gateway.approval_ledger.time.time", return_value=99.99):
        assert ledger.claim(nonce="exact-ok", expiry=100, key_fp=AFP) is True
    token = _token(now=90, ttl=10, nonce="syn-exact")
    assert verify_approval_token(
        secret=SYN_KEY, token=token, action=SYN_ACTION, actor="march7", now=99.99
    ).valid is True
    assert verify_approval_token(
        secret=SYN_KEY, token=token, action=SYN_ACTION, actor="march7", now=100.0
    ).valid is False
    assert verify_approval_token(
        secret=SYN_KEY, token=token, action=SYN_ACTION, actor="march7", now=100.01
    ).valid is False


def test_float_now_around_expiry(tmp_path: Path):
    ledger = ApprovalLedger(tmp_path / "ledger.db")
    with patch("system_gateway.approval_ledger.time.time", return_value=99.999):
        assert ledger.claim(nonce="f1", expiry=100, key_fp=AFP) is True
        # Same nonce replay denied even before expiry.
        assert ledger.claim(nonce="f1", expiry=100, key_fp=AFP) is False
    with patch("system_gateway.approval_ledger.time.time", return_value=100.0):
        assert ledger.claim(nonce="f2", expiry=100, key_fp=AFP) is False
    with patch("system_gateway.approval_ledger.time.time", return_value=100.001):
        assert ledger.claim(nonce="f3", expiry=100, key_fp=AFP) is False


def test_malformed_inputs_fail_closed_without_synthesis(tmp_path: Path):
    ledger = ApprovalLedger(tmp_path / "ledger.db")
    for bad_expiry in (None, "", "bad", "100.5x", 0, -5, True, False):
        assert ledger.claim(nonce="m1", expiry=bad_expiry, key_fp=AFP) is False  # type: ignore[arg-type]
    assert ledger.claim(nonce="", expiry=9_999_999_999, key_fp=AFP) is False
    assert ledger.claim(nonce="m1", expiry=9_999_999_999, key_fp="") is False
    assert ledger.claim(nonce="m1", expiry=9_999_999_999, key_fp=None) is False  # type: ignore[arg-type]
    # Nothing was synthesized/inserted; fresh valid claim succeeds.
    assert ledger.claim(nonce="m1", expiry=9_999_999_999, key_fp=AFP) is True


def test_busy_lock_wait_crossing_expiry_denies(tmp_path: Path):
    # Simulates busy-wait crossing expiry: precheck sees 99.99 (valid) but
    # serialization-point recheck sees 100.01 (expired) and must deny.
    ledger = ApprovalLedger(tmp_path / "ledger.db")
    calls = {"n": 0}

    def fake_time(*args, **kwargs):
        calls["n"] += 1
        return 99.99 if calls["n"] == 1 else 100.01

    with patch("system_gateway.approval_ledger.time.time", side_effect=fake_time):
        assert ledger.claim(nonce="cross", expiry=100, key_fp=AFP) is False
    assert calls["n"] >= 2
    # Failed crossing inserted nothing; future claim succeeds.
    assert ledger.claim(nonce="cross", expiry=9_999_999_999, key_fp=AFP) is True


def test_two_instances_same_sqlite_single_winner_valid_zero_expired(tmp_path: Path):
    path = tmp_path / "shared.db"
    first, second = ApprovalLedger(path), ApprovalLedger(path)
    with patch("system_gateway.approval_ledger.time.time", return_value=99.99):
        assert sorted(
            [
                first.claim(nonce="race-valid", expiry=100, key_fp=AFP),
                second.claim(nonce="race-valid", expiry=100, key_fp=AFP),
            ]
        ) == [False, True]
    with patch("system_gateway.approval_ledger.time.time", return_value=100.01):
        assert [
            first.claim(nonce="race-exp", expiry=100, key_fp=AFP),
            second.claim(nonce="race-exp", expiry=100, key_fp=AFP),
        ] == [False, False]


def test_concurrent_initial_creation_no_permanent_denial(tmp_path: Path):
    path = tmp_path / "race" / "ledger.db"
    barrier = threading.Barrier(2)
    results: list[bool] = []

    def create() -> None:
        barrier.wait(timeout=5)
        try:
            ApprovalLedger(path)
            results.append(True)
        except LedgerUnavailable:
            results.append(False)

    threads = [threading.Thread(target=create) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert sorted(results) == [True, True]
    ledger = ApprovalLedger(path)
    assert ledger.claim(nonce="after-race", expiry=9_999_999_999, key_fp=AFP) is True


@pytest.mark.asyncio
async def test_server_crossing_expiry_no_adapter_call(tmp_path: Path, monkeypatch):
    import asyncio
    import json
    import time as real_time
    import types
    from unittest import mock as umock

    from aiohttp.streams import StreamReader
    from aiohttp.test_utils import make_mocked_request

    import system_gateway.approval_ledger as ledger_mod
    import system_gateway.server as server_module
    from system_gateway.config import GatewayConfig
    from system_gateway.server import auth_middleware, create_app, run_shell
    from twin.shared.system_gateway.auth import (
        canonical_approval_action,
        headers_from_signed,
        sign_request,
    )

    request_secret = "test-request-secret-crossing"
    approval_secret = "test-approval-secret-crossing"
    ledger_path = tmp_path / "ledger.db"

    class _FakeAdapter:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def run_shell(self, command, **kwargs):
            self.calls.append({"command": command, **kwargs})
            return {"ok": True, "output": "mocked", "error": None, "exit_code": 0, "data": {}}

    fake = _FakeAdapter()
    monkeypatch.setattr(server_module, "select_adapter", lambda *a, **k: fake)
    app = create_app(
        GatewayConfig(
            raw_shell_enabled=True,
            shared_secret=request_secret,
            approval_secret=approval_secret,
            approval_ledger_file=ledger_path,
        )
    )
    real_now = real_time.time()
    payload = {"command": "echo hi"}
    action = canonical_approval_action("shell", payload)
    token = mint_approval_token(
        secret=approval_secret, action=action, actor="march7",
        now=real_now, ttl_seconds=5, nonce="crossing-nonce",
    )
    body = json.dumps({"command": "echo hi", "approval_id": token}).encode()
    signed = headers_from_signed(
        sign_request(secret=request_secret, method="POST", path="/shell/run",
                     actor="march7", body=body)
    )

    def _build(body_bytes: bytes, headers: dict):
        loop = asyncio.get_running_loop()
        stream = StreamReader(umock.Mock(), limit=2**16, loop=loop)
        if body_bytes:
            stream.feed_data(body_bytes)
        stream.feed_eof()
        req = make_mocked_request("POST", "/shell/run", headers=headers, payload=stream)
        req._app = app
        return req

    future = real_now + 10.0
    fake_clock = types.SimpleNamespace(time=lambda: future)
    with patch.object(ledger_mod, "time", fake_clock):
        resp = await auth_middleware(_build(body, signed), run_shell)
    assert resp.status == 403
    assert fake.calls == []
