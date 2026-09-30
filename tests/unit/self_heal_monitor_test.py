"""Tests for SelfHealMonitor owner-approval-gated recovery."""
from __future__ import annotations

import pathlib

import aiohttp
import pytest

from twin.evernight.self_heal.monitor import (
    DenyRecoveryExecutor,
    OwnerApprovalRecoveryExecutor,
    RecoveryExecutor,
    SelfHealMonitor,
)
from twin.evernight.host_gateway.monitor import RESTART_ALLOWED_CONTAINERS
from twin.shared.system_gateway.auth import (
    canonical_approval_action,
    verify_approval_token,
)

APPROVAL_KEY = "test-approval-key-abc123"
REQUEST_KEY = "test-request-key-xyz789"


class FakeGatewayMonitor:
    """Minimal fake for gateway monitor dependency."""

    def __init__(self, *, base_url: str = "http://gw", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.restarts: list[tuple[str, str | None]] = []

    async def request_container_restart(self, name, *, approval_id=None):
        self.restarts.append((name, approval_id))
        if name not in RESTART_ALLOWED_CONTAINERS:
            return False, f"container {name!r} not in allowed list"
        if not approval_id:
            return False, "approval required"
        return True, f"restarted {name}"


class FakeDMBackend:
    def __init__(self, *, approved: bool = True):
        self.approved = approved
        self.calls: list[dict] = []

    async def request_owner_approval(
        self, *, owner_user_id, message_text, label, timeout
    ) -> bool:
        self.calls.append(
            {
                "owner_user_id": owner_user_id,
                "message_text": message_text,
                "label": label,
            }
        )
        return self.approved

    async def send_owner_dm(self, *, owner_user_id, content) -> None:
        return None

    async def notify_channel(self, *, channel_id, approved, label) -> None:
        return None


class FakeRecoveryExecutor(RecoveryExecutor):
    def __init__(self):
        self.calls = []

    async def restart_container(self, container_name: str) -> tuple[bool, str]:
        self.calls.append(container_name)
        return True, f"fake-restarted {container_name}"


@pytest.fixture()
def approval_key(monkeypatch):
    monkeypatch.setattr(
        "twin.evernight.server.approval_issuer.read_approval_secret",
        lambda path=None: APPROVAL_KEY,
    )
    return APPROVAL_KEY


# ---------------------------------------------------------------------------
# Executor selection
# ---------------------------------------------------------------------------


def test_prefers_owner_approval_recovery_when_gateway_monitor_provided():
    monitor = FakeGatewayMonitor()
    shm = SelfHealMonitor(gateway_monitor=monitor, notify_user_id=111)

    assert isinstance(shm._recovery_executor, OwnerApprovalRecoveryExecutor)
    assert shm._recovery_executor.gateway_monitor is monitor


def test_prefers_injected_recovery_executor_over_all():
    monitor = FakeGatewayMonitor()
    injected = FakeRecoveryExecutor()
    shm = SelfHealMonitor(
        gateway_monitor=monitor,
        recovery_executor=injected,
    )

    assert shm._recovery_executor is injected


def test_denies_recovery_when_no_monitor(monkeypatch):
    monkeypatch.delenv("SYSTEM_GATEWAY_URL", raising=False)
    shm = SelfHealMonitor()

    assert isinstance(shm._recovery_executor, DenyRecoveryExecutor)


@pytest.mark.asyncio
async def test_deny_executor_never_restarts():
    ok, detail = await DenyRecoveryExecutor().restart_container("march7")

    assert ok is False
    assert detail


# ---------------------------------------------------------------------------
# OwnerApprovalRecoveryExecutor behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approved_restart_sends_bound_grant(approval_key):
    monitor = FakeGatewayMonitor(timeout=30)
    backend = FakeDMBackend(approved=True)
    executor = OwnerApprovalRecoveryExecutor(
        gateway_monitor=monitor, owner_user_id=111, dm_backend=backend
    )

    ok, detail = await executor.restart_container("march7")

    assert ok is True
    assert "restarted" in detail
    assert len(backend.calls) == 1
    # Owner saw the exact full execution, not a truncated label.
    assert "docker restart march7" in backend.calls[0]["message_text"]
    assert "timeout: 30s" in backend.calls[0]["message_text"]
    assert backend.calls[0]["owner_user_id"] == "111"
    assert len(monitor.restarts) == 1
    grant = monitor.restarts[0][1]
    assert grant
    expected = canonical_approval_action(
        "shell",
        {"command": "docker restart march7", "timeout": 30},
    )
    result = verify_approval_token(
        secret=APPROVAL_KEY, token=grant, action=expected, actor="evernight"
    )
    assert result.valid, result.reason
    # The request key cannot validate an approval grant.
    forged = verify_approval_token(
        secret=REQUEST_KEY, token=grant, action=expected, actor="evernight"
    )
    assert forged.valid is False


@pytest.mark.asyncio
async def test_rejected_approval_never_restarts(approval_key):
    monitor = FakeGatewayMonitor()
    backend = FakeDMBackend(approved=False)
    executor = OwnerApprovalRecoveryExecutor(
        gateway_monitor=monitor, owner_user_id=111, dm_backend=backend
    )

    ok, detail = await executor.restart_container("march7")

    assert ok is False
    assert "denied" in detail
    assert monitor.restarts == []


@pytest.mark.asyncio
async def test_unknown_owner_never_restarts(approval_key):
    monitor = FakeGatewayMonitor()
    backend = FakeDMBackend(approved=True)
    executor = OwnerApprovalRecoveryExecutor(
        gateway_monitor=monitor, owner_user_id=None, dm_backend=backend
    )

    ok, detail = await executor.restart_container("march7")

    assert ok is False
    assert "owner" in detail.lower()
    assert monitor.restarts == []
    assert backend.calls == []


@pytest.mark.asyncio
async def test_unavailable_dm_path_never_restarts(approval_key):
    monitor = FakeGatewayMonitor()
    executor = OwnerApprovalRecoveryExecutor(
        gateway_monitor=monitor, owner_user_id=111, discord_adapter=None
    )

    ok, detail = await executor.restart_container("march7")

    assert ok is False
    assert monitor.restarts == []


@pytest.mark.asyncio
async def test_missing_approval_key_never_restarts(monkeypatch):
    monkeypatch.setattr(
        "twin.evernight.server.approval_issuer.read_approval_secret",
        lambda path=None: None,
    )
    monitor = FakeGatewayMonitor()
    backend = FakeDMBackend(approved=True)
    executor = OwnerApprovalRecoveryExecutor(
        gateway_monitor=monitor, owner_user_id=111, dm_backend=backend
    )

    ok, detail = await executor.restart_container("march7")

    assert ok is False
    assert "approval key" in detail.lower()
    assert monitor.restarts == []


@pytest.mark.asyncio
async def test_rejects_unallowed_container_without_bothering_owner(approval_key):
    monitor = FakeGatewayMonitor()
    backend = FakeDMBackend(approved=True)
    executor = OwnerApprovalRecoveryExecutor(
        gateway_monitor=monitor, owner_user_id=111, dm_backend=backend
    )

    ok, detail = await executor.restart_container("postgres")

    assert ok is False
    assert "not in allowed list" in detail
    assert backend.calls == []
    assert monitor.restarts == []


def test_no_direct_docker_side_channel():
    source = pathlib.Path(
        "twin/evernight/self_heal/monitor.py"
    ).read_text(encoding="utf-8")
    assert "create_subprocess" not in source
    assert "DockerCommandRecoveryExecutor" not in source
    assert "mint_approval_token" not in source


# ---------------------------------------------------------------------------
# Health probing uses /health, never the agent card
# ---------------------------------------------------------------------------


class _FakeHealthResponse:
    def __init__(self, status: int):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeHealthSession:
    last_url: str | None = None
    status: int = 200

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def get(self, url: str):
        type(self).last_url = url
        return _FakeHealthResponse(type(self).status)


@pytest.mark.asyncio
async def test_health_check_uses_health_endpoint(monkeypatch):
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeHealthSession)
    _FakeHealthSession.status = 200
    shm = SelfHealMonitor(march7_url="http://march7:8000")

    assert await shm._check_health() is True
    assert _FakeHealthSession.last_url == "http://march7:8000/health"


@pytest.mark.asyncio
async def test_health_503_is_unhealthy_even_when_card_would_be_200(monkeypatch):
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeHealthSession)
    _FakeHealthSession.status = 503
    shm = SelfHealMonitor(march7_url="http://march7:8000")

    assert await shm._check_health() is False
    assert _FakeHealthSession.last_url == "http://march7:8000/health"
    assert "agent.json" not in (_FakeHealthSession.last_url or "")


# ---------------------------------------------------------------------------
# SelfHealMonitor start / stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_heal_monitor_start_stop():
    shm = SelfHealMonitor()

    assert shm._running is False
    assert shm._task is None

    await shm.start()
    assert shm._running is True
    assert shm._task is not None

    await shm.stop()
    assert shm._running is False
    assert shm._task is None or shm._task.done()


@pytest.mark.asyncio
async def test_self_heal_monitor_stop_is_idempotent():
    shm = SelfHealMonitor()

    await shm.start()
    await shm.stop()
    await shm.stop()  # should not raise

    assert shm._running is False
