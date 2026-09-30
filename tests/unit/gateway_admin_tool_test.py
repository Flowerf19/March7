"""Tests for GatewayAdminTool owner gate and command routing."""
from __future__ import annotations

import pathlib

import pytest

from twin.shared.tools.approval_context import ApprovalDecision
from twin.shared.tools.modules.system.gateway_admin_tool import GatewayAdminTool


class _FakeApprovalContext:
    def __init__(self, user_id: str, platform: str = "discord"):
        self.user_id = user_id
        self.platform = platform


class _FakeGatewayMonitor:
    def __init__(self, *, version: str | None = "0.1.0", base_url: str = "http://gw"):
        self.base_url = base_url
        self.timeout = 30
        self._version = version
        self._snapshot = _FakeSnapshot(version=version)
        self.refresh_calls = 0

    @property
    def snapshot(self):
        return self._snapshot

    async def refresh_once(self):
        self.refresh_calls += 1

    def status_for_chat(self) -> str:
        return "Gateway status line"


class _FakeSnapshot:
    def __init__(self, version: str | None = "0.1.0"):
        self.version = version


class _FakeHostGatewayClient:
    def __init__(self, shared_secret: str | None = None, actor: str = "evernight"):
        self.shared_secret = shared_secret
        self.actor = actor


class _FakeGate:
    def __init__(self, decision: ApprovalDecision):
        self.decision = decision
        self.calls: list[dict] = []

    async def authorize_host(self, **kwargs):
        self.calls.append(kwargs)
        return self.decision


def _ctx(monkeypatch, user_id: str, platform: str = "discord"):
    ctx = _FakeApprovalContext(user_id=user_id, platform=platform)
    monkeypatch.setattr(
        "twin.shared.tools.modules.system.gateway_admin_tool.get_current_approval_context",
        lambda: ctx,
    )
    return ctx


# ---------------------------------------------------------------------------
# Owner gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_owner_gate_rejects_non_owner(monkeypatch):
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=_FakeGatewayMonitor(),
    )
    _ctx(monkeypatch, "someone-else")
    result = await tool.execute(command="status")
    assert "❌" in result
    assert "owner" in result.lower()


@pytest.mark.asyncio
async def test_owner_gate_rejects_when_no_context(monkeypatch):
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=_FakeGatewayMonitor(),
    )
    monkeypatch.setattr(
        "twin.shared.tools.modules.system.gateway_admin_tool.get_current_approval_context",
        lambda: None,
    )
    result = await tool.execute(command="status")
    assert "❌" in result
    assert "owner" in result.lower()


@pytest.mark.asyncio
async def test_owner_gate_allows_owner(monkeypatch):
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=_FakeGatewayMonitor(),
    )
    _ctx(monkeypatch, "owner-123")
    result = await tool.execute(command="status")
    assert "❌" not in result


@pytest.mark.asyncio
async def test_a2a_owner_id_cannot_use_admin(monkeypatch):
    """Peer-chosen A2A sessionId never proves Discord ownership."""
    gate = _FakeGate(ApprovalDecision.approved_with_grant("must-not-be-used"))
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=_FakeGatewayMonitor(),
        host_gateway_client=_FakeHostGatewayClient(shared_secret="s"),
        approval_gate=gate,  # type: ignore[arg-type]
    )
    _ctx(monkeypatch, "owner-123", platform="a2a")

    for cmd in ("status", "doctor", "install_hint", "install", "update"):
        result = await tool.execute(command=cmd, target_version="0.2.0")
        assert "owner" in result.lower()
    assert gate.calls == []


# ---------------------------------------------------------------------------
# Command routing: status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_returns_error_when_monitor_missing(monkeypatch):
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=None,
    )
    _ctx(monkeypatch, "owner-123")
    result = await tool.execute(command="status")
    assert "❌" in result
    assert "GatewayMonitor" in result


@pytest.mark.asyncio
async def test_status_returns_monitor_chat_status(monkeypatch):
    monitor = _FakeGatewayMonitor()
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=monitor,
    )
    _ctx(monkeypatch, "owner-123")
    result = await tool.execute(command="status")
    assert result == "Gateway status line"


# ---------------------------------------------------------------------------
# Command routing: doctor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doctor_refreshes_monitor_and_returns_status(monkeypatch):
    monitor = _FakeGatewayMonitor()
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=monitor,
    )
    _ctx(monkeypatch, "owner-123")
    result = await tool.execute(command="doctor")
    assert monitor.refresh_calls == 1
    assert result == "Gateway status line"


# ---------------------------------------------------------------------------
# Command routing: install_hint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_hint_returns_bootstrap_hint(monkeypatch):
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=None,
    )
    _ctx(monkeypatch, "owner-123")
    result = await tool.execute(command="install_hint")
    assert "System Gateway bootstrap" in result
    assert "```" in result


@pytest.mark.asyncio
async def test_install_returns_manual_bootstrap_hint(monkeypatch):
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=None,
    )
    _ctx(monkeypatch, "owner-123")
    monkeypatch.setenv("SYSTEM_GATEWAY_BOOTSTRAP_REPO_ROOT", "/repo")
    result = await tool.execute(command="install")
    assert "Legacy bootstrap executor" in result
    assert "System Gateway bootstrap" in result
    assert "cd /path/to/march7" in result
    assert "scripts/bootstrap_system_gateway.py" in result


@pytest.mark.asyncio
async def test_install_still_requires_owner(monkeypatch):
    tool = GatewayAdminTool(owner_user_id="owner-123")
    _ctx(monkeypatch, "someone-else")
    result = await tool.execute(command="install")
    assert "owner" in result.lower()


# ---------------------------------------------------------------------------
# Command routing: update (structured owner approval, consume-only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_returns_error_when_monitor_missing(monkeypatch):
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=None,
        approval_gate=_FakeGate(ApprovalDecision.denied("x")),  # type: ignore[arg-type]
    )
    _ctx(monkeypatch, "owner-123")
    result = await tool.execute(command="update")
    assert "❌" in result
    assert "GatewayMonitor" in result


@pytest.mark.asyncio
async def test_update_returns_error_when_gateway_version_unknown(monkeypatch):
    monitor = _FakeGatewayMonitor(version=None)
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=monitor,
        approval_gate=_FakeGate(ApprovalDecision.denied("x")),  # type: ignore[arg-type]
    )
    _ctx(monkeypatch, "owner-123")
    result = await tool.execute(command="update")
    assert "❌" in result
    assert "chưa cài đặt" in result


@pytest.mark.asyncio
async def test_update_consumes_structured_grant(monkeypatch):
    from twin.evernight.server.approval_issuer import issue_self_update_grant

    monitor = _FakeGatewayMonitor(version="0.1.0")
    client = _FakeHostGatewayClient(shared_secret="test-secret")
    grant = issue_self_update_grant(
        actor="evernight",
        from_version="0.1.0",
        to_version="0.2.0",
        approval_secret="approval-key",
    )
    gate = _FakeGate(ApprovalDecision.approved_with_grant(grant))
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=monitor,
        host_gateway_client=client,
        approval_gate=gate,  # type: ignore[arg-type]
    )
    _ctx(monkeypatch, "owner-123")

    coordinator_calls = []

    class _FakeCoordinator:
        def __init__(self, base_url, timeout, shared_secret=None, actor="evernight"):
            coordinator_calls.append({
                "base_url": base_url,
                "timeout": timeout,
                "shared_secret": shared_secret,
                "actor": actor,
            })

        async def request_update(self, current_version, target_version=None, approval_id=None):
            coordinator_calls.append({
                "current_version": current_version,
                "target_version": target_version,
                "approval_id": approval_id,
            })
            return True, "update queued"

    monkeypatch.setattr(
        "twin.evernight.host_gateway.installer.InstallerCoordinator",
        _FakeCoordinator,
    )

    result = await tool.execute(command="update", target_version="0.2.0")
    assert "✅" in result
    assert gate.calls[0]["action"] == "self.update"
    assert gate.calls[0]["actor"] == "evernight"
    assert gate.calls[0]["from_version"] == "0.1.0"
    assert gate.calls[0]["to_version"] == "0.2.0"
    assert coordinator_calls[1]["approval_id"] == grant

    from twin.shared.system_gateway.auth import (
        canonical_approval_action,
        verify_approval_token,
    )

    expected = canonical_approval_action(
        "self.update", {"from_version": "0.1.0", "to_version": "0.2.0"}
    )
    assert verify_approval_token(
        secret="approval-key", token=grant, action=expected, actor="evernight"
    ).valid is True
    # Request key cannot validate; tampered versions/actor fail.
    assert verify_approval_token(
        secret="test-secret", token=grant, action=expected, actor="evernight"
    ).valid is False
    tampered = canonical_approval_action(
        "self.update", {"from_version": "0.1.0", "to_version": "9.9.9"}
    )
    assert verify_approval_token(
        secret="approval-key", token=grant, action=tampered, actor="evernight"
    ).valid is False
    assert verify_approval_token(
        secret="approval-key", token=grant, action=expected, actor="march7"
    ).valid is False


@pytest.mark.asyncio
async def test_update_denied_timeout_or_missing_grant_never_executes(monkeypatch):
    for decision in (
        ApprovalDecision.denied("rejected_or_timeout"),
        ApprovalDecision.denied("dm_unavailable"),
        ApprovalDecision(approved=True, grant=None, reason="no-grant"),
    ):
        monitor = _FakeGatewayMonitor(version="0.1.0")
        gate = _FakeGate(decision)
        tool = GatewayAdminTool(
            owner_user_id="owner-123",
            gateway_monitor=monitor,
            host_gateway_client=_FakeHostGatewayClient(shared_secret="s"),
            approval_gate=gate,  # type: ignore[arg-type]
        )
        _ctx(monkeypatch, "owner-123")

        class _FailingCoordinator:
            def __init__(self, *args, **kwargs):
                raise AssertionError("must not construct native client on denial")

            async def request_update(self, *args, **kwargs):
                raise AssertionError("must not request native update on denial")

        monkeypatch.setattr(
            "twin.evernight.host_gateway.installer.InstallerCoordinator",
            _FailingCoordinator,
        )
        result = await tool.execute(command="update", target_version="0.2.0")
        assert result.startswith("❌")
        assert len(gate.calls) == 1


@pytest.mark.asyncio
async def test_update_without_shared_secret_fails_before_dm(monkeypatch):
    monitor = _FakeGatewayMonitor(version="0.1.0")
    gate = _FakeGate(ApprovalDecision.approved_with_grant("g"))
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=monitor,
        host_gateway_client=_FakeHostGatewayClient(shared_secret=None),
        approval_gate=gate,  # type: ignore[arg-type]
    )
    _ctx(monkeypatch, "owner-123")
    result = await tool.execute(command="update")
    assert "shared secret" in result
    assert gate.calls == []


@pytest.mark.asyncio
async def test_update_without_gate_fails_closed(monkeypatch):
    monitor = _FakeGatewayMonitor(version="0.1.0")
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=monitor,
        host_gateway_client=_FakeHostGatewayClient(shared_secret="s"),
        approval_gate=None,
    )
    _ctx(monkeypatch, "owner-123")

    class _FailingCoordinator:
        def __init__(self, *args, **kwargs):
            raise AssertionError("must not construct native client without gate")

        async def request_update(self, *args, **kwargs):
            raise AssertionError("must not request native update without gate")

    monkeypatch.setattr(
        "twin.evernight.host_gateway.installer.InstallerCoordinator",
        _FailingCoordinator,
    )
    result = await tool.execute(command="update")
    assert "approval" in result.lower()


def test_shared_tool_consumes_grant_never_mints():
    source = pathlib.Path(
        "twin/shared/tools/modules/system/gateway_admin_tool.py"
    ).read_text(encoding="utf-8")
    assert "approval_issuer" not in source
    assert "issue_self_update_grant" not in source
    assert "mint_approval_token" not in source
    assert "read_approval_secret" not in source
    assert "authorize_host" in source


# ---------------------------------------------------------------------------
# Command routing: invalid command
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_command_returns_error(monkeypatch):
    tool = GatewayAdminTool(
        owner_user_id="owner-123",
        gateway_monitor=_FakeGatewayMonitor(),
    )
    _ctx(monkeypatch, "owner-123")
    result = await tool.execute(command="invalid_cmd")
    assert "❌" in result
    assert "không hợp lệ" in result
    assert "status" in result
