"""Owner-only System Gateway administration tool for Evernight."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from twin.evernight.host_gateway import installer
from twin.evernight.host_gateway.monitor import GatewayMonitor
from twin.shared.tools.approval_context import get_current_approval_context
from twin.shared.tools.registry.base import BaseTool

if TYPE_CHECKING:
    from twin.shared.tools.approval_gate import ApprovalGate

logger = logging.getLogger(__name__)


class GatewayAdminTool(BaseTool):
    """Owner-only System Gateway administration.

    This tool is visible only to Evernight. Every command verifies that the
    caller's user_id matches the configured owner before doing anything.
    """

    def __init__(
        self,
        owner_user_id: str,
        gateway_monitor: GatewayMonitor | None = None,
        host_gateway_client=None,
        base_url: str | None = None,
        shared_secret: str | None = None,
        timeout: int = 10,
        approval_gate: ApprovalGate | None = None,
    ):
        self.owner_user_id = str(owner_user_id)
        self._gateway_monitor = gateway_monitor
        self._host_gateway_client = host_gateway_client
        self._base_url = base_url
        self._shared_secret = shared_secret
        self._timeout = timeout
        self._approval_gate = approval_gate

    @property
    def name(self) -> str:
        return "gateway_admin"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "doctor", "install_hint", "install", "update"],
                    "description": "Lệnh quản trị gateway.",
                },
                "target_version": {
                    "type": "string",
                    "description": "Phiên bản mục tiêu khi command=update.",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        target_version: str | None = None,
    ) -> str:
        if not self._is_owner():
            return "❌ Lệnh này chỉ dành cho owner."

        cmd = (command or "").strip().lower()
        if cmd == "status":
            if self._gateway_monitor is None:
                return "❌ GatewayMonitor chưa được cấu hình."
            return self._gateway_monitor.status_for_chat()
        if cmd == "doctor":
            return await self._doctor()
        if cmd == "install_hint":
            return self._install_hint()
        if cmd == "install":
            return await self._install()
        if cmd == "update":
            return await self._update(target_version)
        return (
            f"❌ Lệnh `{command}` không hợp lệ. "
            "Các lệnh hỗ trợ: status, doctor, install_hint, install, update."
        )

    def _is_owner(self) -> bool:
        context = get_current_approval_context()
        if context is None:
            logger.warning("gateway_admin: no approval context; rejecting as non-owner")
            return False
        # Trusted platform gate: only the Discord adapter derives owner
        # context. A2A sessionId is peer-chosen and never proves ownership.
        if getattr(context, "platform", None) != "discord":
            logger.warning(
                "gateway_admin: non-discord platform %r rejected as non-owner",
                getattr(context, "platform", None),
            )
            return False
        caller = str(context.user_id or "").strip()
        owner = self.owner_user_id.strip()
        if not caller or not owner:
            return False
        return caller == owner

    async def _doctor(self) -> str:
        if self._gateway_monitor is None:
            return "❌ GatewayMonitor chưa được cấu hình."
        snapshot = await self._gateway_monitor.refresh_once()
        # refresh_once() may return None in tests/fakes; fall back to cached snapshot.
        if snapshot is None:
            snapshot = self._gateway_monitor.snapshot
        # Real snapshots have render_for_chat(); fakes may not — fall back to monitor.
        if hasattr(snapshot, "render_for_chat"):
            rendered = snapshot.render_for_chat()
        else:
            rendered = self._gateway_monitor.status_for_chat()
        lines = [rendered]
        status = getattr(snapshot, "status", None)
        status_value = getattr(status, "value", status)
        if status_value == "missing":
            lines.append("")
            lines.append(self._install_hint())
        return "\n".join(lines)

    def _install_hint(self) -> str:
        return installer.build_bootstrap_hint().render_for_chat()

    async def _install(self) -> str:
        return "\n".join(
            [
                "ℹ️ Legacy bootstrap executor đã bị gỡ. Chạy install command này trên host:",
                "",
                self._install_hint(),
            ]
        )

    async def _update(self, target_version: str | None) -> str:
        if self._gateway_monitor is None:
            return "❌ GatewayMonitor chưa được cấu hình."

        current_version = self._gateway_monitor.snapshot.version
        if current_version is None:
            return "❌ Gateway chưa cài đặt hoặc không phản hồi — không thể update."

        base_url = self._base_url
        if base_url is None and self._host_gateway_client is not None:
            base_url = getattr(self._host_gateway_client, "base_url", None)
        if base_url is None and self._gateway_monitor is not None:
            base_url = getattr(self._gateway_monitor, "base_url", None)
        if base_url is None and self._gateway_monitor is not None:
            client = getattr(self._gateway_monitor, "_client", None)
            if client is not None:
                base_url = getattr(client, "base_url", None)
        if base_url is None:
            return "❌ Gateway base URL chưa được cấu hình."

        secret = self._shared_secret
        if secret is None and self._host_gateway_client is not None:
            secret = getattr(self._host_gateway_client, "shared_secret", None)
        if not secret:
            return "❌ Gateway shared secret chưa được cấu hình."
        if self._approval_gate is None:
            return "❌ Owner approval chưa được cấu hình (DM approval unavailable)."
        # Owner context alone is not consent to a version-changing action.
        # Every update needs an explicit owner button approval bound to the
        # exact current/target versions; this tool only consumes the grant.
        decision = await self._approval_gate.authorize_host(
            action="self.update",
            actor="evernight",
            from_version=current_version,
            to_version=target_version,
        )
        if not decision.approved or not decision.grant:
            reason = decision.reason or "bị từ chối"
            return f"❌ Update bị từ chối bởi owner approval ({reason})."
        approval_id = decision.grant
        actor = "evernight"
        if self._host_gateway_client is not None:
            actor = getattr(self._host_gateway_client, "actor", actor)

        coordinator = installer.InstallerCoordinator(
            base_url=base_url,
            timeout=self._timeout,
            shared_secret=secret,
            actor=actor,
        )

        ok, message = await coordinator.request_update(
            current_version=current_version,
            approval_id=approval_id,
            target_version=target_version,
        )
        if ok:
            return f"✅ Update request accepted: {message}"
        return f"⚠️ Update request failed: {message}"
