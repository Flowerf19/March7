"""HostSystemTool - capability-based host interaction through System Gateway."""
from __future__ import annotations

import json
import logging
from typing import Any

from twin.shared.system_gateway import (
    GatewayShellRequest,
    HostGatewayClient,
    HostGatewayError,
    HostGatewayUnavailableError,
)
from twin.shared.tools.approval_gate import ApprovalGate
from twin.shared.tools.registry.base import BaseTool, ToolExecutionError

logger = logging.getLogger(__name__)


class HostSystemTool(BaseTool):
    """Capability-based host interaction via the native System Gateway."""

    def __init__(
        self,
        approval_gate: ApprovalGate,
        host_gateway_client: HostGatewayClient | None = None,
        host_gateway_timeout: int = 30,
    ):
        self.approval_gate = approval_gate
        self.host_gateway_client = host_gateway_client
        self.timeout = host_gateway_timeout

    @property
    def name(self) -> str:
        return "host_system"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["capabilities", "shell"],
                    "description": "Kiểu yêu cầu: capabilities (xem OS/shell) hoặc shell (chạy lệnh).",
                },
                "command": {
                    "type": "string",
                    "description": "Lệnh raw shell khi mode=shell. Viết lệnh phù hợp OS từ /capabilities.",
                },
                "shell": {
                    "type": "string",
                    "description": "Shell mong muốn nếu gateway hỗ trợ, ví dụ bash/zsh/powershell.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Thư mục làm việc cho shell.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout giây, mặc định 30, tối đa 120.",
                },
            },
            "required": ["mode"],
        }

    async def execute(
        self,
        mode: str,
        command: str | None = None,
        shell: str | None = None,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> str:
        if not self.host_gateway_client:
            return self._render_needs_install()

        timeout = max(5, min(120, timeout or self.timeout))
        mode = (mode or "").strip().lower()

        try:
            if mode == "capabilities":
                return await self._render_capabilities()
            if mode == "shell":
                return await self._run_shell(command, shell, cwd, timeout)
            return "Lỗi: mode phải là capabilities hoặc shell."
        except HostGatewayUnavailableError:
            return self._render_needs_install()
        except HostGatewayError as exc:
            return f"❌ System Gateway lỗi: {exc}"
        except Exception as exc:
            logger.error("Unexpected host_system error: %s", exc)
            raise ToolExecutionError(self.name, f"Lỗi không xác định: {exc}", exc)

    async def _render_capabilities(self) -> str:
        capabilities = await self.host_gateway_client.capabilities()
        lines = [
            "🖥️ **System Gateway capabilities**",
            f"- Platform: {capabilities.platform}",
            f"- Raw shell: {'enabled' if capabilities.raw_shell else 'disabled'}",
        ]
        if capabilities.shells:
            lines.append(f"- Shells: {', '.join(capabilities.shells)}")
        if capabilities.features:
            lines.append(f"- Features: {', '.join(capabilities.features)}")
        if capabilities.structured_actions:
            lines.append(f"- Actions: {', '.join(capabilities.structured_actions)}")
        if capabilities.unsupported:
            lines.append(f"- Unsupported: {', '.join(capabilities.unsupported)}")
        if capabilities.notes:
            lines.append(f"- Notes: {'; '.join(capabilities.notes)}")
        return "\n".join(lines)

    async def _run_shell(
        self,
        command: str | None,
        shell: str | None,
        cwd: str | None,
        timeout: int,
    ) -> str:
        if not command or not command.strip():
            return "Lỗi: mode=shell cần tham số command."

        capabilities = await self.host_gateway_client.capabilities()
        if not capabilities.raw_shell:
            return "❌ Shell execution đang bị tắt trên System Gateway (raw_shell=false)."

        clean_command = command.strip()
        actor = getattr(self.host_gateway_client, "actor", "march7") or "march7"
        if hasattr(self.host_gateway_client, "shared_secret") and not self.host_gateway_client.shared_secret:
            return "❌ System Gateway shared secret chưa được cấu hình (request-signing credential missing)."
        # March7 never mints: the grant below is issued by owner-trusted
        # Evernight (holder of the separate approval key) only after an
        # actual owner approval, bound to this exact execution + actor.
        decision = await self.approval_gate.authorize_host(
            action="shell",
            command=clean_command,
            shell=shell,
            cwd=cwd,
            timeout=timeout,
            actor=actor,
        )
        if not decision.approved or not decision.grant:
            reason = decision.reason or "bị từ chối"
            return f"❌ Lệnh host_system bị từ chối bởi Trạm Gác ({reason})."

        approval_id = decision.grant

        response = await self.host_gateway_client.run_shell(
            GatewayShellRequest(
                command=clean_command,
                shell=shell,
                cwd=cwd,
                timeout=timeout,
                approval_id=approval_id,
            )
        )
        return self._format_response(clean_command, response.ok, response.output, response.error)

    def _render_needs_install(self) -> str:
        from twin.evernight.host_gateway.installer import build_bootstrap_hint

        hint = build_bootstrap_hint()
        return json.dumps(
            {
                "needs_install": True,
                "platform": hint.platform,
                "bootstrap_command": hint.command,
                "notes": list(hint.notes),
            },
            ensure_ascii=False,
            indent=2,
        )

    def _format_response(
        self,
        label: str,
        ok: bool,
        output: str,
        error: str | None,
    ) -> str:
        lines = [f"🖥️ **Host:** `{label[:100]}`"]
        if output:
            lines.append("```")
            lines.append(self._truncate(output, 4000))
            lines.append("```")
        if error:
            lines.append(f"⚠️ {error}")
        lines.append("✅ Thành công" if ok else "⚠️ Không thành công")
        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return f"{text[:half]}\n... [truncated: {len(text)} chars total] ...\n{text[-half:]}"
