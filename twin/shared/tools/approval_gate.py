"""ApprovalGate - guard dangerous tools behind user approval.

Shared approval code is platform-neutral. Native UI such as Discord buttons is
provided by the gateway adapter through an ApprovalBackend, and host grants
are issued by owner-trusted Evernight over authenticated A2A.

Fail-closed everywhere: missing context, missing backend, missing owner, or
an unavailable DM path denies. There is no auto-approve bypass.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from twin.shared.tools.approval_context import (
    ApprovalBackend,
    ApprovalDecision,
    ApprovalRequestContext,
    get_current_approval_context,
)

if TYPE_CHECKING:
    from twin.shared.tools.dm_client import DMClient

logger = logging.getLogger(__name__)


class ApprovalGate:
    """Guard dangerous tools behind a platform approval capability."""

    def __init__(
        self,
        dm_client: Optional["DMClient"] = None,
        channel_approval_backend: ApprovalBackend | None = None,
    ):
        """
        Initialize ApprovalGate.

        Args:
            dm_client: Optional client for owner/private approval.
            channel_approval_backend: Optional fallback backend. Platform
                adapters usually provide this on the current context.
        """
        self.dm_client = dm_client
        self.channel_approval_backend = channel_approval_backend

    async def check_approval(self, tool_name: str, command: str) -> bool:
        logger.warning(f"🔐 APPROVAL REQUESTED: {tool_name} -> {command[:100]}")

        approved = await self._request_user_approval(tool_name, command)

        if approved:
            logger.info(f"✅ APPROVED: {tool_name}")
        else:
            logger.warning(f"❌ REJECTED: {tool_name}")

        return approved

    async def authorize_host(
        self,
        *,
        action: str,
        command: str | None = None,
        shell: str | None = None,
        cwd: str | None = None,
        timeout: int | None = None,
        actor: str,
        from_version: str | None = None,
        to_version: str | None = None,
    ) -> ApprovalDecision:
        """Authorize a host execution through owner-trusted Evernight.

        Returns the structured issuer decision (approved/grant/reason). Host
        tools send the returned grant to the native gateway and never mint.
        DM-only: no channel fallback, since only Evernight holds the approval
        key. Any failure denies.
        """
        from twin.shared.tools.dm_client import DMUnavailableError

        context = get_current_approval_context()
        if context is None:
            logger.warning("Host approval denied: no approval context")
            return ApprovalDecision.denied("no_approval_context")
        if self.dm_client is None:
            logger.warning("Host approval denied: DM approval path unavailable")
            return ApprovalDecision.denied("dm_unavailable")
        try:
            return await self.dm_client.request_authorization(
                action=action,
                command=command,
                shell=shell,
                cwd=cwd,
                timeout=timeout,
                actor=actor,
                from_version=from_version,
                to_version=to_version,
                context=context,
            )
        except DMUnavailableError as exc:
            logger.warning("Host approval denied: Evernight unreachable (%s)", exc)
            return ApprovalDecision.denied(f"dm_unavailable: {exc}")
        except Exception as exc:
            logger.exception("Host approval denied: DM request failed")
            return ApprovalDecision.denied(f"dm_error: {exc}")

    async def _request_user_approval(self, tool_name: str, command: str) -> bool:
        from twin.shared.tools.dm_client import DMUnavailableError

        context = get_current_approval_context()
        if context is None:
            logger.warning("No approval context, rejecting approval request")
            return False

        # Try DM/private approval first if configured.
        if self.dm_client:
            try:
                return await self.dm_client.request_approval(
                    command=command,
                    context=context,
                )
            except DMUnavailableError:
                logger.warning("DM approval failed, falling back to channel approval")
            except Exception:
                logger.warning("DM approval failed, falling back to channel approval")

        return await self._request_channel_approval(context, command)

    async def _request_channel_approval(
        self,
        context: ApprovalRequestContext,
        command: str,
    ) -> bool:
        """Fallback: ask approval through the platform channel backend."""
        backend = context.approval_backend or self.channel_approval_backend
        if backend is None:
            label = f"{context.platform}:{context.conversation_id or context.channel_id or 'unknown'}"
            logger.warning(
                "No approval backend for %s, rejecting approval request", label
            )
            return False

        return await backend.request_channel_approval(context, command)
