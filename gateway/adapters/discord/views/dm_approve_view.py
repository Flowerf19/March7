"""
DMApproveView — Discord View gửi qua DM để owner approve lệnh host.

Thin wrapper around ApproveView with longer timeout (60s) and DM-specific label.
Ownership is mandatory: unknown/empty owner denies every interaction.
"""

from __future__ import annotations

from gateway.adapters.discord.views.approve_view import ApproveView

DM_APPROVAL_TIMEOUT = 60


class DMApproveView(ApproveView):
    """Approval View sent via DM from Evernight bot."""

    def __init__(
        self,
        command: str,
        *,
        owner_user_id: str | int | None,
        timeout: float = DM_APPROVAL_TIMEOUT,
    ):
        super().__init__(
            command,
            owner_user_id=owner_user_id,
            timeout=timeout,
            context_label="!9 muốn chạy lệnh trên host:",
        )
