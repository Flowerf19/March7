"""
ApproveView - Discord View yêu cầu owner approve trước khi chạy lệnh host.

Hiện ra khi ApprovalGate cần xác nhận cho một thao tác có side effect.
Có 2 nút: [Approve] cho phép chạy, [Reject] từ chối. Timeout 30s.

Chỉ interaction của owner đã cấu hình mới được tính. Requester (người gửi
lệnh gốc) có thể không phải owner, nên view không bao giờ suy luận quyền
duyệt từ requester -- chỉ `interaction.user.id` mới có giá trị.
"""

from __future__ import annotations

import asyncio
import logging

import discord

logger = logging.getLogger(__name__)

APPROVAL_TIMEOUT = 30


class ApproveView(discord.ui.View):
    def __init__(
        self,
        command: str,
        *,
        owner_user_id: str | int | None,
        timeout: int = APPROVAL_TIMEOUT,
        context_label: str = "",
    ):
        super().__init__(timeout=timeout)
        self.command = command
        self._timeout = timeout
        self._context_label = context_label
        self._owner_user_id = (
            str(owner_user_id).strip() if owner_user_id not in (None, "") else ""
        )
        self._result: asyncio.Event = asyncio.Event()
        self._approved: bool = False

    def _disable_all(self):
        for child in self.children:
            child.disabled = True

    def _is_owner(self, interaction: discord.Interaction) -> bool:
        """True only when the clicking user is the configured owner."""
        if not self._owner_user_id:
            return False
        user = getattr(interaction, "user", None)
        user_id = getattr(user, "id", None)
        if user_id is None:
            return False
        return str(user_id) == self._owner_user_id

    async def _deny_interaction(self, interaction: discord.Interaction) -> None:
        try:
            await interaction.response.send_message(
                "❌ Chỉ owner mới được duyệt lệnh host.", ephemeral=True
            )
        except discord.HTTPException:
            logger.warning("Failed to send approval denial notice")
        except Exception:
            logger.exception("Unexpected error denying non-owner interaction")

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self._is_owner(interaction):
            return True
        await self._deny_interaction(interaction)
        return False

    async def _edit_interaction_message(
        self,
        interaction: discord.Interaction,
        content: str,
    ) -> None:
        try:
            await interaction.response.edit_message(content=content, view=self)
        except discord.NotFound:
            logger.warning("Approval message disappeared before it could be updated")
        except discord.HTTPException as exc:
            logger.warning("Failed to update approval message: %s", exc)

    @discord.ui.button(label="✅ Approve", style=discord.ButtonStyle.green)
    async def approve_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Re-verify here: interaction_check can be bypassed by direct callback
        # invocation, so the callback itself is the authority boundary.
        if not self._is_owner(interaction):
            await self._deny_interaction(interaction)
            return
        self._approved = True
        self._result.set()
        self._disable_all()
        await self._edit_interaction_message(
            interaction,
            "✅ **Đã approve lệnh host.**\nĐang thực thi...",
        )

    @discord.ui.button(label="❌ Reject", style=discord.ButtonStyle.red)
    async def reject_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._is_owner(interaction):
            await self._deny_interaction(interaction)
            return
        self._approved = False
        self._result.set()
        self._disable_all()
        await self._edit_interaction_message(
            interaction,
            "❌ **Đã từ chối lệnh host.**",
        )

    async def wait_for_decision(self) -> bool:
        try:
            await asyncio.wait_for(self._result.wait(), timeout=self._timeout)
            return self._approved
        except asyncio.TimeoutError:
            self._disable_all()
            return False

    async def on_timeout(self):
        self._disable_all()
