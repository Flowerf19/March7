"""Discord approval backend for shared ApprovalGate."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from twin.shared.config.settings import Config
from twin.shared.tools.approval_context import (
    ApprovalRequestContext,
    build_owner_approval_message,
)

if TYPE_CHECKING:
    import discord

logger = logging.getLogger(__name__)


class DiscordApprovalBackend:
    """Render channel approval through owner-bound Discord buttons.

    Only the configured owner's click counts; the requester may be anyone.
    The full command is shown -- never truncated -- and payloads that cannot
    be displayed unambiguously within Discord bounds are denied.
    """

    def __init__(self, owner_user_id: str | int | None = None):
        self._owner_user_id = (
            str(owner_user_id).strip() if owner_user_id not in (None, "") else ""
        )

    def _resolve_owner(self) -> str:
        if self._owner_user_id:
            return self._owner_user_id
        return str(getattr(Config, "EVERNIGHT_OWNER_USER_ID", "") or "").strip()

    async def request_channel_approval(
        self,
        context: ApprovalRequestContext,
        command: str,
    ) -> bool:
        from gateway.adapters.discord.views.approve_view import ApproveView

        owner = self._resolve_owner()
        if not owner:
            logger.warning("Channel approval denied: owner is not configured")
            return False

        message = context.native_message
        if message is None or not hasattr(message, "channel"):
            return False

        prompt, reason = build_owner_approval_message(
            str(command or ""),
            channel_name=context.channel_name,
            timeout_seconds=30,
        )
        if prompt is None:
            logger.warning("Channel approval denied: %s", reason)
            return False

        view = ApproveView(command=str(command or ""), owner_user_id=owner)
        sent_msg = await message.channel.send(prompt, view=view)

        try:
            return await view.wait_for_decision()
        finally:
            try:
                await sent_msg.delete()
            except Exception:
                pass


def build_discord_approval_context(
    message: "discord.Message",
    *,
    approval_backend: DiscordApprovalBackend | None = None,
) -> ApprovalRequestContext:
    """Build a neutral approval context from a Discord message."""
    channel = message.channel
    guild = message.guild

    channel_name = str(channel)
    if getattr(channel, "name", None):
        channel_name = f"#{channel.name}"
    if guild:
        channel_name = f"{guild.name}/{channel_name}"

    channel_id = str(channel.id)
    return ApprovalRequestContext(
        platform="discord",
        user_id=str(message.author.id),
        conversation_id=channel_id,
        channel_id=channel_id,
        message_id=str(message.id),
        channel_name=channel_name,
        space_id=str(guild.id) if guild else None,
        user_name=message.author.display_name,
        native_message=message,
        approval_backend=approval_backend or DiscordApprovalBackend(),
    )
