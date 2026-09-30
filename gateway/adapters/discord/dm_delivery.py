"""Discord DM delivery backend for owner approvals.

Concrete Discord I/O used by the Evernight approval issuer: owner DMs with
owner-bound approve/reject buttons, and result notifications back to the
source channel. Result notifications never include approval grants/tokens.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import discord

from gateway.adapters.discord.views.dm_approve_view import (
    DM_APPROVAL_TIMEOUT,
    DMApproveView,
)

if TYPE_CHECKING:
    from discord.ext import commands

logger = logging.getLogger(__name__)

# Post-decision channel echo is a short label, never the approval basis (the
# owner already saw the exact full prompt in the DM) and never the grant.
CHANNEL_ECHO_CHARS = 200


class DiscordDMDelivery:
    """Send owner approval DMs through the Evernight Discord bot."""

    def __init__(self, bot: commands.Bot):
        self._bot = bot

    @property
    def available(self) -> bool:
        try:
            return bool(self._bot and self._bot.is_ready())
        except Exception:
            return False

    async def request_owner_approval(
        self,
        *,
        owner_user_id: str,
        message_text: str,
        label: str,
        timeout: float = DM_APPROVAL_TIMEOUT,
    ) -> bool:
        """DM the exact prompt to the owner and wait for their decision."""
        if not self.available:
            logger.warning("Discord bot not available for owner approval DM")
            return False
        try:
            user = await self._bot.fetch_user(int(owner_user_id))
        except Exception:
            logger.exception("Failed to fetch owner user for approval DM")
            return False

        view = DMApproveView(
            command=label, owner_user_id=owner_user_id, timeout=timeout
        )
        try:
            sent_msg = await user.send(message_text, view=view)
        except Exception:
            logger.exception("Failed to send owner approval DM")
            return False
        try:
            return await view.wait_for_decision()
        finally:
            try:
                await sent_msg.delete()
            except Exception:
                pass

    async def send_owner_dm(self, *, owner_user_id: str, content: str) -> None:
        """Send a plain DM to the owner."""
        user = await self._bot.fetch_user(int(owner_user_id))
        await user.send(content)

    async def notify_channel(
        self, *, channel_id: Any, approved: bool, label: str
    ) -> None:
        """Report the decision back to the source channel (no grant echoed)."""
        if not self.available:
            return
        try:
            channel = self._bot.get_channel(channel_id)
            if channel is None:
                channel = await self._bot.fetch_channel(channel_id)
            echo = str(label or "")[:CHANNEL_ECHO_CHARS]
            if approved:
                content = f"\U0001f510 **L\u1ec7nh \u0111\u00e3 \u0111\u01b0\u1ee3c approve:**\n`{echo}`"
            else:
                content = f"\U0001f510 **L\u1ec7nh \u0111\u00e3 b\u1ecb t\u1eeb ch\u1ed1i:**\n`{echo}`"
            await channel.send(content)
        except (discord.Forbidden, discord.NotFound) as exc:
            logger.warning(
                "Skipping approval result notification for inaccessible channel %s: %s",
                channel_id,
                exc,
            )
        except discord.HTTPException as exc:
            logger.warning(
                "Failed to send approval result to channel %s: %s", channel_id, exc
            )
        except Exception:
            logger.exception(
                "Unexpected error sending approval result to channel %s", channel_id
            )
