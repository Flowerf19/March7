"""Platform-neutral chat handler for unified gateway messages."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, TYPE_CHECKING

from gateway.shared.handler_base import GatewayHandler
from gateway.shared.model import UnifiedEvent, UnifiedMessage
from twin.shared.observability import call_with_langsmith_extra, langsmith_extra
from twin.shared.observability.langsmith import summarize_trace_output, traceable

if TYPE_CHECKING:
    from gateway.core.agent_router import AgentRouter

logger = logging.getLogger(__name__)

ERROR_MESSAGE = "Hệ thống não bộ của tớ đang bị quá tải xíu, cậu thử lại sau vài giây nhé!"

# Wait this long after a message before replying, coalescing a burst of rapid
# messages in the same scope into a single reply (the latest message wins).
MESSAGE_DEBOUNCE_SECONDS = float(os.getenv("MESSAGE_DEBOUNCE_SECONDS", "3.5"))


class GatewayChatHandler(GatewayHandler):
    """Routes unified chat messages to March7/Evernight without platform SDKs."""

    def __init__(self, agent_router: AgentRouter | None = None) -> None:
        self._agent_router = agent_router
        self._scope_locks: dict[str, asyncio.Lock] = {}
        self._latest_msg: dict[str, str] = {}
        self._burst_addressed: dict[str, bool] = {}

    async def handle_message(self, msg: UnifiedMessage) -> str:
        content = msg.content.strip()
        if not content:
            return ""

        hints = self._route_hints(msg)
        agent_name = str(hints.get("agent_name") or "march7")
        extra = langsmith_extra(
            tags=[msg.user.platform_name],
            metadata={
                "workflow": "gateway.chat",
                "agent_name": agent_name,
                "platform": msg.user.platform_name,
                "channel_type": msg.channel.channel_type,
                "channel_id": msg.channel.channel_id,
                "guild_id": msg.channel.guild_id,
                "user_id": msg.user.platform_id,
                "message_id": msg.message_id,
                "scope_key": self._scope_key(msg, hints),
            },
        )
        return await self._handle_message_traced(
            msg,
            content=content,
            hints=hints,
            agent_name=agent_name,
            langsmith_extra=extra,
        )

    @traceable(
        name="gateway.handle_message",
        run_type="chain",
        tags=["gateway", "chat"],
        process_outputs=summarize_trace_output,
    )
    async def _handle_message_traced(
        self,
        msg: UnifiedMessage,
        *,
        content: str,
        hints: dict[str, Any],
        agent_name: str,
    ) -> str:
        is_dm = msg.channel.channel_type == "dm"
        should_observe = bool(hints.get("observe", True))
        should_respond = bool(hints.get("should_respond", True))
        is_addressed = bool(hints.get("is_addressed", is_dm))
        allow_silence_candidate = bool(hints.get("allow_silence", False))

        if should_observe and agent_name == "march7":
            await self._observe_message(msg, content)
        elif should_observe and agent_name == "evernight":
            await self._observe_evernight_message(msg, content)

        if not should_respond:
            return ""

        scope_key = self._scope_key(msg, hints)
        self._latest_msg[scope_key] = msg.message_id
        self._burst_addressed[scope_key] = (
            self._burst_addressed.get(scope_key, False) or is_addressed
        )

        await asyncio.sleep(MESSAGE_DEBOUNCE_SECONDS)
        if self._latest_msg.get(scope_key) != msg.message_id:
            return ""

        async with self._get_lock(scope_key):
            if self._latest_msg.get(scope_key) != msg.message_id:
                return ""

            addressed = self._burst_addressed.pop(scope_key, False)
            allow_silence = allow_silence_candidate and not addressed

            try:
                return await self._route_to_agent(
                    msg=msg,
                    content=content,
                    allow_silence=allow_silence,
                    hints=hints,
                    agent_name=agent_name,
                )
            except Exception:
                logger.exception("Error processing message")
                return ERROR_MESSAGE

    async def handle_event(self, event: UnifiedEvent, msg: UnifiedMessage) -> None:
        logger.debug("Gateway event %s for message %s (no-op)", event, msg.message_id)

    async def _route_to_agent(
        self,
        *,
        msg: UnifiedMessage,
        content: str,
        allow_silence: bool,
        hints: dict[str, Any],
        agent_name: str,
    ) -> str:
        if not self._agent_router:
            return ERROR_MESSAGE

        logger.info(
            "Routing to %s agent: platform=%s user=%s content=%.80s",
            agent_name,
            msg.user.platform_name,
            msg.user.platform_id,
            content,
        )
        response = await call_with_langsmith_extra(
            self._agent_router.route,
            agent_name=agent_name,
            user_id=msg.user.platform_id,
            content=content,
            channel_id=self._conversation_id(msg, hints),
            observe_input=False,
            guild_id=self._space_id(msg, hints),
            bot_id=self._assistant_id(hints),
            bot_name=self._assistant_name(hints),
            allow_silence=allow_silence,
            user_name=msg.user.display_name,
            mentioned_users=self._mentioned_users(msg),
            langsmith_extra=langsmith_extra(
                tags=["router"],
                metadata={
                    "workflow": "gateway.route_agent",
                    "agent_name": agent_name,
                    "platform": msg.user.platform_name,
                    "channel_type": msg.channel.channel_type,
                    "channel_id": self._conversation_id(msg, hints),
                    "guild_id": self._space_id(msg, hints),
                    "user_id": msg.user.platform_id,
                    "message_id": msg.message_id,
                    "scope_key": self._scope_key(msg, hints),
                    "allow_silence": allow_silence,
                },
            ),
        )
        logger.info("Got response from %s: %.80s", agent_name, response)
        return response

    async def _observe_message(self, msg: UnifiedMessage, content: str) -> None:
        if not self._agent_router:
            return
        memory = getattr(getattr(self._agent_router, "march7", None), "memory", None)
        if memory is None:
            return

        if msg.channel.channel_type == "dm":
            await memory.observe_user_message(
                user_id=msg.user.platform_id,
                role="user",
                content=content,
            )
            return

        await memory.observe_channel_message(
            guild_id=str(msg.channel.guild_id or ""),
            channel_id=str(msg.channel.channel_id),
            author_id=msg.user.platform_id,
            author_name=msg.user.display_name,
            message_id=msg.message_id,
            content=content,
            reply_to=msg.reply_to,
        )

    async def _observe_evernight_message(self, msg: UnifiedMessage, content: str) -> None:
        # Persist one Evernight input BEFORE the reply debounce so a burst
        # keeps every entry. Only the router's local Evernight agent memory is
        # used; when Evernight is remote (A2A client) there is nothing local to
        # persist to and the server-side handle_chat observes with its default.
        # March7 T1 is never touched here.
        if not self._agent_router:
            return
        agent = getattr(self._agent_router, "evernight", None)
        memory = getattr(agent, "memory", None)
        if memory is None:
            return
        await memory.add_message(
            user_id=msg.user.platform_id, role="user", content=content
        )

    def _get_lock(self, scope_key: str) -> asyncio.Lock:
        lock = self._scope_locks.get(scope_key)
        if lock is None:
            lock = asyncio.Lock()
            self._scope_locks[scope_key] = lock
        return lock

    @staticmethod
    def _scope_key(msg: UnifiedMessage, hints: dict[str, Any]) -> str:
        if hints.get("scope_key"):
            return str(hints["scope_key"])
        if msg.channel.channel_type != "dm" and msg.channel.channel_id:
            return f"channel:{msg.channel.channel_id}"
        return f"dm:{msg.user.platform_id}"

    @staticmethod
    def _route_hints(msg: UnifiedMessage) -> dict[str, Any]:
        hints = dict(msg.extensions or {})

        # Compatibility with the old Discord converter names while adapters
        # migrate toward platform-neutral route hints.
        if "is_addressed" not in hints:
            hints["is_addressed"] = bool(
                hints.get("is_mentioned")
                or hints.get("is_reply_to_bot")
                or msg.channel.channel_type == "dm"
            )

        respond_mode = hints.get("respond_mode")
        if respond_mode is None:
            respond_mode = hints.get("channel_mode")

        if "should_respond" not in hints:
            hints["should_respond"] = bool(
                hints["is_addressed"]
                or respond_mode in {"respond", "respond_allowed", "always"}
            )

        if "allow_silence" not in hints:
            hints["allow_silence"] = bool(
                respond_mode in {"respond", "respond_allowed", "always"}
                and not hints["is_addressed"]
            )

        return hints

    @staticmethod
    def _conversation_id(msg: UnifiedMessage, hints: dict[str, Any]) -> str | None:
        if hints.get("conversation_id"):
            return str(hints["conversation_id"])
        if msg.channel.channel_type != "dm":
            return str(msg.channel.channel_id)
        return None

    @staticmethod
    def _space_id(msg: UnifiedMessage, hints: dict[str, Any]) -> str | None:
        if hints.get("space_id"):
            return str(hints["space_id"])
        return str(msg.channel.guild_id) if msg.channel.guild_id else None

    @staticmethod
    def _assistant_id(hints: dict[str, Any]) -> str | None:
        raw = hints.get("assistant_id")
        if raw is None:
            raw = hints.get("bot_id")
        return str(raw) if raw is not None else None

    @staticmethod
    def _assistant_name(hints: dict[str, Any]) -> str | None:
        raw = hints.get("assistant_name")
        if raw is None:
            raw = hints.get("bot_display_name")
        if raw is None:
            raw = hints.get("bot_name")
        return str(raw) if raw is not None else None

    @staticmethod
    def _mentioned_users(msg: UnifiedMessage) -> list[dict[str, Any]]:
        return [
            {
                "user_id": mention.platform_id,
                "display_name": mention.display_name,
                "is_bot": mention.is_bot,
            }
            for mention in msg.mentions
            if not mention.is_bot and mention.platform_id != msg.user.platform_id
        ]
