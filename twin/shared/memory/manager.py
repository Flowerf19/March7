"""Agent-facing shared memory manager for T1/T2/T3."""
from __future__ import annotations

import logging
from typing import Any, Callable

from twin.shared.memory.active import ActiveEntry, ActiveMemory
from twin.shared.memory.consolidation_coordinator import (
    ConsolidationCoordinator,
    entry_to_snapshot,
    normalize_role,
)
from twin.shared.memory.profile import MarkdownProfileStore
from twin.shared.observability.langsmith import traceable

logger = logging.getLogger(__name__)


class SharedMemoryManager:
    """Bridge agent chat code to the new shared memory stack."""

    def __init__(
        self,
        *,
        active: ActiveMemory,
        profile_store: MarkdownProfileStore,
        timeline_summary_store: Any = None,
        embedding_service: Any = None,
        consolidation_client: Any = None,
        local_consolidator: Any = None,
    ) -> None:
        self.t1 = active
        self.profile = profile_store
        self.t3 = profile_store
        self.timeline_summary_store = timeline_summary_store
        self.embedding_service = embedding_service
        self.consolidation_client = consolidation_client
        self.local_consolidator = local_consolidator
        self._coordinator = ConsolidationCoordinator(
            self.t1,
            lambda: self.consolidation_client,
            lambda: self.local_consolidator,
            get_caller_redis=lambda: getattr(getattr(self.t1, "store", None), "redis", None),
            get_receiver_redis=lambda: getattr(self.timeline_summary_store, "redis", None),
        )

    # ------------------------------------------------------------------ writes

    async def add_message(self, user_id: str, role: str, content: str) -> None:
        await self.observe_user_message(user_id=user_id, role=role, content=content)

    async def observe_user_message(self, user_id: str, role: str, content: str) -> None:
        await self.t1.observe(
            "user",
            str(user_id),
            self._normalize_role(role),
            content,
            author_id=str(user_id) if role != "assistant" else None,
            author_name=str(user_id) if role != "assistant" else None,
        )

    async def add_assistant_message(
        self,
        user_id: str,
        content: str,
        *,
        channel_id: str | None = None,
        guild_id: str | None = None,
        bot_id: str | None = None,
        bot_name: str | None = None,
    ) -> None:
        if channel_id:
            await self.t1.observe(
                "channel",
                str(channel_id),
                "assistant",
                content,
                author_id=bot_id or "march7",
                author_name=bot_name or "March7",
                guild_id=guild_id,
                channel_id=str(channel_id),
            )
            return
        await self.t1.observe(
            "user",
            str(user_id),
            "assistant",
            content,
            author_id=bot_id,
            author_name=bot_name,
        )

    async def observe_channel_message(
        self,
        guild_id: str,
        channel_id: str,
        author_id: str,
        author_name: str,
        message_id: str,
        content: str,
        reply_to: str | None = None,
    ) -> None:
        await self.t1.observe(
            "channel",
            str(channel_id),
            "user",
            content,
            author_id=str(author_id),
            author_name=author_name,
            message_id=message_id,
            guild_id=str(guild_id),
            channel_id=str(channel_id),
            reply_to=reply_to,
        )

    # ------------------------------------------------------------------- reads

    async def get_context(
        self,
        user_id: str,
        current_query: str,
        channel_id: str | None = None,
        user_name: str | None = None,
        mentioned_users: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict]]:
        # Decision 2026-07-03 — T2 recall is TOOL-ONLY via the search_memory
        # tool (see tools/prompts/guides/search_memory.md). Automatic T2
        # preflight injection has been removed twice already; do NOT re-add it
        # without new measured data. `current_query` stays in the signature for
        # existing callers even though nothing embeds it here anymore.
        del current_query
        scope = "channel" if channel_id else "user"
        scope_id = str(channel_id or user_id)

        entries = await self.t1.get_context(scope, scope_id)
        messages = self._entries_to_messages(entries)

        # Anchor WHO is speaking right now. In a channel the transcript carries
        # many authors, so every author line includes both display name and
        # stable platform ID. The header pins the current speaker explicitly.
        if user_name:
            user_id_header = (
                "=== CURRENT USER ===\n"
                f"Người đang nói chuyện với bạn ngay lúc này: {user_name} "
                f"(Platform user ID: {user_id})"
            )
        else:
            user_id_header = f"=== CURRENT USER ===\nPlatform user ID: {user_id}"
        if channel_id:
            # The search_memory tool recalls channel summaries via their scope
            # id (T2 stores them under user_id=channel_id) — expose the id so
            # the model can pass `channel_id` when it calls the tool.
            user_id_header += (
                f"\nĐang chat trong kênh chung (Platform channel ID: {scope_id})"
            )
        mentioned_context = await self._mentioned_users_context(
            str(user_id),
            mentioned_users,
        )
        profile_context = await self.profile.get_system_prompt_context(str(user_id))
        system_parts = [user_id_header] + [
            part
            for part in (mentioned_context, profile_context)
            if part
        ]
        return "\n\n".join(system_parts), messages

    async def get_snapshot(self, user_id: str) -> list[dict]:
        entries = await self.t1.get_context("user", str(user_id), limit=200)
        return [self._entry_to_snapshot(e) for e in entries]

    async def clear_session(self, user_id: str) -> None:
        await self.t1.reset_scope("user", str(user_id))

    # -------------------------------------------------------------- consolidate

    @traceable(
        name="memory.request_consolidation",
        run_type="chain",
        tags=["memory", "consolidation", "request"],
    )
    async def consolidate_scope(
        self,
        scope: str,
        scope_id: str,
        entries: list[dict] | None = None,
    ) -> dict:
        """Consolidate T1 messages, then trim on validated acknowledgement."""
        return await self._coordinator.consolidate_scope(scope, scope_id, entries)

    async def consolidate_snapshot(
        self,
        user_id: str,
        snapshot: list[dict],
        reason: str = "manual",
    ) -> bool:
        return await self._coordinator.consolidate_snapshot(
            user_id, snapshot, reason=reason,
        )

    # ---------------------------------------------------------------- helpers

    async def _mentioned_users_context(
        self,
        current_user_id: str,
        mentioned_users: list[dict[str, Any]] | None,
    ) -> str:
        users = self._normalize_mentioned_users(current_user_id, mentioned_users)
        if not users:
            return ""

        lines = [
            "=== MENTIONED USERS ===",
            "Những người được nhắc tới trong tin nhắn hiện tại. "
            "Đây không phải người đang nói, trừ khi trùng với CURRENT USER.",
        ]
        for user in users:
            user_id = user["user_id"]
            display_name = user.get("display_name") or user_id
            lines.append(f"- {display_name} (Platform user ID: {user_id})")
            for bullet in await self._mentioned_user_basic_bullets(user_id):
                lines.append(f"  - {bullet}")
        return "\n".join(lines)

    async def _mentioned_user_basic_bullets(self, user_id: str) -> list[str]:
        reader = getattr(self.profile, "read_section_if_exists", None)
        if reader is None:
            return []
        try:
            bullets = await reader(str(user_id), "basic")
        except Exception as exc:
            logger.debug("T3: mentioned profile read failed user=%s: %s", user_id, exc)
            return []
        return list(bullets[:3])

    @staticmethod
    def _normalize_mentioned_users(
        current_user_id: str,
        mentioned_users: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in mentioned_users or []:
            raw_id = item.get("user_id") or item.get("platform_id")
            user_id = str(raw_id or "").strip()
            if not user_id or user_id == current_user_id or user_id in seen:
                continue
            seen.add(user_id)
            display_name = str(item.get("display_name") or "").strip()
            normalized.append({"user_id": user_id, "display_name": display_name})
        return normalized

    @staticmethod
    def _normalize_role(role: str) -> str:
        return normalize_role(role)

    @staticmethod
    def _entries_to_messages(entries: list[ActiveEntry]) -> list[dict]:
        messages: list[dict] = []
        for entry in entries:
            role = "assistant" if entry.role == "assistant" else "user"
            content = entry.content
            if entry.role == "assistant":
                lowered = content.lower()
                # Only mark prior assistant turns that look like host-state
                # output — narrow tokens avoid over-marking casual chat. The
                # marker nudges the LLM to re-call host_system instead of reusing
                # stale numbers when the user asks again about realtime host.
                if any(
                    token in lowered
                    for token in (
                        "uptime",
                        "load average",
                        "df -h",
                        "docker ps",
                        "docker logs",
                        "container",
                        "systemctl",
                    )
                ):
                    content = (
                        "[context cũ — số liệu host có thể stale; nếu user hỏi lại trạng thái host BẮT BUỘC gọi host_system lại] "
                        + content
                    )
            if entry.scope == "channel" and entry.role == "user" and entry.author_name:
                label = entry.author_name
                if entry.author_id:
                    label = f"{label} [user_id={entry.author_id}]"
                content = f"{label}: {content}"
            messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _entry_to_snapshot(entry: ActiveEntry) -> dict:
        return entry_to_snapshot(entry)
