"""ActiveMemory facade — orchestrates observe/get_context/trim + thresholds."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

from twin.shared.config.settings import Config
from twin.shared.memory.active.constants import (
    KEEP_RECENT_MESSAGES_AFTER_SUMMARY,
    RECENT_CATALOGS_WINDOW,
    TOKEN_THRESHOLD,
    TOPIC_SHIFT_MIN_CONFIDENCE,
    TOPIC_SHIFT_MIN_STABLE_TURNS,
    TRIGGER_COOLDOWN_SECONDS,
)
from twin.shared.memory.active.detector import FastPathDetector
from twin.shared.memory.active.models import ActiveEntry
from twin.shared.memory.active.store import ActiveStore
from twin.shared.memory.consolidation_journal import discard_caller_pending

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]
TriggerCallback = Callable[[str, str], Awaitable[None]]


def _default_token_counter(content: str) -> int:
    return len(content.split())


class ActiveMemory:
    """High-level T1 facade."""

    def __init__(
        self,
        store: ActiveStore,
        detector: FastPathDetector,
        token_counter: TokenCounter | None = None,
        trigger_callback: TriggerCallback | None = None,
    ) -> None:
        self.store = store
        self.detector = detector
        self.token_counter = token_counter or _default_token_counter
        self.trigger_callback = trigger_callback
        self._in_progress: set[tuple[str, str]] = set()
        self._pending_tasks: set[asyncio.Task] = set()
        self._cooldown_until: dict[tuple[str, str], float] = {}

    async def observe(
        self,
        scope: str,
        scope_id: str,
        role: str,
        content: str,
        *,
        author_id: str | None = None,
        author_name: str | None = None,
        message_id: str | None = None,
        guild_id: str | None = None,
        channel_id: str | None = None,
        reply_to: str | None = None,
    ) -> ActiveEntry:
        tokens = int(self.token_counter(content))
        entry = ActiveEntry(
            scope=scope,
            scope_id=scope_id,
            role=role,
            author_id=author_id,
            author_name=author_name,
            message_id=message_id,
            guild_id=guild_id,
            channel_id=channel_id,
            reply_to=reply_to,
            content=content,
            tokens=tokens,
        )
        # Single Lua: entry+index+tokens+last_ts linearize on Redis.
        new_total = await self.store.observe_entry(entry)

        if role == "user":
            hit = self.detector.is_critical(content)
            if hit:
                logger.info(
                    "T1: fast-path catalog hit=%s scope=%s/%s entry=%s",
                    hit,
                    scope,
                    scope_id,
                    entry.entry_id,
                )

        if new_total >= TOKEN_THRESHOLD and self.trigger_callback is not None:
            self._maybe_fire_trigger(scope, scope_id, new_total)

        return entry

    def _maybe_fire_trigger(self, scope: str, scope_id: str, new_total: int) -> None:
        key = (scope, scope_id)
        if key in self._in_progress:
            logger.info(
                "memory.trigger_skipped scope=%s/%s reason=in_progress", scope, scope_id,
            )
            return
        cooldown_until = self._cooldown_until.get(key)
        if cooldown_until is not None and time.monotonic() < cooldown_until:
            logger.info(
                "memory.trigger_skipped scope=%s/%s reason=cooldown remaining_s=%.1f",
                scope, scope_id, cooldown_until - time.monotonic(),
            )
            return

        logger.info(
            "T1: threshold reached (%d>=%d) scope=%s/%s — triggering consolidator",
            new_total,
            TOKEN_THRESHOLD,
            scope,
            scope_id,
        )
        self._in_progress.add(key)
        # asyncio.create_task() copies the current contextvars.Context at
        # creation time, so this detached task automatically inherits the
        # caller's active LangSmith run (set via contextvars) and shows up as
        # a child of the originating message-handling trace — no manual
        # parent-header plumbing needed here (unlike the cross-process A2A
        # path, which does need it; see observability.a2a_parent_headers()).
        task = asyncio.create_task(self.trigger_callback(scope, scope_id))
        self._pending_tasks.add(task)
        task.add_done_callback(lambda t: self._on_trigger_done(key, t))

    def _on_trigger_done(self, key: tuple[str, str], task: asyncio.Task) -> None:
        self._pending_tasks.discard(task)
        self._in_progress.discard(key)
        self._cooldown_until[key] = time.monotonic() + TRIGGER_COOLDOWN_SECONDS
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            logger.error(
                "T1: consolidation trigger failed scope=%s/%s: %s",
                *key,
                exc,
                exc_info=exc,
            )

    async def get_context(
        self, scope: str, scope_id: str, *, limit: int = 50
    ) -> list[ActiveEntry]:
        return await self.store.list_entries(scope, scope_id, limit=limit)

    async def get_entries_by_ids(
        self, scope: str, scope_id: str, entry_ids: list[str]
    ) -> list[ActiveEntry]:
        return await self.store.get_entries_by_ids(scope, scope_id, entry_ids)

    async def list_unsummarized_entries(
        self, scope: str, scope_id: str, *, limit: int = 200
    ) -> list[ActiveEntry]:
        return await self.store.list_unsummarized_entries(scope, scope_id, limit=limit)

    async def trim(
        self,
        scope: str,
        scope_id: str,
        summarized_entry_ids: list[str],
        *,
        keep_recent: int = KEEP_RECENT_MESSAGES_AFTER_SUMMARY,
    ) -> None:
        if not summarized_entry_ids:
            return
        all_entries = await self.store.list_entries(scope, scope_id, limit=10_000)
        # Keep the N most recent entries no matter what (Lua re-applies this
        # authoritatively; the Python copy is only for best-effort archive).
        keep_ids = {e.entry_id for e in all_entries[-keep_recent:]} if keep_recent > 0 else set()
        by_id = {e.entry_id: e for e in all_entries}
        seen: set[str] = set()
        cand: list[str] = []
        for eid in summarized_entry_ids:
            if eid in seen:
                continue
            seen.add(eid)
            if eid not in keep_ids:
                cand.append(eid)
        # W3: park raw entries in cold archive before deleting. Best-effort and
        # never blocks trim; fetch beyond-10k IDs directly for completeness.
        if cand and getattr(Config, "T1_ARCHIVE_ENABLED", True):
            to_archive = []
            for eid in cand:
                ent = by_id.get(eid)
                if ent is None:
                    try:
                        ent = await self.store.get_entry(scope, scope_id, eid)
                    except Exception:
                        ent = None
                if ent is not None:
                    to_archive.append(ent)
            try:
                ttl_days = int(getattr(Config, "T1_ARCHIVE_TTL_DAYS", 90))
                await self.store.archive_entries(
                    scope, scope_id, to_archive, ttl_seconds=ttl_days * 86400,
                )
            except Exception as exc:
                logger.warning(
                    "T1: archive-on-trim failed scope=%s/%s entries=%d -- trimming anyway: %s",
                    scope, scope_id, len(to_archive), exc,
                )
        # Single Lua: delete outside keep, mark retained, decrement once.
        # Concurrent observes linearize before/after; never overwritten.
        res = await self.store.trim_summarized(
            scope, scope_id, summarized_entry_ids, keep_recent=keep_recent,
        )
        logger.info(
            "T1: trim scope=%s/%s deleted=%d remaining_tokens=%d",
            scope, scope_id, len(res["deleted_ids"]), int(res["unsummarized_tokens"]),
        )

    async def trim_consolidated_batch(self, scope: str, scope_id: str, record: dict, *, keep_recent: int = KEEP_RECENT_MESSAGES_AFTER_SUMMARY) -> dict:
        """Guarded consolidation trim: snapshots + single-EVAL trim+trimmed."""
        from twin.shared.memory.active.consolidation_trim import guarded_trim_batch

        return await guarded_trim_batch(self.store, scope, scope_id, record, int(keep_recent))

    async def reset_scope(self, scope: str, scope_id: str) -> None:
        await self.store.clear_scope(scope, scope_id)
        # Own pending batch is unrecoverable once T1 is cleared; receiver
        # witnesses stay (T2 partials may persist) — see journal docstring.
        await discard_caller_pending(self.store.redis, scope, scope_id)
        logger.info("T1: reset scope=%s/%s", scope, scope_id)

    async def push_catalog(
        self, scope: str, scope_id: str, catalog: str, confidence: float
    ) -> bool:
        """Append catalog to rolling window. Returns True if this is a topic shift."""
        state = await self.store.get_state(scope, scope_id)
        recent: list[str] = list(state.get("recent_catalogs") or [])
        shift = self.is_topic_shift(recent, catalog, confidence)
        recent.append(catalog)
        if len(recent) > RECENT_CATALOGS_WINDOW:
            recent = recent[-RECENT_CATALOGS_WINDOW:]
        await self.store.update_state(scope, scope_id, recent_catalogs=recent)
        if shift:
            logger.info("T1: topic shift detected scope=%s/%s -> %s", scope, scope_id, catalog)
        return shift

    @staticmethod
    def is_topic_shift(
        recent_catalogs: list[str], new_catalog: str, new_confidence: float
    ) -> bool:
        if new_confidence < TOPIC_SHIFT_MIN_CONFIDENCE:
            return False
        if len(recent_catalogs) < TOPIC_SHIFT_MIN_STABLE_TURNS:
            return False
        last_n = recent_catalogs[-TOPIC_SHIFT_MIN_STABLE_TURNS:]
        if len(set(last_n)) == 1 and new_catalog != last_n[-1]:
            return True
        return False

    # Used by callers that want a fresh utc datetime without importing.
    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)
