"""Consolidation writers: idempotent T2 stores + atomic T3 apply."""
from __future__ import annotations

import logging
from typing import Any

from twin.shared.config.settings import Config
from twin.shared.tools.modules.memory.consolidation_ids import build_idempotency_key
from twin.shared.tools.modules.memory.consolidation_schema import TopicPlan

logger = logging.getLogger(__name__)


class ConsolidationWriter:
    """Persist normalized topics and profile mutations (fail-closed)."""

    def __init__(self, embedding_service: Any, timeline_summary_store: Any) -> None:
        self._embeddings = embedding_service
        self._t2 = timeline_summary_store

    async def _embed(self, summary: str) -> list[float]:
        return await self._embeddings.get_embedding(
            f"{Config.EMBEDDING_PASSAGE_PREFIX}{summary}"
        )

    async def _store_one(
        self,
        *,
        scope: str,
        scope_id: str,
        entry_ids: list[str],
        plan: TopicPlan,
        embedding: list[float],
        period_start: float,
        period_end: float,
    ) -> str:
        key = build_idempotency_key(scope, scope_id, entry_ids, plan.topic)
        kwargs: dict[str, Any] = {
            "user_id": scope_id,
            "summary": plan.summary,
            "embedding": embedding,
            "topic": plan.topic,
            "topic_display": plan.topic_display,
            "importance": plan.importance,
            "period_start": period_start,
            "period_end": period_end,
            "source_entry_ids": list(entry_ids),
            "idempotency_key": key,
        }
        # No legacy fallback: every supported diary store accepts
        # idempotency_key, and a store without it must fail closed (the
        # TypeError surfaces as a per-topic failure, never a silent dup).
        return await self._t2.store_summary(**kwargs)

    async def store_topics(
        self,
        *,
        scope: str,
        scope_id: str,
        entry_ids: list[str],
        topics: tuple[TopicPlan, ...],
        period_start: float,
        period_end: float,
    ) -> tuple[list[str], int, int]:
        """Store all topics; returns (summary_ids, attempted, failed)."""
        summary_ids: list[str] = []
        attempted = 0
        failed = 0
        for plan in topics:
            attempted += 1
            try:
                embedding = await self._embed(plan.summary)
                sid = await self._store_one(
                    scope=scope,
                    scope_id=scope_id,
                    entry_ids=entry_ids,
                    plan=plan,
                    embedding=embedding,
                    period_start=period_start,
                    period_end=period_end,
                )
                summary_ids.append(sid)
            except Exception as exc:
                failed += 1
                logger.warning("consolidation: T2 store failed topic=%s: %s", plan.topic, exc)
        return summary_ids, attempted, failed


class ConsolidationProfileWriter:
    """Apply profile mutations atomically when the store supports it."""

    def __init__(self, profile_store: Any) -> None:
        self._profile = profile_store

    async def apply(
        self,
        *,
        scope: str,
        scope_id: str,
        updates: dict[str, list[str]],
        rewrites: dict[str, list[str]],
        expected_profile_hash: str | None,
    ) -> dict[str, Any]:
        """Apply rewrites+appends; channel scope or empty maps are no-ops."""
        if scope == "channel":
            return {
                "ok": True, "skipped": True,
                "updated_sections": [], "rewritten_sections": [],
            }
        updates = {k: v for k, v in (updates or {}).items() if v}
        rewrites = {k: v for k, v in (rewrites or {}).items() if v}
        if not updates and not rewrites:
            return {
                "ok": True, "skipped": False, "written": False,
                "updated_sections": [], "rewritten_sections": [],
            }
        atomic = getattr(self._profile, "apply_consolidation_updates", None)
        if not callable(atomic):
            raise RuntimeError(
                "profile store lacks atomic apply_consolidation_updates; "
                "refusing partial non-atomic profile writes"
            )
        return await atomic(scope_id, rewrites, updates, expected_profile_hash)
