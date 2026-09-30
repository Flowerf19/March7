"""SearchMemoryTool - query T2 timeline memory."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Callable, Coroutine, Optional

from twin.shared.config.settings import Config
from twin.shared.memory.vn_time import vn_now
from twin.shared.tools.modules.memory.memory_format import (
    days_ago_display,
    format_memories,
)
from twin.shared.tools.modules.memory.memory_ranking import (
    dedup_key,
    rank_semantic_global,
    recency_key,
    widen_search,
)
from twin.shared.tools.modules.memory.memory_trace import trace_semantic_results
from twin.shared.tools.registry.base import BaseTool, ToolExecutionError

logger = logging.getLogger(__name__)

_MAX_LIMIT = 20
_WIDEN_MULTIPLIER = 3


class SearchMemoryTool(BaseTool):
    """Tool for querying Redis Stack backed T2 timeline memory.

    Interface v3 (P3.1, fix B1): the only params are `query` and
    `days_back` (plus `user_id`/`channel_id`/`limit`) — `mode`/`topic`/
    `hours`/`days` are gone. Behavior is inferred from which of
    query/days_back are present:
        query + days_back -> hybrid search, time-filtered
        query only        -> hybrid search, all-time
        days_back only    -> timeline in that range, sorted new -> old
        neither            -> recent
    Thin orchestrator: ranking/widen in memory_ranking, rendering in
    memory_format, SEARCH traces in memory_trace.
    """

    def __init__(
        self,
        timeline_summary_store: Optional[Any] = None,
        embedding_service: Optional[Any] = None,
    ):
        self.timeline_summary_store = timeline_summary_store
        self.embedding_service = embedding_service

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Discord user ID.",
                },
                "channel_id": {
                    "type": "string",
                    "description": (
                        "ID kênh Discord hiện tại — chỉ truyền khi đang chat "
                        "trong kênh chung (lấy 'Platform channel ID' từ system "
                        "prompt). Tool sẽ tìm cả ký ức của kênh."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Truy vấn đã rewrite thành từ khóa chủ đề. Bỏ trống = "
                        "xem dòng thời gian gần đây."
                    ),
                },
                "days_back": {
                    "type": "integer",
                    "description": (
                        "Số ngày nhìn lại — chỉ đưa khi user có mốc thời gian "
                        "('hôm qua'→2, 'mấy ngày trước'→7, 'tuần trước'→10, "
                        "'tháng trước'→35)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": f"Số kết quả, tối đa {_MAX_LIMIT}.",
                },
            },
            "required": ["user_id"],
        }

    async def execute(
        self,
        user_id: str,
        channel_id: Optional[str] = None,
        query: Optional[str] = None,
        days_back: Optional[int] = None,
        limit: int = 5,
    ) -> str:
        user_id = str(user_id or "").strip()
        channel_id = str(channel_id or "").strip() or None
        query = (query or "").strip() or None
        days_back = self._positive_int_or_none(days_back)

        if not user_id:
            return "Lỗi: Thiếu user_id."

        if not user_id.isdigit():
            logger.warning("T2: invalid user_id for search_memory: %s", user_id)
            return f"Lỗi: user_id '{user_id}' không hợp lệ. user_id phải là số ID của Discord user."

        if channel_id and not channel_id.isdigit():
            # Auxiliary scope only — drop it rather than failing the recall.
            logger.warning("T2: invalid channel_id for search_memory: %s", channel_id)
            channel_id = None

        if self.timeline_summary_store is None:
            return "Lỗi: timeline_summary_store chưa được cấu hình."

        limit = self._bounded_limit(limit)

        # T2 channel summaries are stored under user_id=channel_id, so a turn
        # in a channel must search BOTH the speaker scope and the channel
        # scope. This tool is the ONLY recall path (no automatic preflight),
        # so dropping the channel scope here would make channel memory
        # permanently unreachable.
        scope_ids = [user_id]
        if channel_id and channel_id != user_id:
            scope_ids.append(channel_id)

        widen_label: Optional[str] = None
        try:
            if query:
                if self.embedding_service is None:
                    return "Lỗi: embedding_service chưa được cấu hình."
                memories, widen_label = await self._search_with_query(
                    scope_ids, query, days_back, limit,
                )
            elif days_back:
                memories, widen_label = await self._search_timeline(
                    scope_ids, days_back, limit,
                )
            else:
                memories = await self._search_recent(scope_ids, limit)
        except Exception as e:
            logger.error("SearchMemoryTool: search failed: %s", e, exc_info=True)
            return f"Lỗi khi tìm kiếm ký ức: {e}"

        return format_memories(memories, widen_label=widen_label)

    # ------------------------------------------------------ behavior matrix

    async def _search_with_query(
        self, scope_ids: list[str], query: str, days_back: Optional[int], limit: int,
    ) -> tuple[list[Any], Optional[str]]:
        """`query` present (P3.1): always hybrid (KNN+BM25) — v3 drops the
        old separate semantic-only mode. `days_back`, if given, widens on
        empty (P3.3); the embedding is computed once and reused across
        widen rounds — only the time window changes between rounds."""
        query_input = f"{Config.EMBEDDING_QUERY_PREFIX}{query}"
        embedding = await self.embedding_service.get_embedding(query_input)

        async def round_fn(since_ts: Optional[float]) -> list[Any]:
            memories, sources = await self._dual_scope_semantic(
                scope_ids, embedding, limit, query, since_ts,
            )
            trace_semantic_results(
                self.embedding_service,
                input_text=query_input,
                current_query=query,
                query_embedding=embedding,
                summaries=memories,
                scope_ids=scope_ids,
                sources=sources,
            )
            return memories

        if days_back is None:
            return await round_fn(None), None
        return await self._widen(round_fn, days_back)

    async def _search_timeline(
        self, scope_ids: list[str], days_back: int, limit: int,
    ) -> tuple[list[Any], Optional[str]]:
        """`days_back` alone (no query): timeline within that range, sorted
        new -> old, no embedding call. Widens on empty (P3.3)."""

        async def round_fn(since_ts: Optional[float]) -> list[Any]:
            return await self._dual_scope_recent(scope_ids, limit, since_ts)

        return await self._widen(round_fn, days_back)

    async def _search_recent(self, scope_ids: list[str], limit: int) -> list[Any]:
        """Neither `query` nor `days_back`: plain recent, no time filter —
        nothing to widen from, so this is a single unlabeled round."""
        return await self._dual_scope_recent(scope_ids, limit, since_ts=None)

    async def _widen(
        self,
        round_fn: Callable[[Optional[float]], Coroutine[Any, Any, list[Any]]],
        days_back: int,
    ) -> tuple[list[Any], Optional[str]]:
        return await widen_search(
            round_fn, days_back,
            since_ts_from_days=self._since_ts_from_days_back,
            multiplier=_WIDEN_MULTIPLIER,
        )

    # ------------------------------------------------------- dual-scope I/O

    async def _dual_scope_semantic(
        self,
        scope_ids: list[str],
        embedding: list[float],
        limit: int,
        query_text: str,
        since_ts: Optional[float],
    ) -> tuple[list[Any], list[str]]:
        """Hybrid-search every scope, dedup, GLOBALLY rank top-K (#32).
        Returns (memories, sources) with sources[i] the scope memories[i]
        came from (needed by the trace log)."""
        pairs: list[tuple[Any, str]] = []
        seen: set[Any] = set()
        # Only pass since_ts when set, so store fakes with the pre-P3.2
        # signature (no since_ts kwarg) stay compatible on the untimed path.
        time_kwargs: dict[str, float] = {}
        if since_ts is not None:
            time_kwargs["since_ts"] = since_ts
        for sid in scope_ids:
            results = await self.timeline_summary_store.search(
                user_id=sid,
                query_embedding=embedding,
                limit=limit,
                query_text=query_text,
                **time_kwargs,
            )
            for summary in results:
                key = dedup_key(summary)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((summary, sid))
        ranked = rank_semantic_global(pairs, embedding, limit)
        return [m for m, _ in ranked], [s for _, s in ranked]

    async def _dual_scope_recent(
        self, scope_ids: list[str], limit: int, since_ts: Optional[float],
    ) -> list[Any]:
        """get_recent every scope, dedup, re-sort merged by content time."""
        memories: list[Any] = []
        seen: set[Any] = set()
        time_kwargs: dict[str, float] = {}
        if since_ts is not None:
            time_kwargs["since_ts"] = since_ts
        for sid in scope_ids:
            results = await self.timeline_summary_store.get_recent(
                user_id=sid, limit=limit, **time_kwargs,
            )
            for summary in results:
                key = dedup_key(summary)
                if key in seen:
                    continue
                seen.add(key)
                memories.append(summary)
        # Each scope returns newest-first; re-sort the merged list so channel
        # hits interleave with speaker hits chronologically (content time).
        memories.sort(key=recency_key, reverse=True)
        return memories[:limit]

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _positive_int_or_none(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _since_ts_from_days_back(days_back: int) -> float:
        return (vn_now() - timedelta(days=days_back)).timestamp()

    @staticmethod
    def _bounded_limit(limit: int) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError):
            value = 5
        return max(1, min(value, _MAX_LIMIT))

    @staticmethod
    def _days_ago_display(memory: Any) -> Optional[str]:
        return days_ago_display(memory, now=vn_now())

    def __repr__(self) -> str:
        return f"<SearchMemoryTool: timeline_summary_store={self.timeline_summary_store is not None}>"
