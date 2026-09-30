"""T2 read queries — DiaryReader (I/O collaborator).

Owns the RediSearch query strings for KNN, BM25, and recent-timeline
reads plus result parsing. "Recent" orders by content time: ``period_end``
with a ``created_at`` fallback for legacy docs (#33) — a summary merged
today must surface as recent even though its ``created_at`` is old.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from twin.shared.memory.diary.codec import (
    escape_tag_value,
    pack_embedding,
    parse_results,
)

logger = logging.getLogger(__name__)


def time_filter_clause(since_ts: float | None, until_ts: float | None) -> str | None:
    """Build the RediSearch OR-fallback time filter clause (P3.2).

    RediSearch has no COALESCE: a doc missing `period_end` (pre-v3) never
    matches any range query on it, so `-@period_end:[-inf +inf]` isolates
    those docs and re-tests them against `created_at` instead. None when
    both bounds are unset.
    """
    if since_ts is None and until_ts is None:
        return None
    lo = since_ts if since_ts is not None else "-inf"
    hi = until_ts if until_ts is not None else "+inf"
    rng = f"[{lo} {hi}]"
    return f"(@period_end:{rng} | (-@period_end:[-inf +inf] @created_at:{rng}))"


def recency_key(doc: dict[str, Any]) -> float:
    """Content-time sort key: period_end, legacy created_at fallback, else 0."""
    for field in ("period_end", "created_at"):
        value = doc.get(field)
        if isinstance(value, datetime):
            return value.timestamp()
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return 0.0


class DiaryReader:
    """Executes FT.SEARCH reads against the timeline index."""

    def __init__(self, redis_client: Any, *, index_name: str, prefix: str):
        self.redis = redis_client
        self.index_name = index_name
        self.prefix = prefix

    async def search_knn(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int,
        topic_filter: str | None,
        *,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """KNN semantic search. Returns raw hits ungated — callers apply the
        cosine gate (search() gates directly for pure-KNN, or post-fusion for
        hybrid so BM25 can't smuggle a gated-out doc back in)."""
        uid = escape_tag_value(user_id)
        tag_filter = f"@user_id:{{{uid}}}"
        if topic_filter:
            tag_filter += f" @topic:{{{escape_tag_value(topic_filter)}}}"
        clause = time_filter_clause(since_ts, until_ts)
        filter_expr = f"{tag_filter} {clause}" if clause else tag_filter
        query = f"({filter_expr})=>[KNN {limit} @embedding $vec AS score]"

        try:
            results = await self.redis.execute_command(
                "FT.SEARCH", self.index_name,
                query,
                "PARAMS", "2", "vec", pack_embedding(query_embedding),
                "SORTBY", "score", "ASC",
                "LIMIT", "0", str(limit),
                "DIALECT", "2",
            )
            return parse_results(results, self.prefix)
        except Exception as exc:
            logger.error("Timeline KNN search failed: %s", exc)
            return []

    async def search_bm25(
        self,
        user_id: str,
        query_text: str,
        limit: int,
        topic_filter: str | None,
        *,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 full-text search on the 'summary' field."""
        safe_text = re.sub(r"[^a-zA-Z0-9\sÀ-ɏẠ-ỹ]", " ", query_text).strip()
        if not safe_text:
            return []

        # OR the terms: lexical recall wants "any keyword matches" (e.g. catch
        # "Rei"/"AMD"), not "all words present". RediSearch ANDs space-separated
        # terms by default, which would make BM25 almost never fire on natural
        # queries — silently degrading hybrid back to KNN-only.
        terms = [t for t in safe_text.split() if t]
        if not terms:
            return []
        term_group = " | ".join(terms)

        uid = escape_tag_value(user_id)
        tag_filter = f"@user_id:{{{uid}}}"
        if topic_filter:
            tag_filter += f" @topic:{{{escape_tag_value(topic_filter)}}}"
        clause = time_filter_clause(since_ts, until_ts)
        if clause:
            tag_filter += f" {clause}"

        bm25_query = f"({tag_filter}) ({term_group})"

        try:
            results = await self.redis.execute_command(
                "FT.SEARCH", self.index_name,
                bm25_query,
                "SCORER", "BM25",
                "WITHSCORES",
                "LIMIT", "0", str(limit),
                "DIALECT", "2",
            )
            return parse_results(results, self.prefix, has_scores=True)
        except Exception as exc:
            logger.error("Timeline BM25 search failed: %s", exc)
            return []

    async def get_recent(
        self,
        user_id: str,
        limit: int,
        *,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """Recent summaries, newest content-time first (#33).

        "Recent" is period_end with a created_at fallback for legacy docs
        missing period_end — but a single SORTBY period_end + LIMIT n window
        can never observe a legacy doc ranked inside the global top-n: n
        current-schema rows with any period_end (even ancient content)
        evict it before the Python re-sort runs. So each partition is
        windowed independently — top-n by period_end plus top-n
        missing-period_end by created_at — then merged, deduped, and
        re-sorted by the exact recency key. A global top-n doc ranks
        within the top-n of its own partition, so each branch fetching
        exactly n is sufficient (no overfetch factor). Indexes that predate
        the diary fields (FT.ALTER failed) fall back to a created_at-only
        query rather than returning nothing.
        """
        tag_filter = f"@user_id:{{{escape_tag_value(user_id)}}}"
        if since_ts is None and until_ts is None:
            modern_query = f"{tag_filter} @period_end:[-inf +inf]"
            legacy_query = f"{tag_filter} -@period_end:[-inf +inf]"
            modern_dialect: list[str] = []
            legacy_dialect = ["DIALECT", "2"]
        else:
            lo = since_ts if since_ts is not None else "-inf"
            hi = until_ts if until_ts is not None else "+inf"
            rng = f"[{lo} {hi}]"
            modern_query = f"({tag_filter} @period_end:{rng})"
            legacy_query = (
                f"({tag_filter} -@period_end:[-inf +inf] @created_at:{rng})"
            )
            modern_dialect = ["DIALECT", "2"]
            legacy_dialect = ["DIALECT", "2"]
        try:
            modern_raw = await self.redis.execute_command(
                "FT.SEARCH", self.index_name,
                modern_query,
                "SORTBY", "period_end", "DESC",
                "LIMIT", "0", str(limit),
                *modern_dialect,
            )
            legacy_raw = await self.redis.execute_command(
                "FT.SEARCH", self.index_name,
                legacy_query,
                "SORTBY", "created_at", "DESC",
                "LIMIT", "0", str(limit),
                *legacy_dialect,
            )
        except Exception as exc:
            if _is_unknown_sort_field(exc):
                return await self._get_recent_legacy_index(
                    tag_filter, limit, since_ts, until_ts,
                )
            logger.error("Timeline get_recent failed: %s", exc)
            return []
        docs = parse_results(modern_raw, self.prefix)
        docs += parse_results(legacy_raw, self.prefix)
        # Partitions are disjoint by construction (has vs missing period_end);
        # dedup defensively anyway — first occurrence wins.
        seen: set[Any] = set()
        merged: list[dict[str, Any]] = []
        for doc in docs:
            key = doc.get("summary_id") or id(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        merged.sort(key=recency_key, reverse=True)
        return merged[:limit]

    async def _get_recent_legacy_index(
        self,
        tag_filter: str,
        limit: int,
        since_ts: float | None,
        until_ts: float | None,
    ) -> list[dict[str, Any]]:
        """created_at-only read for indexes missing the diary fields (degraded).

        References no period_end term at all: on a no-diary-field index any
        such term — or SORTBY period_end — raises unknown-field instead of
        matching nothing, so the time window degrades to created_at-only.
        """
        if since_ts is None and until_ts is None:
            query = tag_filter
            dialect_args: list[str] = []
        else:
            lo = since_ts if since_ts is not None else "-inf"
            hi = until_ts if until_ts is not None else "+inf"
            query = f"({tag_filter} @created_at:[{lo} {hi}])"
            dialect_args = ["DIALECT", "2"]
        try:
            results = await self.redis.execute_command(
                "FT.SEARCH", self.index_name,
                query,
                "SORTBY", "created_at", "DESC",
                "LIMIT", "0", str(limit),
                *dialect_args,
            )
        except Exception as exc:
            logger.error("Timeline get_recent failed: %s", exc)
            return []
        docs = parse_results(results, self.prefix)
        docs.sort(key=recency_key, reverse=True)
        return docs[:limit]


def _is_unknown_sort_field(exc: Exception) -> bool:
    # Covers SORTBY and query-side unknown-field failures alike: on a
    # no-diary-field index any period_end term raises instead of matching.
    msg = str(exc).lower()
    return any(
        frag in msg
        for frag in ("not loaded nor in schema", "unknown field", "no such field",
                     "property", "not sortable")
    )
