"""Timeline summary storage — T2 memory layer (schema v3: diary model).

TimelineSummaryStore orchestrates writes (store_summary, same-day diary
merge) and hybrid KNN+BM25 search over Redis Stack. Field encode/decode
lives in codec.py, same-day merge in merge.py, index DDL/introspection
in schema.py, read queries in reader.py, atomic commits in writer.py,
and the cosine-gate policy in gates.py.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from twin.shared.memory.diary.merge import try_diary_merge
from twin.shared.memory.diary.codec import (
    importance_to_ttl,
    pack_embedding,
)
from twin.shared.memory.diary.gates import (
    gate_bm25_only_by_cosine,
    gate_by_similarity,
)
from twin.shared.memory.diary.idempotency import (
    marker_key as _idempotency_marker_key,
)
from twin.shared.memory.diary.idempotency import (
    normalize_idempotency_key,
)
from twin.shared.memory.diary.reader import DiaryReader, time_filter_clause
from twin.shared.memory.diary.schema import (
    create_timeline_index,
    ensure_diary_fields,
    extract_indexed_dim,
)
from twin.shared.memory.diary.writer import DiaryWriter
from twin.shared.memory.vn_time import vn_day_str

logger = logging.getLogger(__name__)


class TimelineSummary:
    """A timeline summary entry — one topic from one consolidation pass."""

    def __init__(
        self,
        user_id: str,
        summary: str,
        embedding: list[float],
        topic: str = "general",
        topic_display: str = "",
        importance: int = 3,
        version: int = 2,
        summary_id: str | None = None,
        created_at: datetime | None = None,
    ):
        self.summary_id = summary_id or str(uuid.uuid4())
        self.user_id = user_id
        self.summary = summary
        self.topic = topic
        self.topic_display = topic_display
        self.embedding = embedding
        self.importance = importance
        self.version = version
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "user_id": self.user_id,
            "summary": self.summary,
            "topic": self.topic,
            "topic_display": self.topic_display,
            "embedding": self.embedding,
            "importance": self.importance,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
        }


class TimelineSummaryStore:
    """Vector + BM25 store for timeline summaries (T2). Thin orchestrator:
    reads -> DiaryReader, commits -> DiaryWriter, gating -> gates.py.
    A detected index-dim mismatch fails closed (raise, HASHes untouched).
    """

    def __init__(
        self,
        redis_client: Any,
        embedding_dim: int = 384,
        embedding_service: Any = None,
    ):
        self.redis = redis_client
        self.embedding_dim = embedding_dim
        # Only re-embeds merged text on diary-merge writes; None disables
        # merging (append-only), keeping service-less callers working.
        self.embedding_service = embedding_service
        self.index_name = "timeline_summaries"
        self.prefix = "timeline:summary"
        # Flipped False only when initialize() positively detects an index
        # DIM mismatch; unknown (never initialized) stays usable.
        self._index_usable = True
        self._reader = DiaryReader(
            redis_client, index_name=self.index_name, prefix=self.prefix,
        )
        self._writer = DiaryWriter(redis_client, prefix=self.prefix)

    def _require_usable_index(self, op: str) -> None:
        if not self._index_usable:
            raise RuntimeError(
                f"T2 {op} refused: timeline index DIM != embedding_dim="
                f"{self.embedding_dim} (reindex required; index and HASHes "
                "left untouched — this process will not drop or rewrite them)"
            )

    # ---------------------------------------------------------------- init

    async def initialize(self) -> None:
        """Create Redis index (schema v3) if not exists."""
        try:
            info = await self.redis.execute_command("FT.INFO", self.index_name)
            logger.info("Timeline index already exists")
            indexed_dim = extract_indexed_dim(info)
            if indexed_dim is not None and indexed_dim != self.embedding_dim:
                logger.error(
                    "Timeline index %r is indexed with DIM=%d but the configured "
                    "embedding_dim=%d — writes/searches will silently fail to index "
                    "or will target the wrong vector space. A reindex is required "
                    "(this will NOT auto-drop the index).",
                    self.index_name, indexed_dim, self.embedding_dim,
                )
                self._index_usable = False
            else:
                self._index_usable = True
            await ensure_diary_fields(self.redis, self.index_name, info)
        except Exception:
            await create_timeline_index(
                self.redis, self.index_name, self.prefix, self.embedding_dim,
            )
            self._index_usable = True

    # ---------------------------------------------------------------- write

    async def store_summary(
        self,
        user_id: str,
        summary: str,
        embedding: list[float],
        *,
        topic: str = "general",
        topic_display: str = "",
        importance: int = 3,
        period_start: float | None = None,
        period_end: float | None = None,
        source_entry_ids: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Store a topic summary as a diary entry; return summary_id.

        Same-day near-duplicates (cosine >= T2_MERGE_MIN_COSINE, same
        normalized topic) merge into the existing doc instead of appending
        (needs embedding_service). ValueError on embedding dim mismatch —
        storing anyway silently breaks RediSearch indexing.

        idempotency_key (optional, #6): same-key retries return the
        original summary_id. Marker commits atomically with HASH+TTL
        under WATCH/CAS and is honored only while its HASH EXISTS;
        needs a transactional Redis client, fails closed without one.
        """
        self._require_usable_index("store_summary")
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"store_summary: embedding dim mismatch — got {len(embedding)}, "
                f"expected {self.embedding_dim} (user={user_id})"
            )
        idem = normalize_idempotency_key(idempotency_key)

        now_ts = datetime.now(timezone.utc).timestamp()
        ps = float(period_start) if period_start is not None else now_ts
        pe = float(period_end) if period_end is not None else ps
        day = vn_day_str(ps)
        entry_ids = [str(x) for x in (source_entry_ids or [])]

        marker: str | None = None
        if idem is not None:
            if not self._writer.supports_transactions():
                raise RuntimeError(
                    "T2 idempotent store needs a transactional Redis client "
                    "(pipeline/WATCH); refusing a non-atomic marker write"
                )
            marker = _idempotency_marker_key(user_id, topic, entry_ids, idem)
            winner = await self._writer.verified_claim(marker)
            if winner is not None:
                return winner

        if self.embedding_service is not None:
            merged_id = await try_diary_merge(
                self,
                user_id=user_id,
                day=day,
                summary=summary,
                embedding=embedding,
                importance=importance,
                period_start=ps,
                period_end=pe,
                source_entry_ids=entry_ids,
                topic=topic,
                topic_display=topic_display,
                marker_key=marker,
                writer=self._writer,
            )
            if merged_id is not None:
                return merged_id

        entry = TimelineSummary(
            user_id=user_id,
            summary=summary,
            embedding=embedding,
            topic=topic,
            topic_display=topic_display,
            importance=importance,
        )

        key = f"{self.prefix}:{entry.summary_id}"
        ttl_seconds = importance_to_ttl(importance) * 86400
        winner = await self._writer.append_doc(
            key,
            mapping={
                "user_id":       user_id,
                "summary":       summary,
                "topic":         topic,
                "topic_display": topic_display,
                "importance":    importance,
                "created_at":    entry.created_at.timestamp(),
                "version":       entry.version,
                "merge_version": 0,
                "day":           day,
                "period_start":  ps,
                "period_end":    pe,
                "source_entry_ids": json.dumps(entry_ids, ensure_ascii=False),
                "embedding":     pack_embedding(embedding),
            },
            ttl_seconds=ttl_seconds,
            marker_key=marker,
            marker_value=entry.summary_id,
            marker_ttl_seconds=ttl_seconds,
        )
        if winner is not None:
            return winner

        logger.info(
            "Stored T2 summary %s topic=%s user=%s day=%s",
            entry.summary_id, topic, user_id, day,
        )
        return entry.summary_id

    # ---------------------------------------------------------------- search

    async def search(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 5,
        *,
        query_text: str | None = None,
        topic_filter: str | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """KNN-only, or hybrid (KNN + BM25 fused via RRF) with query_text.
        query_embedding is caller-composed (query prefix + text).
        """
        self._require_usable_index("search")
        # Conditional so fakes with the pre-P3.2 4-arg signature keep working.
        time_kwargs: dict[str, float] = {}
        if since_ts is not None:
            time_kwargs["since_ts"] = since_ts
        if until_ts is not None:
            time_kwargs["until_ts"] = until_ts

        knn_results = await self._search_knn(
            user_id, query_embedding, limit, topic_filter, **time_kwargs,
        )

        if not query_text:
            return self._gate_by_similarity(knn_results)

        bm25_results = await self._search_bm25(
            user_id, query_text, limit, topic_filter, **time_kwargs,
        )

        # Fuse untruncated, then gate, THEN limit — truncating first would
        # let a stripped gated doc evict a valid doc below the limit.
        fused = _rrf_fuse(
            knn_results, bm25_results, limit=len(knn_results) + len(bm25_results),
        )

        # Strip KNN-gated ids from the fused output so BM25 can't smuggle
        # them back in; BM25-only docs pass through to the cosine check below.
        kept_ids = {d.get("summary_id") for d in self._gate_by_similarity(knn_results)}
        gated_ids = {d.get("summary_id") for d in knn_results} - kept_ids
        fused = [d for d in fused if d.get("summary_id") not in gated_ids]

        # P3.5 (fix B3): score BM25-only docs in Python against the SAME
        # floor (fail-closed on bad embeddings, #25) so BM25 is ranking-only.
        fused = self._gate_bm25_only_by_cosine(fused, knn_results, query_embedding)

        return fused[:limit]

    @staticmethod
    def _time_filter_clause(since_ts: float | None, until_ts: float | None) -> str | None:
        """RediSearch OR-fallback time filter; implementation lives in reader."""
        return time_filter_clause(since_ts, until_ts)

    def _gate_bm25_only_by_cosine(
        self,
        fused: list[dict[str, Any]],
        knn_results: list[dict[str, Any]],
        query_embedding: list[float],
    ) -> list[dict[str, Any]]:
        return gate_bm25_only_by_cosine(fused, knn_results, query_embedding)

    async def _search_knn(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int,
        topic_filter: str | None,
        *,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        return await self._reader.search_knn(
            user_id, query_embedding, limit, topic_filter,
            since_ts=since_ts, until_ts=until_ts,
        )

    def _gate_by_similarity(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return gate_by_similarity(results)

    async def _search_bm25(
        self,
        user_id: str,
        query_text: str,
        limit: int,
        topic_filter: str | None,
        *,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        return await self._reader.search_bm25(
            user_id, query_text, limit, topic_filter,
            since_ts=since_ts, until_ts=until_ts,
        )

    # ---------------------------------------------------------------- get_recent

    async def get_recent(
        self,
        user_id: str,
        limit: int = 10,
        *,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """Recent summaries, newest content-time first (bounds behave as in
        search(); see _time_filter_clause)."""
        self._require_usable_index("get_recent")
        return await self._reader.get_recent(
            user_id, limit, since_ts=since_ts, until_ts=until_ts,
        )


# -------------------------------------------------------------------- RRF

def _rrf_fuse(
    knn_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    *,
    k: int = 60,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion of two result lists.

    score(d) = sum(1 / (k + rank)) across lists that contain d.
    Dedup by summary_id; KNN dict wins on field conflict.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}

    for rank, doc in enumerate(knn_results, start=1):
        sid = doc.get("summary_id", "")
        if not sid:
            continue
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
        docs.setdefault(sid, doc)

    for rank, doc in enumerate(bm25_results, start=1):
        sid = doc.get("summary_id", "")
        if not sid:
            continue
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
        # merge: keep KNN dict as base, add any missing fields from BM25
        if sid not in docs:
            docs[sid] = doc
        else:
            for k2, v in doc.items():
                docs[sid].setdefault(k2, v)

    ranked = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
    return [docs[sid] for sid in ranked[:limit]]
