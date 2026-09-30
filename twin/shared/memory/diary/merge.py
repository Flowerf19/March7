"""Same-day diary merge for T2 timeline summaries (write-path P2.2).

Merges a new summary into an existing same-user+same-day doc when
cosine similarity clears T2_MERGE_MIN_COSINE, instead of always
appending. Takes the store duck-typed (not TimelineSummaryStore) to
avoid a circular import with store.py.

Concurrency (#8): the read -> await re-embed -> write race commits
through WATCH/CAS (writer.commit_merge). On MergeConflict the merge is
recomputed from a fresh read with a fresh re-embed, bounded by
MERGE_MAX_ATTEMPTS; when the budget is exhausted the write falls back
to appending a safe new summary. Either way both texts and both
provenance sets survive — success is never reported for a lost update.

Topic (#34): only same-normalized-topic docs merge, so new content can
never land under a stale topic label; the merged doc keeps its topic
and takes the new topic_display when one is provided.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from twin.shared.config.settings import Config
from twin.shared.memory.diary.codec import (
    escape_tag_value,
    importance_to_ttl,
    pack_embedding,
    parse_results,
)
from twin.shared.memory.diary.writer import (
    DiaryWriter,
    MergeConflict,
    parse_merge_version,
)

logger = logging.getLogger(__name__)

# Merge attempts per store_summary call (each re-reads + re-embeds); past
# this the write appends instead of merging. Monkeypatchable in tests.
MERGE_MAX_ATTEMPTS = 3


def normalize_topic(topic: Any) -> str:
    """Case/whitespace-insensitive topic identity for merge gating."""
    return (topic or "").strip().casefold() if isinstance(topic, str) else ""


async def try_diary_merge(
    store: Any,
    *,
    user_id: str,
    day: str,
    summary: str,
    embedding: list[float],
    importance: int,
    period_start: float,
    period_end: float,
    source_entry_ids: list[str],
    topic: str = "general",
    topic_display: str = "",
    marker_key: str | None = None,
    writer: DiaryWriter | None = None,
) -> str | None:
    """Merge `summary` into the best same-user same-day doc if similar
    enough; return the kept summary_id, or None to append instead.
    Best-effort: any lookup/re-embed hiccup falls back to append rather
    than risk corrupting an existing doc; CAS conflicts retry then append
    rather than report a lost update as success.
    """
    for attempt in range(1, MERGE_MAX_ATTEMPTS + 1):
        merged_id = await _try_merge_once(
            store,
            user_id=user_id,
            day=day,
            summary=summary,
            embedding=embedding,
            importance=importance,
            period_start=period_start,
            period_end=period_end,
            source_entry_ids=source_entry_ids,
            topic=topic,
            topic_display=topic_display,
            marker_key=marker_key,
            writer=writer,
            attempt=attempt,
        )
        # _try_merge_once returns _RETRY only on CAS conflict; any other
        # outcome (merged id, raced-winner id, or None=append) is final.
        if merged_id is not _RETRY:
            return merged_id
    logger.warning(
        "Diary merge conflict budget exhausted (user=%s day=%s) — "
        "appending a new doc so no update is lost",
        user_id, day,
    )
    return None


class _Retry:
    pass


_RETRY = _Retry()


async def _try_merge_once(
    store: Any,
    *,
    user_id: str,
    day: str,
    summary: str,
    embedding: list[float],
    importance: int,
    period_start: float,
    period_end: float,
    source_entry_ids: list[str],
    topic: str,
    topic_display: str,
    marker_key: str | None,
    writer: DiaryWriter | None,
    attempt: int,
) -> str | None | _Retry:
    candidate = await find_diary_merge_candidate(store, user_id, day, embedding)
    if not candidate or not candidate.get("summary_id"):
        return None

    if normalize_topic(candidate.get("topic")) != normalize_topic(topic):
        logger.info(
            "Diary merge skipped (topic %r != %r) — appending new doc "
            "(user=%s day=%s target=%s)",
            candidate.get("topic"), topic, user_id, day,
            candidate.get("summary_id"),
        )
        return None

    dist = candidate.get("score")
    if dist is None:
        return None
    try:
        similarity = 1.0 - float(dist)
    except (TypeError, ValueError):
        logger.warning(
            "Diary merge skipped (unparseable candidate score=%r) — appending",
            dist,
        )
        return None
    min_cos = float(getattr(Config, "T2_MERGE_MIN_COSINE", 0.60))
    if similarity < min_cos:
        return None

    old_summary = str(candidate.get("summary") or "")
    merged_text = f"{old_summary}\n{summary}" if old_summary else summary
    max_chars = int(getattr(Config, "T2_MERGE_MAX_CHARS", 1500))
    if len(merged_text) > max_chars:
        logger.info(
            "Diary merge skipped (would exceed %d chars) — appending new doc "
            "(user=%s day=%s target=%s)",
            max_chars, user_id, day, candidate["summary_id"],
        )
        return None

    try:
        merged_embedding = await store.embedding_service.get_embedding(
            f"{Config.EMBEDDING_PASSAGE_PREFIX}{merged_text}"
        )
    except Exception as exc:
        logger.warning("Diary merge re-embed failed — appending instead: %s", exc)
        return None
    if not merged_embedding or len(merged_embedding) != store.embedding_dim:
        logger.warning(
            "Diary merge re-embed returned dim=%s (expected %d) — appending instead",
            len(merged_embedding or []), store.embedding_dim,
        )
        return None

    fields, ttl_seconds = _merged_fields(
        candidate,
        importance=importance,
        period_start=period_start,
        period_end=period_end,
        source_entry_ids=source_entry_ids,
        topic_display=topic_display,
    )
    summary_id = str(candidate["summary_id"])
    key = f"{store.prefix}:{summary_id}"
    mapping = {
        "summary": merged_text,
        "embedding": pack_embedding(merged_embedding),
        **fields,
    }
    expected_version = parse_merge_version(candidate)
    active_writer = writer or DiaryWriter(store.redis, prefix=store.prefix)
    try:
        winner = await active_writer.commit_merge(
            key, summary_id, expected_version, mapping, ttl_seconds,
            marker_key=marker_key, marker_ttl_seconds=ttl_seconds,
        )
    except MergeConflict as exc:
        logger.info(
            "Diary merge conflict (attempt %d, target=%s): %s — retrying",
            attempt, summary_id, exc,
        )
        return _RETRY
    logger.info(
        "Merged T2 diary summary %s user=%s day=%s cosine=%.3f chars=%d",
        summary_id, user_id, day, similarity, len(merged_text),
    )
    return winner or summary_id


def _merged_fields(
    candidate: dict[str, Any],
    *,
    importance: int,
    period_start: float,
    period_end: float,
    source_entry_ids: list[str],
    topic_display: str,
) -> tuple[dict[str, Any], int]:
    """Merged HASH fields (minus summary/embedding): max importance,
    unioned span, order-preserving provenance union, refreshed display
    label. `topic` itself is intentionally left untouched — the caller
    only merges normalized-equal topics, so the stored label stays valid.
    """
    old_importance = int(candidate.get("importance") or 3)
    new_importance = max(int(importance), old_importance)
    old_ps = candidate.get("period_start")
    old_pe = candidate.get("period_end")
    merged_ps = min(float(old_ps), period_start) if old_ps is not None else period_start
    merged_pe = max(float(old_pe), period_end) if old_pe is not None else period_end
    old_ids = candidate.get("source_entry_ids")
    if not isinstance(old_ids, list):
        old_ids = []
    # Order-preserving union: old provenance first, then the new batch.
    merged_ids = list(dict.fromkeys([*map(str, old_ids), *source_entry_ids]))
    new_display = (topic_display or "").strip()
    merged_display = new_display or str(candidate.get("topic_display") or "")
    fields = {
        "importance": new_importance,
        "period_start": merged_ps,
        "period_end": merged_pe,
        "source_entry_ids": json.dumps(merged_ids, ensure_ascii=False),
        "topic_display": merged_display,
    }
    return fields, importance_to_ttl(new_importance) * 86400


async def find_diary_merge_candidate(
    store: Any, user_id: str, day: str, embedding: list[float],
) -> dict[str, Any] | None:
    """Nearest same-user same-day doc (KNN 1), or None. The @day TAG
    filter is what enforces "never merge across days" — docs from other
    days (and old v2 docs, which have no day field at all) can't match.
    Topic equality is checked in Python after the fetch (TAG matching is
    case-sensitive; merge identity is normalize_topic).
    """
    query = (
        f"(@user_id:{{{escape_tag_value(user_id)}}} @day:{{{escape_tag_value(day)}}})"
        f"=>[KNN 1 @embedding $vec AS score]"
    )
    try:
        results = await store.redis.execute_command(
            "FT.SEARCH", store.index_name,
            query,
            "PARAMS", "2", "vec", pack_embedding(embedding),
            "SORTBY", "score", "ASC",
            "LIMIT", "0", "1",
            "DIALECT", "2",
        )
        parsed = parse_results(results, store.prefix)
        return parsed[0] if parsed else None
    except Exception as exc:
        logger.warning("Diary merge lookup failed — appending instead: %s", exc)
        return None
