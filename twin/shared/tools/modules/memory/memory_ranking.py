"""Cross-scope ranking policy for T2 recall (pure functions, no I/O).

Dual user/channel retrieval returns a GLOBALLY ranked top-K (#32): both
scopes are searched with the same query embedding, so KNN cosine
distances are directly comparable across scopes and the merged list is
ordered by relevance — not by scope. Ties keep scope order (user scope
first) deterministically via a stable sort.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional

from twin.shared.llm.embedding.embedding_trace_logger import cosine_similarity
from twin.shared.tools.modules.memory.memory_format import memory_field

logger = logging.getLogger(__name__)


def dedup_key(memory: Any) -> Any:
    """Identity for cross-scope dedup: summary_id, else summary text."""
    return (
        memory_field(memory, "summary_id")
        or memory_field(memory, "summary")
        or id(memory)
    )


def recency_key(memory: Any) -> float:
    """Content-time sort key (#33): period_end, legacy created_at fallback."""
    for field_name in ("period_end", "created_at"):
        value = memory_field(memory, field_name)
        if isinstance(value, datetime):
            return value.timestamp()
        try:
            stamp = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(stamp):
            return stamp
    return 0.0


def rank_semantic_global(
    pairs: list[tuple[Any, str]],
    query_embedding: list[float],
    limit: int,
) -> list[tuple[Any, str]]:
    """Globally rank (memory, source-scope) pairs by cosine similarity desc.

    Rank signal prefers each doc's KNN distance (``score`` = 1 - cosine,
    exact and comparable across scopes since both scopes share the query
    embedding), recomputes cosine in Python for BM25-only docs with a
    usable embedding, and parks unscorable docs last in original order.
    """
    query_dim = len(query_embedding) if isinstance(query_embedding, list) else 0
    query_ok = (
        query_dim > 0
        and all(
            isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x)
            for x in query_embedding
        )
    )

    def _cosine(memory: Any) -> float | None:
        score = memory_field(memory, "score")
        if score is not None:
            try:
                sim = 1.0 - float(score)
            except (TypeError, ValueError):
                sim = float("nan")
            if math.isfinite(sim):
                return sim
        if not query_ok:
            return None
        embedding = memory_field(memory, "embedding")
        if (
            not isinstance(embedding, list)
            or len(embedding) != query_dim
            or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x)
                for x in embedding
            )
        ):
            return None
        try:
            sim = cosine_similarity(query_embedding, embedding)
        except ValueError:
            return None
        return sim if math.isfinite(sim) else None

    scored = [(_cosine(memory), memory, source) for memory, source in pairs]
    # Stable: unscored last in original relative order; ties keep scope order.
    scored.sort(key=lambda t: (t[0] is None, -(t[0] or 0.0)))
    return [(memory, source) for _, memory, source in scored[:limit]]


async def widen_search(
    round_fn: Callable[[Optional[float]], Coroutine[Any, Any, list[Any]]],
    days_back: int,
    *,
    since_ts_from_days: Callable[[int], float],
    multiplier: int = 3,
) -> tuple[list[Any], Optional[str]]:
    """Try `days_back`, then `days_back*multiplier`, then no filter at all —
    max 3 rounds, all within one tool call (the LLM still only sees one
    call). A round past the first labels its results so the model stays
    honest about the wider window it actually saw, instead of silently
    answering as if the original window had data.
    """
    widened_days = days_back * multiplier
    rounds: list[tuple[Optional[int], Optional[str]]] = [
        (days_back, None),
        (widened_days, (
            f"(không thấy trong {days_back} ngày — kết quả từ {widened_days} ngày)"
        )),
        (None, (
            f"(không thấy trong {days_back} ngày — kết quả từ toàn bộ)"
        )),
    ]
    for round_days, label in rounds:
        since_ts = (
            since_ts_from_days(round_days)
            if round_days is not None else None
        )
        memories = await round_fn(since_ts)
        if memories:
            return memories, label
    return [], None
