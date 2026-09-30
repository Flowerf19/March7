"""Embedding-trace emission for T2 semantic searches (observability I/O).

SEARCH trace events (used for gate calibration) are emitted from the
recall path — the tool is the only T2 search path since automatic
preflight was removed, so they live here rather than in the manager.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from twin.shared.llm.embedding.embedding_trace_logger import (
    cosine_similarity,
    token_overlap,
)
from twin.shared.tools.modules.memory.memory_format import memory_field

logger = logging.getLogger(__name__)


def trace_semantic_results(
    embedding_service: Any,
    *,
    input_text: str,
    current_query: str,
    query_embedding: list[float],
    summaries: list[dict[str, Any]],
    scope_ids: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
) -> None:
    """Log every semantic search result for embedding model debugging.

    ``scope_ids``/``sources`` record which scopes the dual-scope search
    covered and which one each hit came from, without touching the fixed
    trace schema. No-op unless the service carries an enabled trace logger.
    """
    trace_logger = getattr(embedding_service, "trace_logger", None)
    if not trace_logger or not trace_logger.enabled:
        return

    for rank, summary in enumerate(summaries, start=1):
        matched_text = memory_field(summary, "summary") or memory_field(summary, "content", "")
        matched_embedding = memory_field(summary, "embedding")
        if matched_embedding and query_embedding:
            try:
                cs = cosine_similarity(query_embedding, matched_embedding)
            except (ValueError, TypeError):
                # Trace-only: mixed legacy dims / malformed vectors must not
                # abort recall; strict gates live in ranking/inference.
                cs = None
        else:
            cs = None

        extra = None
        if scope_ids is not None:
            extra = {
                "scope_ids_searched": scope_ids,
                "source_scope_id": sources[rank - 1] if sources and rank <= len(sources) else None,
            }

        try:
            embedding_service._trace_embedding_event(
                input_text=input_text,
                vector=query_embedding,
                raw_dim=None,
                latency_ms=0.0,
                cache_hit=True,
                event_type="SEARCH",
                query_text=current_query,
                matched_text=matched_text,
                cosine_similarity=cs,
                token_overlap=token_overlap(current_query, matched_text),
                action="KNN_RESULT" if rank == 1 else "KNN_CANDIDATE",
                knn_score=memory_field(summary, "score"),
                bm25_score=memory_field(summary, "_score"),
                rrf_rank=rank,
                extra=extra,
            )
        except Exception as exc:
            # Best-effort per hit: a throwing emitter must not abort recall
            # or skip later hits (_trace_embedding_event already guards its
            # own I/O; this covers the call itself).
            logger.warning(
                "SEARCH trace failed (best-effort, continuing): %s", exc,
            )
            continue
