"""Cosine-gate policy for T2 retrieval — pure functions, no Redis I/O.

Both gates enforce the ``T2_MIN_COSINE`` floor. Fail-closed rule (#25): a
doc whose embedding is missing, wrong-dimensional, non-numeric, or
non-finite can never satisfy the floor, so when the gate is active it is
dropped — never kept as "unscorable". With the floor at 0.0 both gates
are a no-op (default config behavior unchanged).
"""
from __future__ import annotations

import logging
import math
from typing import Any

from twin.shared.config.settings import Config
from twin.shared.llm.embedding.embedding_trace_logger import cosine_similarity

logger = logging.getLogger(__name__)


def is_valid_embedding(value: Any, dim: int) -> bool:
    """True iff `value` is a finite float vector of exactly `dim`."""
    if not isinstance(value, list) or dim <= 0 or len(value) != dim:
        return False
    return all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        for x in value
    )


def gate_by_similarity(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop KNN hits below the cosine-similarity floor.

    score = 1 - cosine_similarity (COSINE index). Without this gate, a
    near-empty or off-topic T2 store still injects top-K noise into
    every prompt. No-op when the floor is 0.0 or a hit has no score; a
    hit with an unparseable/non-finite score is dropped, not trusted.
    """
    min_cos = getattr(Config, "T2_MIN_COSINE", 0.0)
    if min_cos <= 0.0:
        return results
    kept: list[dict[str, Any]] = []
    for doc in results:
        dist = doc.get("score")
        if dist is None:
            kept.append(doc)
            continue
        try:
            similarity = 1.0 - float(dist)
        except (TypeError, ValueError):
            logger.debug(
                "T2 gate: drop summary_id=%s (unparseable score=%r)",
                doc.get("summary_id"), dist,
            )
            continue
        if not math.isfinite(similarity):
            logger.debug(
                "T2 gate: drop summary_id=%s (non-finite score=%r)",
                doc.get("summary_id"), dist,
            )
            continue
        if similarity >= min_cos:
            kept.append(doc)
        else:
            logger.debug(
                "T2 gate: drop summary_id=%s cosine=%.3f < %.2f",
                doc.get("summary_id"), similarity, min_cos,
            )
    return kept


def gate_bm25_only_by_cosine(
    fused: list[dict[str, Any]],
    knn_results: list[dict[str, Any]],
    query_embedding: list[float],
) -> list[dict[str, Any]]:
    """Apply the T2_MIN_COSINE floor to BM25-only docs (P3.5, fix B3).

    No-op when the floor is 0.0. Docs already seen by KNN were gated
    above; only docs reachable solely through BM25 are scored here. A
    BM25-only doc with a missing/invalid/wrong-dimensional/non-finite
    embedding is DROPPED while the gate is active (#25) — keeping it
    would let BM25 bypass the cosine floor entirely.
    """
    min_cos = getattr(Config, "T2_MIN_COSINE", 0.0)
    if min_cos <= 0.0:
        return fused
    knn_ids = {d.get("summary_id") for d in knn_results}
    query_dim = len(query_embedding) if isinstance(query_embedding, list) else 0
    query_ok = is_valid_embedding(query_embedding, query_dim)
    kept: list[dict[str, Any]] = []
    for doc in fused:
        sid = doc.get("summary_id")
        if sid in knn_ids:
            kept.append(doc)
            continue
        embedding = doc.get("embedding")
        if not query_ok or not is_valid_embedding(embedding, query_dim):
            logger.debug(
                "T2 gate (BM25-only): drop summary_id=%s "
                "(missing/invalid embedding, floor=%.2f)",
                sid, min_cos,
            )
            continue
        try:
            similarity = cosine_similarity(query_embedding, embedding)
        except ValueError:
            logger.debug(
                "T2 gate (BM25-only): drop summary_id=%s (dim mismatch)",
                sid,
            )
            continue
        if not math.isfinite(similarity) or similarity < min_cos:
            logger.debug(
                "T2 gate (BM25-only): drop summary_id=%s cosine=%.3f < %.2f",
                sid, similarity, min_cos,
            )
            continue
        kept.append(doc)
    return kept
