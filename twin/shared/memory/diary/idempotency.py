"""Idempotency-marker protocol for T2 writes (durable side of #6).

``store_summary(..., idempotency_key=...)`` binds one marker key to one
logical write. The marker lives in Redis (never in-memory-only) and is
always committed atomically with the summary HASH + TTL under WATCH/CAS
(see writer.py) — a retry that finds a claimed marker whose HASH still
EXISTS returns the original summary_id instead of appending/merging a
second copy. A claimed marker whose HASH is gone is stale (TTL race or
eviction): the retry re-stores instead of falsely acknowledging.

Marker identity binds scope + normalized topic + exact source batch +
the caller's key, so two different batches can never share a marker
even if a caller reuses a key string.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

MARKER_PREFIX = "timeline:idem"


def normalize_idempotency_key(value: Any) -> str | None:
    """Validate the caller-supplied key. None stays None (no idempotency);
    blank/non-str is a caller bug and fails closed with ValueError."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "store_summary: idempotency_key must be a non-blank str or None"
        )
    return value.strip()


def marker_key(
    scope_id: str, topic: str, source_entry_ids: list[str], idempotency_key: str,
) -> str:
    """Derive the Redis marker key bound to scope+topic+source batch+key."""
    topic_norm = (topic or "").strip().casefold()
    batch = json.dumps([str(x) for x in source_entry_ids], ensure_ascii=False)
    batch_hash = hashlib.sha1(batch.encode("utf-8")).hexdigest()[:16]
    return f"{MARKER_PREFIX}:{scope_id}:{topic_norm}:{batch_hash}:{idempotency_key}"
