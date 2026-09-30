"""Stable idempotency keys for consolidation T2 writes."""
from __future__ import annotations

import hashlib


def build_idempotency_key(
    scope: str, scope_id: str, entry_ids: list[str], topic: str,
) -> str:
    """Deterministic key from scope + sorted source IDs + normalized topic."""
    scope_norm = str(scope or "user").strip()
    sid_norm = str(scope_id or "").strip()
    ids_norm = sorted(str(x) for x in (entry_ids or []) if str(x))
    topic_norm = str(topic or "").strip().lower()
    payload = "\x1f".join([scope_norm, sid_norm, ",".join(ids_norm), topic_norm])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"consol:{digest}"
