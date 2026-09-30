"""Pure encode/decode helpers for T2 timeline summaries — no Redis I/O.

Embedding pack/unpack, FT.SEARCH result parsing (dict and RESP2 flat
pair-list shapes), RediSearch TAG-value escaping, and the
importance→TTL retention policy.
"""
from __future__ import annotations

import json
import re
import struct
from typing import Any


def pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def unpack_embedding(value: Any) -> list[float]:
    """Unpack FLOAT32 bytes to a Python list of floats.

    Never text-decodes: FLOAT32 bytes that happen to be UTF-8-valid (e.g.
    ``struct.pack('8f', *([0.5] * 8))``) must round-trip to floats, not
    collapse to ``[]`` via a str detour (#25). Anything that is not a raw
    ``4N``-byte buffer (or an honest float list) decodes to ``[]`` — the
    cosine gates treat ``[]`` as unverifiable and fail closed.
    """
    if isinstance(value, list):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return []
    if not isinstance(value, (bytes, bytearray)):
        return []
    if len(value) == 0 or len(value) % 4 != 0:
        return []
    count = len(value) // 4
    try:
        return list(struct.unpack(f"{count}f", value[: count * 4]))
    except struct.error:
        return []


def decode_fields(mapping: Any) -> dict[str, Any]:
    """Decode Redis hash fields to native types; unpack embedding bytes.

    Accepts a dict OR the flat RESP2 ``[k1, v1, k2, v2, ...]`` pair list
    (production client speaks RESP2) — without this branch every T2 read
    crashed here and was swallowed into an empty result (B8).
    """
    if isinstance(mapping, (list, tuple)):
        pairs = list(zip(mapping[0::2], mapping[1::2]))
    else:
        pairs = mapping.items()
    result: dict[str, Any] = {}
    for k, v in pairs:
        field_name = k.decode() if isinstance(k, bytes) else k
        if field_name == "embedding":
            # Unpack from the RAW bytes before any text decoding (#25): a
            # blanket v.decode() first turns UTF-8-valid FLOAT32 buffers
            # into str, and unpack then yields [] — silently disarming the
            # cosine gate for BM25-only docs.
            result[field_name] = unpack_embedding(v)
            continue
        try:
            field_value: Any = v.decode() if isinstance(v, bytes) else v
        except (UnicodeDecodeError, AttributeError):
            field_value = v

        if field_name == "score":
            try:
                field_value = float(field_value)
            except (TypeError, ValueError):
                pass
        elif field_name in {"importance", "version", "merge_version"}:
            try:
                field_value = int(field_value)
            except (TypeError, ValueError):
                pass
        elif field_name in {"created_at", "period_start", "period_end"}:
            try:
                field_value = float(field_value)
            except (TypeError, ValueError):
                pass
        elif field_name == "source_entry_ids":
            try:
                loaded = json.loads(field_value)
                field_value = loaded if isinstance(loaded, list) else []
            except (TypeError, ValueError):
                field_value = []

        result[field_name] = field_value
    return result


def parse_results(
    results: Any,
    prefix: str,
    *,
    has_scores: bool = False,
) -> list[dict[str, Any]]:
    """Parse raw FT.SEARCH results (both dict and list format, with/without scores)."""
    summaries: list[dict[str, Any]] = []

    if isinstance(results, dict):
        raw = results.get(b"results") or results.get("results") or []
        for item in raw:
            key = item.get(b"id") or item.get("id")
            extra = item.get(b"extra_attributes") or item.get("extra_attributes") or {}
            d = decode_fields(extra)
            if key:
                key_str = key.decode() if isinstance(key, bytes) else key
                d["summary_id"] = key_str.replace(f"{prefix}:", "")
            # backward compat: expose 'content' alias for old readers
            if "summary" in d and "content" not in d:
                d["content"] = d["summary"]
            summaries.append(d)
        return summaries

    # List format: [count, key, [fields...], key, [fields...], ...]
    # With scores: [count, key, score, [fields...], ...]
    i = 1
    while i < len(results):
        key = results[i]
        i += 1

        score = None
        if has_scores and i < len(results) and not isinstance(results[i], list):
            try:
                score = float(results[i])
                i += 1
            except (TypeError, ValueError):
                pass

        if i < len(results) and isinstance(results[i], list):
            fields = results[i]
            i += 1
        else:
            continue

        d: dict[str, Any] = decode_fields(fields)

        if key:
            key_str = key.decode() if isinstance(key, bytes) else key
            d["summary_id"] = key_str.replace(f"{prefix}:", "")

        if score is not None:
            d["_score"] = score

        # backward compat alias
        if "summary" in d and "content" not in d:
            d["content"] = d["summary"]

        summaries.append(d)

    return summaries


def escape_tag_value(value: str) -> str:
    """Escape RediSearch TAG-query special chars (e.g. '-' in a day tag
    like '2026-07-03', which the parser would otherwise treat as
    punctuation). Backslash-escapes every non-word char (redis-py
    TagField's documented-safe superset).
    """
    return re.sub(r"([^\w])", r"\\\1", value)


def importance_to_ttl(importance: int) -> int:
    return {5: 365, 4: 180, 3: 90, 2: 30, 1: 7}.get(importance, 90)
