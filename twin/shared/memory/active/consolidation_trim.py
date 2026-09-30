"""Guarded consolidation trim helper (OwnT1 atomic, no raw persistence).

Fetches exact raw JSON snapshots (transient ARGV only), parses ActiveEntry
from the SAME bytes and verifies pinned SHA64, then runs a single EVAL that
re-validates caller identity + snapshots before mutating. Raw never persisted
in journals or side keys. Archive uses only validated snapshots after success.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _decode_raw(value: Any) -> str | None:
    if value is None or value is False:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        return value or None
    return str(value) or None


async def fetch_raw_snapshots(store: Any, scope: str, sid: str,
                              entry_ids: list[str]) -> list[str | None]:
    """Exact JSON.GET per ID in order; None marks missing/undecodable."""
    out: list[str | None] = []
    for eid in entry_ids or []:
        key = store._entry_key(scope, sid, eid)  # noqa: SLF001
        try:
            raw = await store.redis.execute_command("JSON.GET", key)
        except Exception:
            out.append(None)
            continue
        out.append(_decode_raw(raw))
    return out


async def guarded_trim_batch(store: Any, scope: str, scope_id: str,
                             record: dict[str, Any], keep_recent: int) -> dict:
    """Validate snapshots then atomically trim + mark trimmed (single EVAL)."""
    # Local imports avoid cycles at module load.
    from twin.shared.memory.active.models import ActiveEntry
    from twin.shared.memory.consolidation_journal import entry_hashes_of

    entry_ids = list(record.get("entry_ids") or [])
    fingerprints = list(record.get("fingerprints") or [])
    plan_key = str(record.get("plan_key") or "")
    nonce = str(record.get("caller_nonce") or "")
    gen = str(record.get("receiver_generation") or "")
    trim_ids = list(record.get("trim_ids") or [])
    ack = record.get("ack")
    if (
        not entry_ids or len(entry_ids) > 200 or len(entry_ids) != len(set(entry_ids))
        or len(fingerprints) != len(entry_ids)
        or not plan_key or len(nonce) != 32 or not isinstance(ack, dict)
        or not trim_ids or set(trim_ids) != set(entry_ids)
    ):
        return {"status": "failed", "reason": "bad_record"}
    raws = await fetch_raw_snapshots(store, scope, scope_id, entry_ids)
    entries: list[Any] = []
    snapshots: list[str]
    if any(r is None for r in raws):
        # Missing: EVAL decides already (ignore snaps) vs t1_missing. Real
        # where present, dummy where missing; never fail fast (would miss
        # already-trimmed convergence when T1 was legitimately cleared).
        snapshots = [r if r is not None else "{}" for r in raws]
    else:
        try:
            for raw in raws:
                assert raw is not None
                entries.append(ActiveEntry.model_validate(json.loads(raw)))
            pinned_ok = entry_hashes_of(entries) == fingerprints
        except Exception:
            pinned_ok = False
            entries = []
        if pinned_ok:
            snapshots = [str(r) for r in raws if r is not None]
        else:
            # Mutated vs pinned: dummy snaps so acknowledged fails via EVAL
            # (t1_changed) while already-trimmed still converges (ignores).
            # Passing real mismatched snaps would bypass the pinned check.
            snapshots = ["{}" for _ in entry_ids]
            entries = []
    # Single EVAL: Lua re-validates caller + raw snapshots before mutation.
    res = await store.trim_summarized_guarded(
        scope, scope_id, expected_ids=entry_ids, expected_hashes=fingerprints,
        expected_plan=plan_key, expected_nonce=nonce, expected_gen=gen,
        expected_trim_ids=trim_ids, expected_ack=ack,
        snapshots=snapshots,
        keep_recent=int(keep_recent),
    )
    if not isinstance(res, dict) or res.get("status") != "trimmed":
        return res if isinstance(res, dict) else {"status": "failed", "reason": "guard_rejected"}
    # Best-effort cold archive of ONLY validated deleted snapshots.
    try:
        from twin.shared.config.settings import Config

        if bool(getattr(Config, "T1_ARCHIVE_ENABLED", True)):
            by_id = {e.entry_id: e for e in entries}
            doomed = [by_id[i] for i in (res.get("deleted_ids") or []) if i in by_id]
            if doomed:
                ttl_days = int(getattr(Config, "T1_ARCHIVE_TTL_DAYS", 90))
                await store.archive_entries(
                    scope, scope_id, doomed, ttl_seconds=ttl_days * 86400,
                )
    except Exception as exc:
        logger.warning(
            "T1: guarded archive failed scope=%s/%s -- trim already durable: %s",
            scope, scope_id, exc,
        )
    return res
