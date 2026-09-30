"""Atomic T2 write commits — DiaryWriter (I/O collaborator).

Every merge commit goes through WATCH/CAS on the real Redis key (#8):
the writer re-reads ``merge_version`` under WATCH and aborts with
MergeConflict when another worker landed first, so a read -> await
re-embed -> write race can never silently drop an update. Merge callers
retry on conflict (fresh read + fresh re-embed) and finally fall back
to appending a safe new summary — both texts always survive.

Idempotent writes additionally bind the marker key: marker SET+EXPIRE
commit in the SAME MULTI as the HASH+TTL, so a claimed marker always
refers to a durable write and a retry can never store twice (#6).

Clients without ``pipeline()`` (unit-test doubles) get the legacy
direct write for NON-idempotent paths only; idempotent writes fail
closed without a transactional client instead of pretending atomicity.
"""
from __future__ import annotations

import logging
from typing import Any

try:  # redis[hiredis] is a production dependency; keep module importable anyway
    from redis.exceptions import WatchError
except Exception:  # pragma: no cover - only when redis is not installed
    class WatchError(Exception):
        """Fallback so CAS error handling still type-checks without redis."""

logger = logging.getLogger(__name__)


class MergeConflict(Exception):
    """The merge target changed (or an atomic commit raced) — retry or append."""


def parse_merge_version(raw: Any) -> int:
    """merge_version from an HGETALL reply (bytes or str keys); missing or
    garbage normalizes to 0 (legacy / self-heal), never raises."""
    if isinstance(raw, dict):
        value = raw.get(b"merge_version", raw.get("merge_version"))
    else:  # flat pair list, same shape decode_fields already tolerates
        value = None
        items = list(raw) if isinstance(raw, (list, tuple)) else []
        for i in range(0, len(items) - 1, 2):
            k = items[i]
            k = k.decode() if isinstance(k, bytes) else k
            if k == "merge_version":
                value = items[i + 1]
                break
    if isinstance(value, bytes):
        try:
            value = value.decode()
        except UnicodeDecodeError:
            return 0
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


class DiaryWriter:
    """Executes T2 HASH writes: plain appends and CAS merge commits."""

    def __init__(self, redis_client: Any, *, prefix: str):
        self.redis = redis_client
        self.prefix = prefix

    # ------------------------------------------------------------- capability

    def supports_transactions(self) -> bool:
        return callable(getattr(self.redis, "pipeline", None))

    def doc_key(self, summary_id: str) -> str:
        return f"{self.prefix}:{summary_id}"

    # ------------------------------------------------------------- idempotency reads

    async def read_claim(self, marker_key: str) -> str | None:
        """Summary_id a marker currently claims, or None when unclaimed /
        undecodable (stale — the commit path overwrites it atomically)."""
        raw = await self.redis.get(marker_key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            try:
                raw = raw.decode()
            except UnicodeDecodeError:
                return None
        return str(raw) if raw else None

    async def doc_exists(self, key: str) -> bool:
        return bool(await self.redis.exists(key))

    async def verified_claim(self, marker_key: str) -> str | None:
        """Claimed summary_id whose HASH still EXISTS, else None. A marker
        without its doc (TTL race/eviction) must never acknowledge (#6)."""
        claimed = await self.read_claim(marker_key)
        if claimed is None:
            return None
        if await self.doc_exists(self.doc_key(claimed)):
            return claimed
        logger.info(
            "T2 idempotency: marker %s claims missing doc %s — re-storing",
            marker_key, claimed,
        )
        return None

    # ------------------------------------------------------------- append

    async def append_doc(
        self,
        key: str,
        mapping: dict[str, Any],
        ttl_seconds: int,
        *,
        marker_key: str | None = None,
        marker_value: str | None = None,
        marker_ttl_seconds: int | None = None,
    ) -> str | None:
        """Append a new HASH. Returns a raced-winner summary_id when an
        identical idempotent retry claimed the marker concurrently (caller
        reports that id — its text is already durable), else None.

        Idempotent appends REQUIRE a transactional client: marker and
        HASH+TTL commit in one MULTI under WATCH(marker). Without one,
        fail closed instead of risking a duplicate or orphan marker.
        """
        if marker_key is not None and not self.supports_transactions():
            raise RuntimeError(
                "T2 idempotent append needs a transactional Redis client "
                "(pipeline/WATCH); refusing to split marker and HASH writes"
            )
        if not self.supports_transactions():
            await self.redis.hset(key, mapping=mapping)
            await self.redis.expire(key, ttl_seconds)
            return None

        pipe = self.redis.pipeline(transaction=True)
        try:
            if marker_key is not None:
                await pipe.watch(marker_key)
                claimed = await pipe.get(marker_key)
                if claimed is not None:
                    winner = await self.verified_claim(marker_key)
                    if winner is not None:
                        return winner
                    # Stale marker — fall through and overwrite atomically.
                pipe.multi()
                pipe.hset(key, mapping=mapping)
                pipe.expire(key, ttl_seconds)
                pipe.set(marker_key, marker_value)
                pipe.expire(marker_key, marker_ttl_seconds)
                try:
                    await pipe.execute()
                except WatchError:
                    # Marker moved under us: an identical retry committed.
                    winner = await self.verified_claim(marker_key)
                    if winner is not None:
                        return winner
                    raise RuntimeError(
                        "T2 idempotent append aborted without a visible "
                        f"claim on {marker_key} — retry with the same key"
                    )
                return None
            pipe.multi()
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, ttl_seconds)
            await pipe.execute()
            return None
        finally:
            await _release(pipe)

    # ------------------------------------------------------------- CAS merge

    async def commit_merge(
        self,
        key: str,
        summary_id: str,
        expected_version: int,
        mapping: dict[str, Any],
        ttl_seconds: int,
        *,
        marker_key: str | None = None,
        marker_ttl_seconds: int | None = None,
    ) -> str | None:
        """CAS-commit merged fields onto `key`.

        Under WATCH(key[, marker]): abort with MergeConflict when the live
        merge_version differs from `expected_version` (a concurrent merge
        landed between our read and this commit), or when EXEC itself
        aborts. Returns a raced-winner id only when an identical
        idempotent retry already stored this exact batch.
        """
        if not self.supports_transactions():
            if marker_key is not None:
                raise RuntimeError(
                    "T2 idempotent merge needs a transactional Redis client "
                    "(pipeline/WATCH); refusing to split marker and HASH writes"
                )
            # Legacy best-effort path for non-transactional test doubles.
            await self.redis.hset(key, mapping=mapping)
            await self.redis.expire(key, ttl_seconds)
            return None

        pipe = self.redis.pipeline(transaction=True)
        try:
            watched = [key] if marker_key is None else [key, marker_key]
            await pipe.watch(*watched)
            current_version = parse_merge_version(await pipe.hgetall(key))
            if current_version != expected_version:
                raise MergeConflict(
                    f"{key}: merge_version moved {expected_version} -> "
                    f"{current_version} during re-embed"
                )
            if marker_key is not None:
                claimed = await pipe.get(marker_key)
                if claimed is not None:
                    winner = await self.verified_claim(marker_key)
                    if winner is not None:
                        return winner
                    # Stale marker — overwrite below, atomically with the merge.
            merged_mapping = dict(mapping)
            merged_mapping["merge_version"] = expected_version + 1
            pipe.multi()
            pipe.hset(key, mapping=merged_mapping)
            pipe.expire(key, ttl_seconds)
            if marker_key is not None:
                pipe.set(marker_key, summary_id)
                pipe.expire(marker_key, marker_ttl_seconds)
            try:
                await pipe.execute()
            except WatchError:
                raise MergeConflict(f"{key}: watched key modified during commit")
            return None
        finally:
            await _release(pipe)


async def _release(pipe: Any) -> None:
    """Return a WATCH pipeline to a clean state (best-effort)."""
    reset = getattr(pipe, "reset", None)
    if callable(reset):
        try:
            await reset()
            return
        except Exception:
            pass
    unwatch = getattr(pipe, "unwatch", None)
    if callable(unwatch):
        try:
            await unwatch()
        except Exception:
            pass
