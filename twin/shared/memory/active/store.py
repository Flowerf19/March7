"""Redis JSON-backed scope-aware store for T1 active memory."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from twin.shared.memory.active.lua import (
    T1_CLEAR_V1,
    T1_INCR_TOKENS_V1,
    T1_OBSERVE_V1,
    T1_TRIM_GUARDED_V1,
    T1_TRIM_V1,
)
from twin.shared.memory.active.models import ActiveEntry
from twin.shared.memory.vn_time import vn_day_str

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class ActiveStore:
    """Redis JSON-backed store, scope-aware (user|channel).

    Keys:
      entry: ``active:{scope}:{scope_id}:{entry_id}`` (JSON document)
      state: ``active_state:{scope}:{scope_id}`` (HASH)
      index: ``active_index:{scope}:{scope_id}`` (ZSET ts -> entry_id)
      summarized: ``active_summarized:{scope}:{scope_id}`` (SET retained IDs)
    Atomic observe/trim/clear run as Lua EVAL (see lua.py) so concurrent
    clients linearize on one Redis point; no per-process locks.
    """

    def __init__(self, redis_client) -> None:
        self.redis = redis_client

    @staticmethod
    def _entry_key(scope: str, scope_id: str, entry_id: str) -> str:
        return f"active:{scope}:{scope_id}:{entry_id}"

    @staticmethod
    def _state_key(scope: str, scope_id: str) -> str:
        return f"active_state:{scope}:{scope_id}"

    @staticmethod
    def _index_key(scope: str, scope_id: str) -> str:
        return f"active_index:{scope}:{scope_id}"

    @staticmethod
    def _archive_key(scope: str, scope_id: str, day: str) -> str:
        return f"t1:archive:{scope}:{scope_id}:{day}"

    @staticmethod
    def _summarized_key(scope: str, scope_id: str) -> str:
        return f"active_summarized:{scope}:{scope_id}"

    async def observe_entry(self, entry: ActiveEntry) -> int:
        """Atomically persist entry + index + tokens + max(last_entry_ts)."""
        payload = json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, default=_json_default)
        ts = entry.created_at.timestamp()
        res = await self.redis.eval(
            T1_OBSERVE_V1, 3,
            self._entry_key(entry.scope, entry.scope_id, entry.entry_id),
            self._index_key(entry.scope, entry.scope_id),
            self._state_key(entry.scope, entry.scope_id),
            entry.entry_id, payload, str(float(ts)), str(int(entry.tokens)),
        )
        return int(res)

    async def trim_summarized(
        self, scope: str, scope_id: str, summarized_ids: list[str], *, keep_recent: int
    ) -> dict:
        """Atomically delete summarized IDs outside keep tail; mark retained."""
        if not summarized_ids:
            return {"deleted_ids": [], "subtracted": 0, "unsummarized_tokens": 0}
        res = await self.redis.eval(
            T1_TRIM_V1, 3,
            self._index_key(scope, scope_id),
            self._state_key(scope, scope_id),
            self._summarized_key(scope, scope_id),
            scope, scope_id, str(int(keep_recent)),
            str(len(summarized_ids)), *summarized_ids,
        )
        deleted = [(d.decode() if isinstance(d, bytes) else d) for d in res[2:]]
        return {"deleted_ids": deleted, "subtracted": int(res[0]), "unsummarized_tokens": int(res[1])}

    async def trim_summarized_guarded(self, scope: str, scope_id: str, *, expected_ids: list[str], expected_hashes: list[str], expected_plan: str, expected_nonce: str, expected_gen: str, expected_trim_ids: list[str], expected_ack: dict, snapshots: list[str], keep_recent: int) -> dict:
        """Single-EVAL guarded trim + caller trimmed (no fallback, fail closed)."""
        from twin.shared.memory.consolidation_journal import caller_pending_key
        res = await self.redis.eval(T1_TRIM_GUARDED_V1, 4, self._index_key(scope, scope_id), self._state_key(scope, scope_id), self._summarized_key(scope, scope_id), caller_pending_key(scope, scope_id), scope, scope_id, str(int(keep_recent)), str(len(expected_ids)), *expected_ids, json.dumps(expected_hashes, separators=(",", ":")), expected_plan, expected_nonce, expected_gen, json.dumps(expected_trim_ids, separators=(",", ":")), json.dumps(expected_ack, ensure_ascii=True, sort_keys=True, separators=(",", ":")), *snapshots)
        parts = [(v.decode() if isinstance(v, bytes) else v) for v in (res or [])]
        if len(parts) >= 2 and parts[0] == 1 and parts[1] == "already":
            return {"status": "already"}
        if len(parts) >= 4 and parts[0] == 1 and parts[1] == "trimmed":
            return {"status": "trimmed", "subtracted": int(parts[2]), "unsummarized_tokens": int(parts[3]), "deleted_ids": [str(v) for v in parts[4:]]}
        return {"status": "failed", "reason": str(parts[1]) if len(parts) >= 2 else "guard_rejected", "detail": str(parts[2]) if len(parts) >= 3 else ""}

    async def get_entry(self, scope: str, scope_id: str, entry_id: str) -> ActiveEntry | None:
        return await self._load_entry(scope, scope_id, entry_id)

    async def get_entries_by_ids(
        self, scope: str, scope_id: str, entry_ids: list[str]
    ) -> list[ActiveEntry]:
        """Order-preserving exact fetch (missing IDs skipped, window-independent)."""
        found: list[ActiveEntry] = []
        for entry_id in entry_ids or []:
            entry = await self._load_entry(scope, scope_id, entry_id)
            if entry is not None:
                found.append(entry)
        return found

    async def list_summarized_ids(self, scope: str, scope_id: str) -> set[str]:
        members = await self.redis.smembers(self._summarized_key(scope, scope_id))
        return {(m.decode() if isinstance(m, bytes) else m) for m in (members or [])}

    async def list_unsummarized_entries(
        self, scope: str, scope_id: str, *, limit: int = 200, max_pages: int | None = None
    ) -> list[ActiveEntry]:
        """Most-recent unsummarized, chronological, until 200 or index end.

        Consolidation-path only: per-ID summarized checks (no unbounded
        SMEMBERS), pages backwards until limit eligible or actual index
        end. Never false-empty (explicit scan_exhausted if capped).
        """
        if limit <= 0:
            return []
        index_key = self._index_key(scope, scope_id)
        summ_key = self._summarized_key(scope, scope_id)
        collected: list[ActiveEntry] = []
        page = 0
        while True:
            start = -limit * (page + 1)
            stop = -limit * page - 1 if page else -1
            ids = await self.redis.zrange(index_key, start, stop)
            if not ids:
                break
            fresh: list[ActiveEntry] = []
            for raw_id in ids:
                entry_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
                if await self.redis.sismember(summ_key, entry_id):
                    continue
                entry = await self._load_entry(scope, scope_id, entry_id)
                if entry is not None:
                    fresh.append(entry)
                if len(collected) + len(fresh) >= limit:
                    break
            collected = fresh + collected
            if len(collected) >= limit or len(ids) < limit:
                break
            page += 1
            if max_pages is not None and page >= max(1, int(max_pages)):
                raise RuntimeError("scan_exhausted: capped before index end")
        return collected[-limit:]

    async def save(self, entry: ActiveEntry) -> None:
        key = self._entry_key(entry.scope, entry.scope_id, entry.entry_id)
        payload = json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, default=_json_default)
        await self.redis.execute_command("JSON.SET", key, "$", payload)
        await self.redis.zadd(
            self._index_key(entry.scope, entry.scope_id),
            {entry.entry_id: entry.created_at.timestamp()},
        )
        logger.debug("T1: saved entry %s scope=%s/%s", entry.entry_id, entry.scope, entry.scope_id)

    async def list_entries(
        self, scope: str, scope_id: str, limit: int = 50
    ) -> list[ActiveEntry]:
        # Read the `limit` most-recent entries (ZSET ordered by ascending ts),
        # then keep chronological order. Reading the head would pin the window
        # to the oldest messages once a scope exceeds `limit`.
        ids = await self.redis.zrange(self._index_key(scope, scope_id), -limit, -1)
        entries: list[ActiveEntry] = []
        for raw_id in ids:
            entry_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            data = await self._load_entry(scope, scope_id, entry_id)
            if data is not None:
                entries.append(data)
        return entries

    async def _load_entry(
        self, scope: str, scope_id: str, entry_id: str
    ) -> ActiveEntry | None:
        raw = await self.redis.execute_command(
            "JSON.GET", self._entry_key(scope, scope_id, entry_id)
        )
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        return ActiveEntry.model_validate(json.loads(raw))

    async def delete_entries(
        self, scope: str, scope_id: str, entry_ids: list[str]
    ) -> None:
        if not entry_ids:
            return
        keys = [self._entry_key(scope, scope_id, eid) for eid in entry_ids]
        await self.redis.delete(*keys)
        await self.redis.zrem(self._index_key(scope, scope_id), *entry_ids)
        logger.debug("T1: deleted %d entries scope=%s/%s", len(entry_ids), scope, scope_id)

    async def archive_entries(
        self, scope: str, scope_id: str, entries: list[ActiveEntry], *, ttl_seconds: int
    ) -> None:
        """RPUSH serialized entries onto cold per-VN-day archive lists (W3).

        Grouped by the entry's own created_at VN day, not today's — a trim
        can carry messages from before midnight. Every RPUSH refreshes the
        day-key TTL. Pure Redis ops: the enabled/TTL policy lives in the
        caller (ActiveMemory.trim), keeping this layer Config-free.
        """
        if not entries:
            return
        by_day: dict[str, list[str]] = {}
        for entry in entries:
            day = vn_day_str(entry.created_at.timestamp())
            payload = json.dumps(
                entry.model_dump(mode="json"), ensure_ascii=False, default=_json_default
            )
            by_day.setdefault(day, []).append(payload)
        for day, payloads in by_day.items():
            key = self._archive_key(scope, scope_id, day)
            await self.redis.rpush(key, *payloads)
            await self.redis.expire(key, ttl_seconds)
        logger.debug(
            "T1: archived %d entries scope=%s/%s days=%d",
            len(entries), scope, scope_id, len(by_day),
        )

    async def clear_scope(self, scope: str, scope_id: str) -> None:
        # Single Lua: ZRANGE all (no cap) + UNLINK docs + SCAN orphans + DEL
        # index/state/markers. Concurrent observes linearize before/after.
        await self.redis.eval(
            T1_CLEAR_V1, 3,
            self._index_key(scope, scope_id),
            self._state_key(scope, scope_id),
            self._summarized_key(scope, scope_id),
            scope, scope_id,
        )

    async def list_active_scope_ids(self, scope: str) -> list[str]:
        pattern = self._state_key(scope, "*")
        if not hasattr(self.redis, "scan_iter"):
            return []
        active: list[str] = []
        prefix = self._state_key(scope, "")
        async for raw_key in self.redis.scan_iter(match=pattern):
            key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            scope_id = key[len(prefix):]
            state = await self.get_state(scope, scope_id)
            if int(state.get("unsummarized_tokens") or 0) > 0:
                active.append(scope_id)
        return active

    async def get_state(self, scope: str, scope_id: str) -> dict:
        raw = await self.redis.hgetall(self._state_key(scope, scope_id))
        if not raw:
            return {
                "unsummarized_tokens": 0,
                "last_entry_ts": None,
                "recent_catalogs": [],
            }
        decoded = {
            (k.decode() if isinstance(k, bytes) else k): (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in raw.items()
        }
        return {
            "unsummarized_tokens": int(decoded.get("unsummarized_tokens", 0) or 0),
            "last_entry_ts": float(decoded["last_entry_ts"])
            if decoded.get("last_entry_ts")
            else None,
            "recent_catalogs": json.loads(decoded.get("recent_catalogs") or "[]"),
        }

    async def increment_tokens(
        self, scope: str, scope_id: str, tokens: int, *, last_entry_ts: float | None = None
    ) -> int:
        """Atomically HINCRBY tokens + max(last_entry_ts); return new total."""
        res = await self.redis.eval(
            T1_INCR_TOKENS_V1, 1, self._state_key(scope, scope_id),
            str(int(tokens)),
            str(float(last_entry_ts)) if last_entry_ts is not None else "",
        )
        return int(res)

    async def update_state(
        self,
        scope: str,
        scope_id: str,
        *,
        unsummarized_tokens: int | None = None,
        recent_catalogs: list[str] | None = None,
        last_entry_ts: float | None = None,
    ) -> None:
        updates: dict[str, str] = {}
        if unsummarized_tokens is not None:
            updates["unsummarized_tokens"] = str(int(unsummarized_tokens))
        if recent_catalogs is not None:
            updates["recent_catalogs"] = json.dumps(recent_catalogs, ensure_ascii=False)
        if last_entry_ts is not None:
            updates["last_entry_ts"] = str(float(last_entry_ts))
        if not updates:
            return
        await self.redis.hset(self._state_key(scope, scope_id), mapping=updates)

    async def reset_state(
        self, scope: str, scope_id: str, *, keep_recent_catalogs: bool = True
    ) -> None:
        current = await self.get_state(scope, scope_id)
        await self.redis.delete(self._state_key(scope, scope_id))
        if keep_recent_catalogs and current.get("recent_catalogs"):
            await self.update_state(
                scope, scope_id, recent_catalogs=current["recent_catalogs"]
            )
