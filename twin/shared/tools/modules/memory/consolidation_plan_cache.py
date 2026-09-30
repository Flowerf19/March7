"""Canonical consolidation-plan cache: one durable plan per snapshot.

DURABLE-FIRST CONTRACT: a meaningful consolidation must witness its plan
(an atomic SET NX win or a validated cache read) BEFORE any T2/T3 write.
Cache I/O errors, undecodable/corrupt payloads, lost races without a
readable winner, and missing durable capability (a store without usable
Redis) all surface as PlanError so the tool fails closed with no
acknowledgement: T1 is preserved and the summary/profile writers are
never called. There is deliberately no own-plan fallback on transient
failure and no process lock — concurrent writers serialize on Redis.
Noise plans carry no writes and resolve without the cache.

Retry idempotency must not depend on the LLM reproducing the same topic
slugs: a partial T2 failure followed by a retry with drifted labels would
otherwise re-store the already-saved transcript under new topics. The first
validated plan for a scope/source snapshot wins and every retry of the same
snapshot reuses its topics/summaries and — when the profile is unchanged —
its required profile writes.

Snapshot identity binds the full ORDERED prompt/provenance fingerprint
(entry/message/author ids, role, content, author name, timestamp) plus
scope/scope_id as canonical JSON: the summarizer prompt renders entries
sequentially with per-entry VN timestamps, so any date/author/content/
order change mints a fresh plan.

OVERLAP OWNERSHIP: exact-snapshot keys cannot protect overlapping batches
(a partial e1/e2 attempt followed by grown e1/e2/e3 input). The receiver
journal (twin.shared.memory.consolidation_journal) therefore owns entries
per batch: disjoint batches proceed, overlapping different snapshots are
rejected BEFORE any full LLM call or T2/T3 write, and exact retries adopt
the original canonical plan. The canonical plan is persisted atomically
WITH the ownership claim, before any meaningful write.

RETENTION: pending owners/batch/plans carry NO expiry and are never
released on failure — T1 unsummarized entries never expire either, and
each late T2 partial persists up to 365 days from ITS write, so any
claim-time TTL would die first. Only after a validated ACK and a
successful caller trim are the matching owners released; the completed
plan then gets a fresh 365d TTL starting after the last T2 write.
Residual limitation: if journal AND plan keys are lost out-of-band
(eviction, failover, flush) while T2 partials survive, a retry mints
fresh topics and may duplicate — that joint loss is not detectable.

Profile safety: the cached plan records the profile hash it was computed
against. A retry that observes profile drift recomputes ONLY the profile
portion (fresh profile + same messages, strict validation) instead of
blessing the stale rewrite with the new hash. A conflict inside one
attempt still fails closed with no acknowledgement.

The cache holds derived summaries + source entry IDs only (never the raw
transcript beyond what the plan itself contains).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from twin.shared.memory.consolidation_journal import (
    PLAN_CACHE_PREFIX,
    PLAN_CACHE_TTL_SECONDS,
    ReceiverJournal,
    build_plan_cache_key,
    entry_ids_of,
)
from twin.shared.memory.profile.constants import SECTIONS
from twin.shared.tools.modules.memory.consolidation_schema import (
    ConsolidationPlan,
    ConsolidationPlanValidator,
    PlanError,
)

logger = logging.getLogger(__name__)

# PLAN_CACHE_PREFIX, PLAN_CACHE_TTL_SECONDS, build_plan_cache_key and
# entry_ids_of are re-exported from consolidation_journal (canonical home
# of snapshot identity + retention) for existing importers. The TTL is the
# COMPLETION ttl stamped by release(), never a pending ttl.
PLAN_CACHE_VERSION = 1

FetchFullPlan = Callable[[], Awaitable[tuple[ConsolidationPlan | None, PlanError | None]]]
FetchProfilePatch = Callable[
    [list[dict[str, Any]]],
    Awaitable[tuple[dict[str, list[str]], dict[str, list[str]], PlanError | None]],
]


def _topics_json(plan: ConsolidationPlan) -> list[dict[str, Any]]:
    return [
        {
            "topic": t.topic,
            "topic_display": t.topic_display,
            "summary": t.summary,
            "importance": t.importance,
        }
        for t in plan.topics
    ]


def serialize_canonical_plan(
    plan: ConsolidationPlan,
    *,
    scope: str,
    scope_id: str,
    entry_ids: list[str],
    profile_hash: str | None,
) -> str:
    return json.dumps({
        "v": PLAN_CACHE_VERSION,
        "scope": str(scope or "user"),
        "scope_id": str(scope_id or ""),
        "entry_ids": list(entry_ids),
        "topics": _topics_json(plan),
        "profile_updates": {k: list(v) for k, v in plan.profile_updates.items()},
        "profile_rewrites": {k: list(v) for k, v in plan.profile_rewrites.items()},
        "profile_hash": profile_hash,
    }, ensure_ascii=False, sort_keys=True)


def parse_canonical_plan(
    raw: str, *, scope: str, scope_id: str, entry_ids: list[str],
) -> tuple[ConsolidationPlan, str | None] | None:
    """Strict-parse a cached payload; None unless it matches this snapshot.

    The cached side is untrusted: entry ids must be actual non-empty
    strings (never coerced) and the profile hash must be None or str.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("v") != PLAN_CACHE_VERSION:
        return None
    if data.get("scope") != str(scope or "user") or data.get("scope_id") != str(scope_id or ""):
        return None
    cached_ids = data.get("entry_ids")
    if (
        not isinstance(cached_ids, list)
        or any(type(i) is not str or not i for i in cached_ids)
        or sorted(cached_ids) != sorted(entry_ids)
    ):
        return None
    wrapped = {
        "has_meaningful_content": True,
        "topics": data.get("topics", []),
        "profile_updates": data.get("profile_updates", {}),
        "profile_rewrites": data.get("profile_rewrites", {}),
    }
    plan, err = ConsolidationPlanValidator.validate(wrapped)
    if err is not None or plan is None or not plan.has_meaningful:
        return None
    stored_hash = data.get("profile_hash")
    if stored_hash is not None and type(stored_hash) is not str:
        return None
    return plan, stored_hash


def build_profile_recompute_prompt(profile_text: str, messages_text: str) -> str:
    """Profile-only recompute prompt: fresh profile + same messages."""
    sections = ", ".join(SECTIONS)
    return (
        "Bạn là Memory Profiler. Hồ sơ dưới đây đã THAY ĐỔI kể từ lần lập kế hoạch "
        "trước, nên chỉ tính toán lại phần hồ sơ (topics giữ nguyên, không đụng tới).\n\n"
        "=== HỒ SƠ HIỆN TẠI (MỚI NHẤT — mọi bullet còn đúng đều phải giữ) ===\n"
        f"{profile_text}\n\n"
        "=== TIN NHẮN GẦN ĐÂY ===\n"
        f"{messages_text}\n\n"
        "Nhiệm vụ:\n"
        "1. Trích xuất facts mới đáng lưu → profile_updates (bỏ qua nếu đã có trong hồ sơ mới nhất).\n"
        "2. Chỉ khi fact mới MÂU THUẪN với hồ sơ mới nhất: đưa section đó vào profile_rewrites "
        "với TOÀN BỘ bullet viết lại sạch — giữ mọi bullet còn đúng (kể cả bullet vừa được thêm), "
        "bỏ bullet lỗi thời, thêm bullet mới. Không mâu thuẫn → profile_rewrites để {}.\n"
        f"3. Section hợp lệ duy nhất: {sections}.\n\n"
        'Chỉ return JSON, không giải thích: {"profile_updates": {...}, "profile_rewrites": {...}}'
    )


@dataclass(frozen=True, slots=True)
class CacheLoad:
    """Strict plan-cache read outcome.

    ``hit`` carries a validated same-snapshot plan, ``miss`` means the key
    is absent (the caller may mint a plan via claim), and ``error`` means
    a Redis/encoding/corruption failure the caller must fail closed on.
    """

    status: str  # "hit" | "miss" | "error"
    plan: ConsolidationPlan | None = None
    profile_hash: str | None = None
    detail: str = ""


class CanonicalPlanCache:
    """Redis I/O for canonical plans (absent Redis = caching disabled).

    Plans are persisted by the receiver journal claim (atomic with entry
    ownership); this class only reads/refreshes them. Pending plans carry
    no expiry; release() stamps the completion TTL.
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    @classmethod
    def from_store(cls, timeline_summary_store: Any) -> "CanonicalPlanCache | None":
        redis = getattr(timeline_summary_store, "redis", None)
        if redis is None:
            return None
        if not callable(getattr(redis, "get", None)):
            return None
        if not callable(getattr(redis, "set", None)):
            return None
        if not callable(getattr(redis, "eval", None)):
            return None
        return cls(redis)

    async def load(
        self, key: str, *, scope: str, scope_id: str, entry_ids: list[str],
    ) -> CacheLoad:
        """Strict read: HIT only for a validated same-snapshot plan.

        MISS (key absent) lets the caller mint a plan via claim(); ERROR
        (Redis I/O failure, undecodable bytes, corrupt payload, snapshot
        mismatch) must fail the attempt closed — a missing/unreadable
        winner must never fall back to an unwitnessed own plan.
        """
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            logger.warning("plan cache GET failed key=%s: %s", key, exc)
            return CacheLoad("error", detail=f"redis GET failed: {exc}")
        if raw is None:
            return CacheLoad("miss")
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("plan cache undecodable bytes key=%s", key)
                return CacheLoad("error", detail="cached plan bytes are not UTF-8")
        parsed = parse_canonical_plan(
            str(raw), scope=scope, scope_id=scope_id, entry_ids=entry_ids,
        )
        if parsed is None:
            logger.warning("plan cache corrupt/snapshot-mismatch key=%s", key)
            return CacheLoad("error", detail="cached plan corrupt or not for this snapshot")
        plan, plan_hash = parsed
        return CacheLoad("hit", plan=plan, profile_hash=plan_hash)

    async def refresh(self, key: str, payload: str) -> None:
        # Pending refresh: no expiry (the batch is still unacknowledged;
        # release() stamps the completion TTL after trim).
        try:
            await self._redis.set(key, payload)
        except Exception as exc:
            logger.debug("plan cache refresh failed: %s", exc)


@dataclass(frozen=True, slots=True)
class PlanResolution:
    plan: ConsolidationPlan | None = None
    error: PlanError | None = None
    refresh: tuple[str, str] | None = None  # (key, payload) after success
    generation: str = ""


class CanonicalPlanResolver:
    """Resolve one attempt's canonical plan (journal + cache + LLM fetchers).

    Durable-first: no meaningful plan is returned without a witnessed
    journal claim (exact retries adopt the witnessed plan via a verified
    read). Ownership is probed BEFORE the full LLM call; the atomic claim
    rechecks ownership while persisting the plan, closing concurrent
    races. Journal/cache errors, corruption, overlapping batches, lost
    races without a readable winner, and missing durable capability all
    surface as PlanError so the tool fails closed with no acknowledgement
    (T1 preserved, no T2/T3 writes). Noise plans carry no writes and
    resolve without claiming — unless they overlap owned entries.
    """

    def __init__(
        self,
        cache: CanonicalPlanCache | None,
        journal: ReceiverJournal | None,
    ) -> None:
        self._cache = cache
        self._journal = journal

    @classmethod
    def from_store(cls, timeline_summary_store: Any) -> "CanonicalPlanResolver":
        return cls(
            CanonicalPlanCache.from_store(timeline_summary_store),
            ReceiverJournal.from_store(timeline_summary_store),
        )

    async def refresh(self, key: str, payload: str) -> None:
        if self._cache is not None:
            await self._cache.refresh(key, payload)

    async def resolve(
        self,
        *,
        scope: str,
        scope_id: str,
        entries: list[Any],
        profile_hash: str | None,
        fetch_full_plan: FetchFullPlan,
        fetch_profile_patch: FetchProfilePatch,
    ) -> PlanResolution:
        entry_ids = entry_ids_of(entries)
        if self._cache is None or self._journal is None:
            plan, err = await fetch_full_plan()
            if err is not None or plan is None:
                return PlanResolution(error=err or PlanError("invalid_schema", "empty plan"))
            if not plan.has_meaningful:
                return PlanResolution(plan=plan)
            return PlanResolution(error=PlanError(
                "plan_cache_unavailable",
                "meaningful consolidation needs a durable plan journal "
                "(store has no usable Redis); refusing unwitnessed T2/T3 writes",
            ))
        probe = await self._journal.probe(scope, scope_id, entries)
        if probe.status == "error":
            return PlanResolution(error=PlanError(
                "plan_cache_error", probe.detail or "receiver journal read failed",
            ))
        if probe.status == "conflict":
            return PlanResolution(error=PlanError("pending_overlap", probe.detail))
        if probe.status == "exact":
            # Same batch: adopt the witnessed plan; a missing/corrupt plan
            # fails closed here — never a fresh LLM mint.
            loaded = await self._cache.load(
                probe.plan_key, scope=scope, scope_id=scope_id, entry_ids=entry_ids,
            )
            if loaded.status != "hit" or loaded.plan is None:
                return PlanResolution(error=PlanError(
                    "plan_cache_error",
                    "pending batch plan missing/corrupt; refusing fresh mint",
                ))
            return await self._on_cache_plan(
                probe.plan_key, (loaded.plan, loaded.profile_hash), entry_ids,
                scope, scope_id, profile_hash, fetch_profile_patch,
                generation=probe.generation,
            )
        plan, err = await fetch_full_plan()
        if err is not None or plan is None:
            return PlanResolution(error=err or PlanError("invalid_schema", "empty plan"))
        if not plan.has_meaningful:
            return PlanResolution(plan=plan)
        key = build_plan_cache_key(scope, scope_id, entries)
        payload = serialize_canonical_plan(
            plan, scope=scope, scope_id=scope_id,
            entry_ids=entry_ids, profile_hash=profile_hash,
        )
        claimed = await self._journal.claim_new(scope, scope_id, entries, payload)
        if claimed.status == "claimed":
            return PlanResolution(plan=plan, generation=claimed.generation)
        if claimed.status == "adopted":
            winner = parse_canonical_plan(
                claimed.payload, scope=scope, scope_id=scope_id, entry_ids=entry_ids,
            )
            if winner is None:
                return PlanResolution(error=PlanError(
                    "plan_cache_corrupt",
                    "concurrent winner unreadable for this snapshot; "
                    "refusing divergent T2/T3 writes",
                ))
            logger.info("plan journal adopted concurrent winner key=%s", key)
            return await self._on_cache_plan(
                key, winner, entry_ids, scope, scope_id, profile_hash,
                fetch_profile_patch, generation=claimed.generation,
            )
        if claimed.status == "conflict":
            return PlanResolution(error=PlanError("pending_overlap", claimed.detail))
        return PlanResolution(error=PlanError(
            "plan_cache_error", claimed.detail or "durable journal claim failed",
        ))

    async def _on_cache_plan(
        self,
        key: str,
        cached: tuple[ConsolidationPlan, str | None],
        entry_ids: list[str],
        scope: str,
        scope_id: str,
        profile_hash: str | None,
        fetch_profile_patch: FetchProfilePatch,
        generation: str = "",
    ) -> PlanResolution:
        plan, plan_hash = cached
        if plan_hash == profile_hash or not (plan.profile_updates or plan.profile_rewrites):
            return PlanResolution(plan=plan, generation=generation)
        # Profile drifted since the plan was cached: recompute ONLY the
        # profile portion against the fresh profile; topics stay pinned.
        updates, rewrites, err = await fetch_profile_patch(_topics_json(plan))
        if err is not None:
            return PlanResolution(error=err)
        fresh = ConsolidationPlan(True, plan.topics, updates, rewrites)
        payload = serialize_canonical_plan(
            fresh, scope=scope, scope_id=scope_id,
            entry_ids=entry_ids, profile_hash=profile_hash,
        )
        return PlanResolution(plan=fresh, refresh=(key, payload), generation=generation)
