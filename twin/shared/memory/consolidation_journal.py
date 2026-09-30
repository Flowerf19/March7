"""Durable consolidation journals: receiver ownership + caller pending batch.

Exact-snapshot plan caching cannot protect OVERLAPPING batches: a partial
e1/e2 attempt followed by a grown e1/e2/e3 snapshot mints a fresh plan key,
re-runs the LLM, and may drift topics (A->A2) so already-stored facts are
duplicated across topics with overlapping provenance. A fixed claim-time TTL
also dies before late T2 partials (each T2 write carries its own 7-365d TTL).

Two persistent journals close this (raw transcript stays in T1):

- Receiver journal (shared timeline Redis): per-entry owner ``consol:own:*``
  plus a bounded batch record ``consol:batch:*`` (ordered IDs, per-entry
  SHA256 provenance hashes, canonical plan key). Lua-atomic all-or-none
  claim: disjoint batches proceed, overlapping different snapshots are
  rejected BEFORE any T2/T3 write or full LLM call, and exact retries adopt
  the original canonical plan. The canonical plan is persisted atomically
  WITH the ownership claim, before any meaningful write.
- Caller journal (own T1 Redis): one pending record per scope
  ``consol:caller:*`` claiming the FIRST ordered batch before dispatch, so
  auto retries exact-fetch the pinned IDs instead of re-reading the moving
  last-200 window (which also cannot see entries pushed out by >200 new
  arrivals, and must exclude the retained summarized tail).

Retention: pending owners/batch/plans carry NO expiry and are NEVER
released on a failed attempt (a zero-store report cannot prove no
concurrent worker committed). Only after a validated ACK *and* a successful
caller trim are the matching owners released; the completed plan then gets
a fresh 365d TTL starting AFTER the last T2 write. Manual T1 clear/reset
discards only the caller's own pending record (its entries are gone);
receiver witnesses stay because T2 partials may persist.

Recovery: a scope stuck in ``pending_unrecoverable`` (T1 entries deleted or
altered out-of-band) needs explicit restore from ``t1:archive:*`` followed
by a fresh cycle, or a T2 audit plus manual deletion of the
``consol:own``/``consol:batch``/``consol:caller`` keys. There is no admin
release command by design. Out-of-band loss of BOTH journal and plan keys
(eviction/failover/flush) cannot be universally detected; the residual risk
is documented, not claimed away.
"""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- identity
# Canonical snapshot identity lives here (shared layer) so the coordinator
# never depends on the tool layer. consolidation_plan_cache re-exports the
# names its existing importers/tests use; no second implementation exists.

PLAN_CACHE_PREFIX = "consol:plan"

# Completion TTL, stamped once by release() AFTER the last T2 write of the
# batch (never a sliding pending TTL: pending witnesses must not expire).
PLAN_CACHE_TTL_SECONDS = 365 * 86400

RECEIVER_OWNER_PREFIX = "consol:own"
RECEIVER_BATCH_PREFIX = "consol:batch"
CALLER_PENDING_PREFIX = "consol:caller"

CALLER_RECORD_VERSION = 3
RECEIVER_RECORD_VERSION = 2

# Auto batches are bounded (also bounds the journal record + Lua key count).
MAX_AUTO_BATCH = 200


def _norm_str(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _norm_scope(scope: Any) -> str:
    return str(scope or "user").strip() or "user"


def _norm_sid(scope_id: Any) -> str:
    return str(scope_id or "").strip()


def entry_epoch(entry: Any) -> float | None:
    """Epoch seconds for shipped dicts (ISO timestamp) or ActiveEntry."""
    if isinstance(entry, dict):
        raw = entry.get("timestamp")
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(str(raw))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            return dt.timestamp()
        except (OverflowError, OSError, ValueError):
            return None
    created = getattr(entry, "created_at", None)
    try:
        return created.timestamp() if created is not None else None
    except (AttributeError, TypeError, ValueError, OverflowError, OSError):
        return None


def entry_fingerprint(entry: Any) -> dict[str, Any]:
    """Prompt/provenance-relevant fields, normalized across representations.

    Mirrors ConsolidationPromptBuilder.format_messages (role/content
    fallbacks) plus provenance (entry/message/author ids) and the VN-time
    source (timestamp/created_at via entry_epoch), so an ActiveEntry and
    its shipped dict fingerprint identically. Order is significant: callers
    keep entry order because the prompt renders entries sequentially.
    """
    if isinstance(entry, dict):
        role = entry.get("role") or "user"
        content = entry.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        author_id = entry.get("author_id")
        author_name = entry.get("author_name")
        entry_id = entry.get("entry_id")
        message_id = entry.get("message_id")
    else:
        role = getattr(entry, "role", None) or "user"
        content = getattr(entry, "content", None)
        if content is None:
            content = str(entry)
        elif not isinstance(content, str):
            content = str(content)
        author_id = getattr(entry, "author_id", None)
        author_name = getattr(entry, "author_name", None)
        entry_id = getattr(entry, "entry_id", None)
        message_id = getattr(entry, "message_id", None)
    return {
        "entry_id": _norm_str(entry_id),
        "message_id": _norm_str(message_id),
        "role": _norm_str(role),
        "content": content,
        "author_id": _norm_str(author_id),
        "author_name": _norm_str(author_name),
        "ts": entry_epoch(entry),
    }


def fingerprint_hash(entry: Any) -> str:
    """SHA256 of canonical full fingerprint JSON (persisted, privacy-safe).

    Full dict lives only in memory for identity computation; journals store
    only this 64-lower-hex digest plus ordered entry IDs. Covers the full
    ordered identity (role/content/author/date/message/entry IDs).
    """
    canonical = json.dumps(
        entry_fingerprint(entry),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def entry_hashes_of(entries: list[Any]) -> list[str]:
    return [fingerprint_hash(e) for e in (entries or [])]


def _is_hex64(value: Any) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    for ch in value:
        if ch not in "0123456789abcdef":
            return False
    return True


def _valid_entry_ids(ids: Any) -> bool:
    if not isinstance(ids, list) or not ids or len(ids) > MAX_AUTO_BATCH:
        return False
    seen: set[str] = set()
    for i in ids:
        if type(i) is not str or not i.strip():
            return False
        if i in seen:
            return False
        seen.add(i)
    return True


def _valid_fp_hashes(fps: Any, count: int) -> bool:
    if not isinstance(fps, list) or len(fps) != count:
        return False
    return all(_is_hex64(h) for h in fps)


def _is_hex32(value: Any) -> bool:
    if type(value) is not str or len(value) != 32:
        return False
    for ch in value:
        if ch not in "0123456789abcdef":
            return False
    return True


def _new_generation() -> str:
    return secrets.token_hex(16)


def validate_ack_generation(ack: dict) -> tuple[bool, str]:
    """Pure pre-journal gen check: meaningful True needs 32-lower-hex, noise False needs empty/absent."""
    if not isinstance(ack, dict):
        return False, "malformed_result"
    meaningful = ack.get("has_meaningful_content")
    gen = ack.get("receiver_generation", "")
    if gen is None:
        gen = ""
    if meaningful is True:
        return (True, "") if _is_hex32(gen) else (False, "bad_generation")
    if meaningful is False:
        return (True, "") if gen == "" else (False, "noise_with_generation")
    return False, "missing_meaningful_flag"


def _split_owner(value: str) -> tuple[str, str] | None:
    if type(value) is not str or ":" not in value:
        return None
    batch, _, gen = value.partition(":")
    if not batch or not gen:
        return None
    return batch, gen


def entry_ids_of(entries: list[Any]) -> list[str]:
    ids: list[str] = []
    for entry in entries or []:
        eid = entry_fingerprint(entry)["entry_id"]
        if eid:
            ids.append(eid)
    return ids


def build_plan_cache_key(scope: str, scope_id: str, entries: list[Any]) -> str:
    """Snapshot-identity key: scope + ordered fingerprint digest."""
    scope_norm = _norm_scope(scope)
    sid_norm = _norm_sid(scope_id)
    canonical = json.dumps(
        {
            "scope": scope_norm,
            "scope_id": sid_norm,
            "entries": [entry_fingerprint(e) for e in (entries or [])],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"{PLAN_CACHE_PREFIX}:{scope_norm}:{sid_norm}:{digest}"


def batch_id_of_plan_key(plan_key: str) -> str:
    return str(plan_key or "").rsplit(":", 1)[-1]


def receiver_owner_key(scope: str, scope_id: str, entry_id: str) -> str:
    return f"{RECEIVER_OWNER_PREFIX}:{_norm_scope(scope)}:{_norm_sid(scope_id)}:{entry_id}"


def receiver_batch_key(scope: str, scope_id: str, batch_id: str) -> str:
    return f"{RECEIVER_BATCH_PREFIX}:{_norm_scope(scope)}:{_norm_sid(scope_id)}:{batch_id}"


def caller_pending_key(scope: str, scope_id: str) -> str:
    return f"{CALLER_PENDING_PREFIX}:{_norm_scope(scope)}:{_norm_sid(scope_id)}"


# --------------------------------------------------------------------- Lua

# Claim owners + batch record + canonical plan atomically (no expiry while
# pending). Same-batch racers converge on winner's generation + plan; any
# entry owned by a different batch aborts with NOTHING written.
# KEYS[1..N]=owner keys, KEYS[N+1]=batch key, KEYS[N+2]=plan key
# ARGV[1]=N ARGV[2]=batch_id ARGV[3]=batch_json ARGV[4]=plan_payload
# ARGV[5]=generation (random per-claim owner nonce)
_RECEIVER_CLAIM_V1 = """-- RECEIVER_CLAIM_V1
local n = tonumber(ARGV[1])
local batch_id = ARGV[2]
local own_gen = ARGV[5]
local expected = batch_id .. ':' .. own_gen
local winner_gen = nil
for i=1,n do
  local cur = redis.call('GET', KEYS[i])
  if cur ~= false then
    local sep = string.find(cur, ':', 1, true)
    if sep == nil then
      return {0, 'conflict', KEYS[i], cur}
    end
    local cur_batch = string.sub(cur, 1, sep-1)
    local cur_gen = string.sub(cur, sep+1)
    if cur_batch ~= batch_id then
      return {0, 'conflict', KEYS[i], cur}
    end
    if winner_gen == nil then
      winner_gen = cur_gen
    elseif winner_gen ~= cur_gen then
      return {0, 'batch_mismatch', ''}
    end
  end
end
local bkey = KEYS[n+1]
local pkey = KEYS[n+2]
local eb = redis.call('GET', bkey)
if eb ~= false and eb ~= ARGV[3] then
  return {0, 'batch_mismatch', ''}
end
local ep = redis.call('GET', pkey)
for i=1,n do
  if redis.call('GET', KEYS[i]) == false then
    redis.call('SET', KEYS[i], expected)
  end
end
if eb == false then redis.call('SET', bkey, ARGV[3]) end
if ep == false then
  redis.call('SET', pkey, ARGV[4])
  redis.call('PERSIST', bkey)
  redis.call('PERSIST', pkey)
  return {1, 'claimed', ARGV[4], own_gen}
end
redis.call('PERSIST', bkey)
redis.call('PERSIST', pkey)
local ret_gen = winner_gen
if ret_gen == nil then ret_gen = own_gen end
return {1, 'adopted', ep, ret_gen}
"""

# Release ONLY owners matching this generation; prevalidate ALL first:
# any foreign aborts with NO deletes and NO TTL (fail closed, no partial).
# Idempotent retry (all missing) succeeds with released=0.
# KEYS[1..N]=owner keys, KEYS[N+1]=batch key, KEYS[N+2]=plan key
# ARGV[1]=N ARGV[2]=expected_owner(batch_id:generation) ARGV[3]=ttl
_RECEIVER_RELEASE_V1 = """-- RECEIVER_RELEASE_V1
local n = tonumber(ARGV[1])
local expected = ARGV[2]
local ttl = tonumber(ARGV[3]) or 0
for i=1,n do
  local cur = redis.call('GET', KEYS[i])
  if cur ~= false and cur ~= expected then
    return {0, 'foreign', KEYS[i]}
  end
end
local released = 0
for i=1,n do
  local cur = redis.call('GET', KEYS[i])
  if cur ~= false and cur == expected then
    redis.call('DEL', KEYS[i])
    released = released + 1
  end
end
redis.call('DEL', KEYS[n+1])
if ttl > 0 then redis.call('EXPIRE', KEYS[n+2], ttl) end
return {released}
"""

# Claim one pending record per caller scope (no expiry); concurrent autos
# converge on the existing record instead of minting rival batches.
# KEYS[1]=caller key ARGV[1]=pending_json
_CALLER_CLAIM_V1 = """-- CALLER_CLAIM_V1
local cur = redis.call('GET', KEYS[1])
if cur == false then
  redis.call('SET', KEYS[1], ARGV[1])
  return {1, 'claimed', ARGV[1]}
end
return {0, 'exists', cur}
"""

# pending -> acknowledged for same batch identity (idempotent already).
# KEYS[1]=caller key ARGV[1]=expected_ids_json ARGV[2]=acked_json
# ARGV[3]=expected_fp_json ARGV[4]=expected_plan_key ARGV[5]=expected_nonce
_CALLER_ACK_V1 = """-- CALLER_ACK_V1
local cur = redis.call('GET', KEYS[1])
if cur == false then return {0, 'missing', ''} end
local ok, rec = pcall(cjson.decode, cur)
if not ok or type(rec) ~= 'table' then return {0, 'corrupt', ''} end
local eok, exp = pcall(cjson.decode, ARGV[1])
if not eok or type(exp) ~= 'table' then return {0, 'mismatch', ''} end
local ids = rec['entry_ids']
if type(ids) ~= 'table' or #ids ~= #exp then return {0, 'mismatch', ''} end
for i=1,#exp do if ids[i] ~= exp[i] then return {0, 'mismatch', ''} end end
local fok, efp = pcall(cjson.decode, ARGV[3])
if not fok or type(efp) ~= 'table' then return {0, 'mismatch', ''} end
local fps = rec['fingerprints']
if type(fps) ~= 'table' or #fps ~= #efp then return {0, 'mismatch', ''} end
for i=1,#efp do if fps[i] ~= efp[i] then return {0, 'mismatch', ''} end end
if rec['plan_key'] ~= ARGV[4] then return {0, 'mismatch', ''} end
if rec['caller_nonce'] ~= ARGV[5] then return {0, 'mismatch', ''} end
if rec['stage'] == 'acknowledged' or rec['stage'] == 'trimmed' then return {1, 'already', cur} end
if rec['stage'] ~= 'pending' then return {0, 'mismatch', ''} end
redis.call('SET', KEYS[1], ARGV[2])
return {1, 'acked', ARGV[2]}
"""

# Delete only an acknowledged/trimmed record with matching IDs+nonce.
# KEYS[1]=caller key ARGV[1]=expected_ids_json ARGV[2]=expected_nonce
_CALLER_CLEAR_V1 = """-- CALLER_CLEAR_V1
local cur = redis.call('GET', KEYS[1])
if cur == false then return {1, 'cleared', ''} end
local ok, rec = pcall(cjson.decode, cur)
if not ok or type(rec) ~= 'table' then return {0, 'corrupt', ''} end
if rec['stage'] ~= 'acknowledged' and rec['stage'] ~= 'trimmed' then return {0, 'not_acked', ''} end
local eok, exp = pcall(cjson.decode, ARGV[1])
if not eok or type(exp) ~= 'table' then return {0, 'mismatch', ''} end
local ids = rec['entry_ids']
if type(ids) ~= 'table' then return {0, 'mismatch', ''} end
if #ids ~= #exp then return {0, 'mismatch', ''} end
for i=1,#exp do if ids[i] ~= exp[i] then return {0, 'mismatch', ''} end end
if rec['caller_nonce'] ~= ARGV[2] then return {0, 'mismatch', ''} end
redis.call('DEL', KEYS[1])
return {1, 'cleared', ''}
"""

# acknowledged -> trimmed after T1 trim, before receiver release.
# Idempotent already when already trimmed with same IDs+nonce.
# KEYS[1]=caller key ARGV[1]=expected_ids_json ARGV[2]=expected_nonce
_CALLER_MARK_TRIMMED_V1 = """-- CALLER_MARK_TRIMMED_V1
local cur = redis.call('GET', KEYS[1])
if cur == false then return {0, 'missing', ''} end
local ok, rec = pcall(cjson.decode, cur)
if not ok or type(rec) ~= 'table' then return {0, 'corrupt', ''} end
local eok, exp = pcall(cjson.decode, ARGV[1])
if not eok or type(exp) ~= 'table' then return {0, 'mismatch', ''} end
local ids = rec['entry_ids']
if type(ids) ~= 'table' or #ids ~= #exp then return {0, 'mismatch', ''} end
for i=1,#exp do if ids[i] ~= exp[i] then return {0, 'mismatch', ''} end end
if rec['caller_nonce'] ~= ARGV[2] then return {0, 'mismatch', ''} end
if rec['stage'] == 'trimmed' then return {1, 'already', cur} end
if rec['stage'] ~= 'acknowledged' then return {0, 'mismatch', ''} end
rec['stage'] = 'trimmed'
local nxt = cjson.encode(rec)
redis.call('SET', KEYS[1], nxt)
return {1, 'trimmed', nxt}
"""


# ------------------------------------------------------------------ helpers

def _decode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return value if isinstance(value, str) else str(value)


def _decode_list(value: Any) -> list | None:
    if not isinstance(value, (list, tuple)):
        return None
    out: list[Any] = []
    for item in value:
        if isinstance(item, bytes):
            try:
                out.append(item.decode("utf-8"))
            except UnicodeDecodeError:
                return None
        else:
            out.append(item)
    return out


def _batch_json(scope: str, scope_id: str, entry_ids: list[str],
                fingerprints: list[str], plan_key: str) -> str:
    return json.dumps({
        "v": RECEIVER_RECORD_VERSION,
        "scope": _norm_scope(scope),
        "scope_id": _norm_sid(scope_id),
        "entry_ids": list(entry_ids),
        "fingerprints": fingerprints,
        "plan_key": plan_key,
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _parse_batch(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("v") != RECEIVER_RECORD_VERSION:
        return None
    ids = data.get("entry_ids")
    if not _valid_entry_ids(ids):
        return None
    assert isinstance(ids, list)
    if not _valid_fp_hashes(data.get("fingerprints"), len(ids)):
        return None
    pk = data.get("plan_key")
    if type(pk) is not str or not pk.startswith(PLAN_CACHE_PREFIX + ":"):
        return None
    if type(data.get("scope")) is not str or type(data.get("scope_id")) is not str:
        return None
    return data


def build_caller_pending(scope: str, scope_id: str, entries: list[Any]) -> dict[str, Any]:
    return {
        "v": CALLER_RECORD_VERSION,
        "stage": "pending",
        "scope": _norm_scope(scope),
        "scope_id": _norm_sid(scope_id),
        "entry_ids": entry_ids_of(entries),
        "fingerprints": entry_hashes_of(entries),
        "plan_key": build_plan_cache_key(scope, scope_id, entries),
        "caller_nonce": _new_generation(),
    }


def _pending_json(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def parse_caller_record(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("v") != CALLER_RECORD_VERSION:
        return None
    if data.get("stage") not in ("pending", "acknowledged", "trimmed"):
        return None
    ids = data.get("entry_ids")
    if not _valid_entry_ids(ids):
        return None
    assert isinstance(ids, list)
    if not _valid_fp_hashes(data.get("fingerprints"), len(ids)):
        return None
    pk = data.get("plan_key")
    if type(pk) is not str or not pk.startswith(PLAN_CACHE_PREFIX + ":"):
        return None
    if type(data.get("scope")) is not str or type(data.get("scope_id")) is not str:
        return None
    if not _is_hex32(data.get("caller_nonce")):
        return None
    if data["stage"] in ("acknowledged", "trimmed"):
        tids = data.get("trim_ids")
        if (
            not isinstance(tids, list) or not tids
            or any(type(i) is not str or not i.strip() for i in tids)
            or len(tids) != len(set(tids))
            or set(tids) != set(ids)
        ):
            return None
        ack = data.get("ack")
        if not isinstance(ack, dict):
            return None
        try:
            from twin.shared.memory.consolidation_ack import ack_trim_ids as _ack_check
        except Exception:
            return None
        valid, _ = _ack_check(ack, set(ids))
        if valid is None or set(valid) != set(tids):
            return None
        rg = data.get("receiver_generation", "")
        if type(rg) is not str:
            return None
        if ack.get("has_meaningful_content") is True:
            if not _is_hex32(rg):
                return None
        elif ack.get("has_meaningful_content") is False:
            if rg != "":
                return None
        else:
            return None
    return data


def build_caller_acked(pending: dict[str, Any], trim_ids: list[str], ack: dict[str, Any]) -> dict[str, Any]:
    rec = dict(pending)
    rec["stage"] = "acknowledged"
    rec["trim_ids"] = list(trim_ids)
    rec["ack"] = dict(ack)
    rec["receiver_generation"] = str(ack.get("receiver_generation") or "")
    return rec


def incoming_matches_pending(entries: list[Any], record: dict[str, Any]) -> bool:
    try:
        if entry_ids_of(entries) != list(record.get("entry_ids") or []):
            return False
        if entry_hashes_of(entries) != list(record.get("fingerprints") or []):
            return False
        scope = str(record.get("scope") or "user")
        sid = str(record.get("scope_id") or "")
        return build_plan_cache_key(scope, sid, entries) == record.get("plan_key")
    except Exception:
        return False


def _has_journal_caps(redis: Any, *, need_delete: bool = False) -> bool:
    if redis is None:
        return False
    if not callable(getattr(redis, "get", None)) or not callable(getattr(redis, "eval", None)):
        return False
    if need_delete and not callable(getattr(redis, "delete", None)):
        return False
    return True


# ------------------------------------------------------- receiver journal

@dataclass(frozen=True, slots=True)
class ReceiverProbe:
    status: str  # "fresh" | "exact" | "conflict" | "error"
    detail: str = ""
    plan_key: str = ""
    batch_id: str = ""
    generation: str = ""


@dataclass(frozen=True, slots=True)
class ReceiverClaim:
    status: str  # "claimed" | "adopted" | "conflict" | "error"
    detail: str = ""
    payload: str = ""
    generation: str = ""


@dataclass(frozen=True, slots=True)
class ReceiverRelease:
    ok: bool
    detail: str = ""
    released: int = 0


class ReceiverJournal:
    """Per-entry ownership + batch witness in shared timeline Redis."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    @classmethod
    def from_redis(cls, redis_client: Any) -> "ReceiverJournal | None":
        return cls(redis_client) if _has_journal_caps(redis_client) else None

    @classmethod
    def from_store(cls, timeline_summary_store: Any) -> "ReceiverJournal | None":
        return cls.from_redis(getattr(timeline_summary_store, "redis", None))

    async def probe(self, scope: str, scope_id: str, entries: list[Any]) -> ReceiverProbe:
        """Pre-LLM ownership check: fresh / exact / conflict / error.

        Exact requires the SAME ordered IDs, per-entry SHA256 hashes
        and canonical plan key as the owning batch; anything else that
        touches an owned entry is a conflict. Missing/corrupt batch data
        fails closed. The atomic claim rechecks afterwards to close races.
        """
        ids = entry_ids_of(entries)
        if not ids:
            return ReceiverProbe("fresh")
        try:
            owners: dict[str, str | None] = {}
            for eid in ids:
                raw = await self._redis.get(receiver_owner_key(scope, scope_id, eid))
                if raw is None:
                    owners[eid] = None
                    continue
                if isinstance(raw, bytes):
                    try:
                        owners[eid] = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        return ReceiverProbe("error", "owner bytes undecodable")
                else:
                    owners[eid] = raw if isinstance(raw, str) else str(raw)
        except Exception as exc:
            logger.warning("receiver journal probe GET failed: %s", exc)
            return ReceiverProbe("error", f"journal GET failed: {exc}")
        distinct = {o for o in owners.values() if o}
        if not distinct:
            return ReceiverProbe("fresh")
        batches: dict[str, str] = {}
        for full in distinct:
            split = _split_owner(full)
            if split is None:
                return ReceiverProbe("error", "owner record corrupt")
            b, g = split
            if b in batches and batches[b] != g:
                return ReceiverProbe("error", "owner generations diverged")
            batches[b] = g
        mine = build_plan_cache_key(scope, scope_id, entries)
        my_batch = batch_id_of_plan_key(mine)
        try:
            raws = {}
            for batch in batches:
                raw_b = await self._redis.get(receiver_batch_key(scope, scope_id, batch))
                if isinstance(raw_b, bytes):
                    try:
                        raw_b = raw_b.decode("utf-8")
                    except UnicodeDecodeError:
                        return ReceiverProbe("error", "batch bytes undecodable")
                elif raw_b is not None and not isinstance(raw_b, str):
                    raw_b = str(raw_b)
                raws[batch] = raw_b
        except Exception as exc:
            logger.warning("receiver journal batch GET failed: %s", exc)
            return ReceiverProbe("error", f"journal GET failed: {exc}")
        mine_fp = entry_hashes_of(entries)
        for batch, raw in raws.items():
            if raw is None:
                return ReceiverProbe("error", f"owner {batch} has no batch record")
            if _parse_batch(raw) is None:
                return ReceiverProbe("error", "batch record corrupt")
        if len(distinct) == 1 and len(batches) == 1:
            batch = next(iter(batches))
            gen = batches[batch]
            rec = _parse_batch(raws[batch])  # validated above
            assert rec is not None
            if (
                batch == my_batch
                and rec["scope"] == _norm_scope(scope)
                and rec["scope_id"] == _norm_sid(scope_id)
                and rec["entry_ids"] == ids
                and rec["fingerprints"] == mine_fp
                and rec["plan_key"] == mine
            ):
                return ReceiverProbe("exact", plan_key=mine, batch_id=batch,
                                     generation=gen)
        held = sorted(distinct)
        return ReceiverProbe(
            "conflict",
            f"{len(held)} other pending batch(es) own {sum(1 for o in owners.values() if o)} entr(y/ies)",
        )

    async def claim_new(self, scope: str, scope_id: str, entries: list[Any],
                        plan_payload: str) -> ReceiverClaim:
        """Atomically persist plan + claim ownership (no expiry)."""
        ids = entry_ids_of(entries)
        plan_key = build_plan_cache_key(scope, scope_id, entries)
        batch_id = batch_id_of_plan_key(plan_key)
        batch = _batch_json(
            scope, scope_id, ids, entry_hashes_of(entries), plan_key,
        )
        gen = _new_generation()
        keys = [receiver_owner_key(scope, scope_id, eid) for eid in ids]
        keys += [receiver_batch_key(scope, scope_id, batch_id), plan_key]
        try:
            res = _decode_list(await self._redis.eval(
                _RECEIVER_CLAIM_V1, len(keys), *keys,
                str(len(ids)), batch_id, batch, plan_payload, gen,
            ))
        except Exception as exc:
            logger.warning("receiver journal claim EVAL failed: %s", exc)
            return ReceiverClaim("error", f"journal EVAL failed: {exc}")
        if not res or type(res[0]) is not int or res[0] not in (0, 1):
            return ReceiverClaim("error", "journal claim returned malformed reply")
        if res[0] == 1 and res[1] in ("claimed", "adopted") and len(res) == 4:
            if type(res[2]) is not str or not _is_hex32(res[3]):
                return ReceiverClaim("error", "journal claim winner not decodable")
            return ReceiverClaim(res[1], payload=res[2], generation=str(res[3]))
        if res[0] == 0 and res[1] == "conflict":
            return ReceiverClaim("conflict", "entries claimed by another pending batch")
        if res[0] == 0 and res[1] in ("batch_mismatch",):
            return ReceiverClaim("error", "batch record mismatch for this snapshot")
        return ReceiverClaim("error", "journal claim rejected")

    async def release(self, scope: str, scope_id: str, entry_ids: list[str],
                      batch_id: str, plan_key: str,
                      generation: str = "") -> ReceiverRelease:
        """Release matching generation only; foreign fails with no writes."""
        expected = f"{batch_id}:{generation}" if generation else ""
        keys = [receiver_owner_key(scope, scope_id, eid) for eid in (entry_ids or [])]
        keys += [receiver_batch_key(scope, scope_id, batch_id), plan_key]
        try:
            res = _decode_list(await self._redis.eval(
                _RECEIVER_RELEASE_V1, len(keys), *keys,
                str(len(entry_ids or [])), expected, str(PLAN_CACHE_TTL_SECONDS),
            ))
        except Exception as exc:
            logger.warning("receiver journal release EVAL failed: %s", exc)
            return ReceiverRelease(False, f"journal EVAL failed: {exc}")
        if not res or type(res[0]) is not int:
            return ReceiverRelease(False, "journal release returned malformed reply")
        if res[0] == 0 and len(res) >= 2 and res[1] == "foreign":
            return ReceiverRelease(False, "foreign generation owns entries")
        if len(res) != 1:
            return ReceiverRelease(False, "journal release returned malformed reply")
        return ReceiverRelease(True, released=res[0])

    async def check_owners(self, scope: str, scope_id: str,
                           entry_ids: list[str], batch_id: str,
                           generation: str) -> tuple[bool, str]:
        """True iff owners match expected generation (overlap guard, not per-batch durability proof).

        Production meaningful batches always claim owners before T2/T3 writes, and a durable caller ACK
        represents completed required writes; owner keys guard overlapping batches. All-missing means
        out-of-band metadata loss (eviction/failover), not a mock allowance: it lets an already-ACKed
        batch finish without reminting (avoiding duplicate re-store), while partial/stale/foreign still
        reject and a new generation stays protected. Joint plan+ownership loss may duplicate; accepted.
        """
        expected = f"{batch_id}:{generation}" if generation else ""
        try:
            seen: list[str | None] = []
            for eid in entry_ids or []:
                raw = await self._redis.get(receiver_owner_key(scope, scope_id, eid))
                if raw is None:
                    seen.append(None)
                elif isinstance(raw, bytes):
                    try:
                        seen.append(raw.decode("utf-8"))
                    except UnicodeDecodeError:
                        return False, "owner bytes undecodable"
                else:
                    seen.append(raw if isinstance(raw, str) else str(raw))
            if generation:
                if all(v is None for v in seen):
                    return True, ""
                for cur in seen:
                    if cur != expected:
                        return False, "generation mismatch"
            else:
                for cur in seen:
                    if cur is not None:
                        return False, "foreign owner present"
        except Exception as exc:
            return False, f"journal GET failed: {exc}"
        return True, ""


# --------------------------------------------------------- caller journal

@dataclass(frozen=True, slots=True)
class CallerLoad:
    status: str  # "missing" | "pending" | "acknowledged" | "trimmed" | "error"
    record: dict[str, Any] | None = None
    detail: str = ""


class CallerJournal:
    """One pending/acknowledged batch per scope in the caller's own T1 Redis."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    @classmethod
    def from_redis(cls, redis_client: Any) -> "CallerJournal | None":
        return cls(redis_client) if _has_journal_caps(redis_client, need_delete=True) else None

    async def load(self, scope: str, scope_id: str) -> CallerLoad:
        try:
            raw_v = await self._redis.get(caller_pending_key(scope, scope_id))
        except Exception as exc:
            logger.warning("caller journal GET failed: %s", exc)
            return CallerLoad("error", detail=f"journal GET failed: {exc}")
        if raw_v is None:
            return CallerLoad("missing")
        if isinstance(raw_v, bytes):
            try:
                raw = raw_v.decode("utf-8")
            except UnicodeDecodeError:
                return CallerLoad("error", detail="pending bytes undecodable")
        else:
            raw = raw_v if isinstance(raw_v, str) else str(raw_v)
        rec = parse_caller_record(raw)
        if rec is None:
            return CallerLoad("error", detail="pending record corrupt")
        if rec["scope"] != _norm_scope(scope) or rec["scope_id"] != _norm_sid(scope_id):
            return CallerLoad("error", detail="pending record scope mismatch")
        status = rec["stage"] if rec["stage"] in ("pending", "acknowledged", "trimmed") else "error"
        if status == "error":
            return CallerLoad("error", detail="pending record corrupt")
        return CallerLoad(status, record=rec)

    async def claim(self, scope: str, scope_id: str,
                    pending: dict[str, Any]) -> tuple[str, dict[str, Any] | str]:
        """Claim the pending batch; concurrent callers get the existing one."""
        try:
            res = _decode_list(await self._redis.eval(
                _CALLER_CLAIM_V1, 1, caller_pending_key(scope, scope_id),
                _pending_json(pending),
            ))
        except Exception as exc:
            logger.warning("caller journal claim EVAL failed: %s", exc)
            return "error", f"journal EVAL failed: {exc}"
        if not res or type(res[0]) is not int or res[0] not in (0, 1) or len(res) != 3:
            return "error", "journal claim returned malformed reply"
        if res[0] == 1 and res[1] == "claimed":
            return "claimed", pending
        if res[0] == 0 and res[1] == "exists" and type(res[2]) is str:
            rec = parse_caller_record(res[2])
            if rec is None:
                return "error", "existing pending record corrupt"
            return "exists", rec
        return "error", "journal claim rejected"

    async def mark_acknowledged(self, scope: str, scope_id: str, pending: dict[str, Any],
                                trim_ids: list[str], ack: dict[str, Any]) -> tuple[str, dict[str, Any] | str]:
        """Persist validated ACK BEFORE trim; returns stored winner on already."""
        expected = json.dumps(list(pending.get("entry_ids") or []),
                              ensure_ascii=True, separators=(",", ":"))
        expected_fp = json.dumps(list(pending.get("fingerprints") or []),
                                 ensure_ascii=True, separators=(",", ":"))
        expected_plan = str(pending.get("plan_key") or "")
        expected_nonce = str(pending.get("caller_nonce") or "")
        try:
            res = _decode_list(await self._redis.eval(
                _CALLER_ACK_V1, 1, caller_pending_key(scope, scope_id),
                expected, _pending_json(build_caller_acked(pending, trim_ids, ack)),
                expected_fp, expected_plan, expected_nonce,
            ))
        except Exception as exc:
            logger.warning("caller journal ack EVAL failed: %s", exc)
            return "error", f"journal EVAL failed: {exc}"
        if not res or type(res[0]) is not int or res[0] not in (0, 1) or len(res) != 3:
            return "error", "journal ack returned malformed reply"
        if res[0] == 1 and res[1] in ("acked", "already") and type(res[2]) is str:
            rec = parse_caller_record(res[2])
            if rec is None:
                return "error", "stored ACK record corrupt"
            return res[1], rec
        if res[1] in ("missing", "mismatch", "corrupt"):
            return res[1], f"cannot acknowledge: {res[1]}"
        return "error", "journal ack rejected"

    async def clear(self, scope: str, scope_id: str, entry_ids: list[str],
                  caller_nonce: str = "") -> tuple[str, str]:
        """Delete only an acknowledged/trimmed record with matching IDs+nonce."""
        expected = json.dumps(list(entry_ids or []), ensure_ascii=True, separators=(",", ":"))
        try:
            res = _decode_list(await self._redis.eval(
                _CALLER_CLEAR_V1, 1, caller_pending_key(scope, scope_id),
                expected, str(caller_nonce or ""),
            ))
        except Exception as exc:
            logger.warning("caller journal clear EVAL failed: %s", exc)
            return "error", f"journal EVAL failed: {exc}"
        if not res or type(res[0]) is not int or res[0] not in (0, 1) or len(res) != 3:
            return "error", "journal clear returned malformed reply"
        if res[0] == 1 and res[1] == "cleared":
            return "cleared", ""
        return res[1] if res[1] in ("not_acked", "mismatch", "corrupt") else "error", \
            f"cannot clear: {res[1]}"

    async def mark_trimmed(self, scope: str, scope_id: str,
                           entry_ids: list[str], caller_nonce: str) -> tuple[str, str]:
        """CAS acknowledged -> trimmed after T1 trim (idempotent already)."""
        expected = json.dumps(list(entry_ids or []), ensure_ascii=True, separators=(",", ":"))
        try:
            res = _decode_list(await self._redis.eval(
                _CALLER_MARK_TRIMMED_V1, 1, caller_pending_key(scope, scope_id),
                expected, str(caller_nonce or ""),
            ))
        except Exception as exc:
            logger.warning("caller mark-trimmed EVAL failed: %s", exc)
            return "error", f"journal EVAL failed: {exc}"
        if not res or type(res[0]) is not int or res[0] not in (0, 1) or len(res) != 3:
            return "error", "journal mark-trimmed malformed reply"
        if res[0] == 1 and res[1] in ("trimmed", "already"):
            return res[1], ""
        return res[1] if res[1] in ("missing", "mismatch", "corrupt") else "error", \
            f"cannot mark-trimmed: {res[1]}"

    async def discard(self, scope: str, scope_id: str) -> bool:
        """Best-effort delete (manual T1 reset only; never auto-called)."""
        try:
            await self._redis.delete(caller_pending_key(scope, scope_id))
            return True
        except Exception as exc:
            logger.debug("caller journal discard failed: %s", exc)
            return False


async def discard_caller_pending(redis_client: Any, scope: str, scope_id: str) -> bool:
    journal = CallerJournal.from_redis(redis_client)
    if journal is None:
        return False
    return await journal.discard(scope, scope_id)


# ------------------------------------------------------------- selection

@dataclass(frozen=True, slots=True)
class CallerSelection:
    """Caller-side batch decision.

    ``ready`` ships ``entries`` (ActiveEntry objects on the auto path,
    shipped dicts when explicit) tracked by ``pending`` (None only for the
    explicit-[] passthrough, which claims nothing). ``empty`` means no
    unsummarized input (skip without dispatch). ``acknowledged`` means a
    durable ACK already exists: finalize it (trim/release/clear) without
    redispatch. ``error`` fails closed with ``reason``.
    """

    status: str  # "ready" | "empty" | "acknowledged" | "error"
    entries: list[Any] = field(default_factory=list)
    pending: dict[str, Any] | None = None
    reason: str = ""
    detail: str = ""


def _coerce_limit(limit: Any) -> int:
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = MAX_AUTO_BATCH
    return min(max(n, 1), MAX_AUTO_BATCH)


async def _reuse_pending(t1: Any, scope: str, scope_id: str,
                         record: dict[str, Any]) -> CallerSelection:
    """Exact-fetch pinned IDs order-preservingly, window-independently."""
    try:
        fetch = t1.get_entries_by_ids
    except AttributeError:
        return CallerSelection("error", reason="journal_error",
                               detail="T1 facade lacks exact-ID fetch")
    try:
        found = await fetch(scope, scope_id, list(record.get("entry_ids") or []))
    except Exception as exc:
        return CallerSelection("error", reason="journal_error",
                               detail=f"exact ID fetch failed: {exc}")
    if entry_ids_of(found or []) != list(record.get("entry_ids") or []):
        return CallerSelection("error", reason="pending_unrecoverable",
                               detail="pinned entries missing from T1")
    if not incoming_matches_pending(list(found or []), record):
        return CallerSelection("error", reason="pending_unrecoverable",
                               detail="pinned entries altered (content/role/order/time)")
    return CallerSelection("ready", entries=list(found or []), pending=record)


async def select_caller_batch(*, t1: Any, journal: CallerJournal | None,
                              scope: str, scope_id: str,
                              explicit: list[dict] | None,
                              limit: int = MAX_AUTO_BATCH) -> CallerSelection:
    """Pending-first batch selection shared by coordinator + local tool read.

    Explicit [] passes through untouched (no claim). Explicit shipments are
    checked against pending (incompatible overlap is rejected, never
    silently replaced). Auto (None) reuses pending via exact ID fetch, else
    claims the first fresh unsummarized batch (bounded, summarized-tail
    excluded). Missing/corrupt journal or facade capability fails closed.
    """
    if explicit is not None and len(explicit) == 0:
        return CallerSelection("ready", entries=[])
    if journal is None:
        return CallerSelection("error", reason="journal_error",
                               detail="caller journal unavailable")
    loaded = await journal.load(scope, scope_id)
    if loaded.status == "error":
        return CallerSelection("error", reason="journal_error", detail=loaded.detail)
    if loaded.status in ("acknowledged", "trimmed"):
        assert loaded.record is not None
        if explicit is None or incoming_matches_pending(explicit, loaded.record):
            return CallerSelection("acknowledged", pending=loaded.record)
        return CallerSelection("error", reason="pending_mismatch",
                               detail="previous batch has a durable ACK pending finalize")
    if loaded.status == "pending":
        assert loaded.record is not None
        if explicit is None:
            return await _reuse_pending(t1, scope, scope_id, loaded.record)
        if incoming_matches_pending(explicit, loaded.record):
            return CallerSelection("ready", entries=list(explicit), pending=loaded.record)
        return CallerSelection("error", reason="pending_mismatch",
                               detail="explicit shipment overlaps a different pending batch")
    # No pending: claim first.
    if explicit is None:
        try:
            fresh = await t1.list_unsummarized_entries(
                scope, scope_id, limit=_coerce_limit(limit))
        except AttributeError:
            return CallerSelection("error", reason="journal_error",
                                   detail="T1 facade lacks unsummarized selection")
        except Exception as exc:
            return CallerSelection("error", reason="journal_error",
                                   detail=f"fresh selection failed: {exc}")
        if not fresh:
            return CallerSelection("empty")
        pending = build_caller_pending(scope, scope_id, list(fresh))
        status, rec = await journal.claim(scope, scope_id, pending)
        if status == "claimed":
            return CallerSelection("ready", entries=list(fresh), pending=pending)
        if status == "exists" and isinstance(rec, dict):
            if rec.get("stage") in ("acknowledged", "trimmed"):
                return CallerSelection("acknowledged", pending=rec)
            if rec.get("stage") == "pending":
                return await _reuse_pending(t1, scope, scope_id, rec)
        detail = rec if isinstance(rec, str) else "concurrent pending unreadable"
        return CallerSelection("error", reason="journal_error", detail=detail)
    pending = build_caller_pending(scope, scope_id, explicit)
    status, rec = await journal.claim(scope, scope_id, pending)
    if status == "claimed":
        return CallerSelection("ready", entries=list(explicit), pending=pending)
    if status == "exists" and isinstance(rec, dict):
        if rec.get("stage") in ("acknowledged", "trimmed") and incoming_matches_pending(explicit, rec):
            return CallerSelection("acknowledged", pending=rec)
        if rec.get("stage") == "pending" and incoming_matches_pending(explicit, rec):
            return CallerSelection("ready", entries=list(explicit), pending=rec)
        return CallerSelection("error", reason="pending_mismatch",
                               detail="concurrent pending batch differs")
    detail = rec if isinstance(rec, str) else "concurrent pending unreadable"
    return CallerSelection("error", reason="journal_error", detail=detail)


class LocalBatchError(Exception):
    """Local T1 batch selection failed closed (tool maps to failed JSON)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


async def read_local_tool_batch(memory_manager: Any, scope: str, scope_id: str,
                                limit: Any) -> list[Any]:
    """Pending-first local read for direct tool calls (no trim here).

    A direct consolidate_memory call with entries=None pins/reuses the same
    caller batch the coordinator would ship, so a direct local re-read never
    moves the batch. The returned ACK must still be finalized by the
    coordinator (validated trim + receiver release); this helper never trims.
    """
    t1 = memory_manager.t1
    journal = CallerJournal.from_redis(getattr(getattr(t1, "store", None), "redis", None))
    sel = await select_caller_batch(t1=t1, journal=journal, scope=scope,
                                    scope_id=scope_id, explicit=None, limit=limit)
    if sel.status == "ready":
        return sel.entries
    if sel.status == "empty":
        return []
    if sel.status == "acknowledged":
        raise LocalBatchError(
            "ack_pending_finalize",
            "a durable ACK is pending finalize; run the coordinator (trim+release) first",
        )
    raise LocalBatchError(sel.reason or "journal_error", sel.detail)
