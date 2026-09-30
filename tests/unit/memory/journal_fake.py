"""Shared fake Redis for consolidation-journal unit tests.

JournalFakeRedis mirrors the production Lua in
twin/shared/memory/consolidation_journal.py: bytes-on-GET, NX/EX SET, and
EVAL branches for RECEIVER_CLAIM/RELEASE + CALLER_CLAIM/ACK/CLEAR with the
same all-or-none semantics (sync dict ops linearize like single-threaded
Redis). Controllable clock (``now``), TTL recording, and error injection
(``get_error``/``set_error``/``eval_error``).

Not collected by pytest (filename matches no python_files pattern); owned
consol/cache/manager/active test modules subclass or instantiate it so
every fake exercises the real journal capability. There is deliberately no
production fallback for fakes without EVAL.
"""
from __future__ import annotations

import json
from typing import Any


class JournalFakeRedis:
    """In-memory async Redis strings + journal EVAL subset."""

    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.expiries: dict[str, float] = {}
        self.now = 0.0
        self.get_error: Exception | None = None
        self.set_error: Exception | None = None
        self.eval_error: Exception | None = None
        self.get_calls = 0
        self.set_calls = 0
        self.eval_calls = 0

    # ------------------------------------------------------------- strings

    def _collect(self, key: str) -> None:
        if key in self.expiries and self.now >= self.expiries[key]:
            self.strings.pop(key, None)
            self.expiries.pop(key, None)

    def _get_str(self, key: str) -> str | None:
        self._collect(key)
        value = self.strings.get(key)
        if value is None:
            return None
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def _set_str(self, key: str, value: Any, ex: int | None = None) -> None:
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        self.strings[key] = text
        if ex is not None:
            self.ttls[key] = int(ex)
            self.expiries[key] = self.now + int(ex)
        else:
            # Real SET without EX clears TTL (persist); mirror for refresh/PERSIST parity.
            self.ttls.pop(key, None)
            self.expiries.pop(key, None)

    async def get(self, key: str):
        self.get_calls += 1
        if self.get_error is not None:
            raise self.get_error
        value = self._get_str(str(key))
        return value.encode("utf-8") if isinstance(value, str) else None

    async def set(self, key: str, value, ex=None, nx=False):
        self.set_calls += 1
        if self.set_error is not None:
            raise self.set_error
        key = str(key)
        self._collect(key)
        if nx and key in self.strings:
            return None
        self._set_str(key, value, ex)
        return True

    async def expire(self, key: str, seconds: int) -> bool:
        key = str(key)
        self._collect(key)
        if key not in self.strings:
            return False
        self.ttls[key] = int(seconds)
        self.expiries[key] = self.now + int(seconds)
        return True

    async def delete(self, *keys: str) -> int:
        n = 0
        for key in keys:
            key = str(key)
            self._collect(key)
            if key in self.strings:
                del self.strings[key]
                n += 1
            self.expiries.pop(key, None)
        return n

    # ------------------------------------------------------------------ eval

    async def eval(self, script: str, numkeys: int, *keys_and_args):
        self.eval_calls += 1
        if self.eval_error is not None:
            raise self.eval_error
        keys = [str(k) for k in keys_and_args[:numkeys]]
        args = [a.decode("utf-8") if isinstance(a, bytes) else str(a)
                for a in keys_and_args[numkeys:]]
        if "RECEIVER_CLAIM_V1" in script:
            return self._receiver_claim(keys, args)
        if "RECEIVER_RELEASE_V1" in script:
            return self._receiver_release(keys, args)
        if "CALLER_CLAIM_V1" in script:
            return self._caller_claim(keys, args)
        if "CALLER_MARK_TRIMMED_V1" in script:
            return self._caller_mark_trimmed(keys, args)
        if "CALLER_ACK_V1" in script:
            return self._caller_ack(keys, args)
        if "CALLER_CLEAR_V1" in script:
            return self._caller_clear(keys, args)
        raise RuntimeError("unsupported EVAL script")

    # ------------------------------------------------------- script mirrors

    def _receiver_claim(self, keys: list[str], args: list[str]):
        n = int(args[0])
        batch_id, batch_json, payload = args[1], args[2], args[3]
        own_gen = args[4] if len(args) > 4 else ""
        expected = f"{batch_id}:{own_gen}"
        winner_gen = None
        for key in keys[:n]:
            cur = self._get_str(key)
            if cur is not None:
                if ":" not in cur:
                    return [0, "conflict", key, cur]
                cur_batch, _, cur_gen = cur.partition(":")
                if cur_batch != batch_id:
                    return [0, "conflict", key, cur]
                if winner_gen is None:
                    winner_gen = cur_gen
                elif winner_gen != cur_gen:
                    return [0, "batch_mismatch", ""]
        bkey, pkey = keys[n], keys[n + 1]
        eb = self._get_str(bkey)
        if eb is not None and eb != batch_json:
            return [0, "batch_mismatch", ""]
        ep = self._get_str(pkey)
        for key in keys[:n]:
            if self._get_str(key) is None:
                self.strings[key] = expected
            self.ttls.pop(key, None)
            self.expiries.pop(key, None)
        if eb is None:
            self.strings[bkey] = batch_json
        self.ttls.pop(bkey, None)
        self.expiries.pop(bkey, None)
        self.ttls.pop(pkey, None)
        self.expiries.pop(pkey, None)
        if ep is None:
            self.strings[pkey] = payload
            return [1, "claimed", payload, own_gen]
        ret_gen = winner_gen if winner_gen is not None else own_gen
        return [1, "adopted", ep, ret_gen]

    def _receiver_release(self, keys: list[str], args: list[str]):
        n = int(args[0])
        expected = args[1]
        try:
            ttl = int(args[2])
        except ValueError:
            ttl = 0
        for key in keys[:n]:
            cur = self._get_str(key)
            if cur is not None and cur != expected:
                return [0, "foreign", key]
        released = 0
        for key in keys[:n]:
            cur = self._get_str(key)
            if cur is not None and cur == expected:
                del self.strings[key]
                released += 1
        bkey = keys[n]
        self._collect(bkey)
        self.strings.pop(bkey, None)
        if ttl > 0:
            pkey = keys[n + 1]
            self._collect(pkey)
            if pkey in self.strings:
                self.ttls[pkey] = ttl
                self.expiries[pkey] = self.now + ttl
        return [released]

    def _caller_claim(self, keys: list[str], args: list[str]):
        cur = self._get_str(keys[0])
        if cur is None:
            self.strings[keys[0]] = args[0]
            return [1, "claimed", args[0]]
        return [0, "exists", cur]

    @staticmethod
    def _decode_json(text: str) -> Any:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _caller_ack(self, keys: list[str], args: list[str]):
        cur = self._get_str(keys[0])
        if cur is None:
            return [0, "missing", ""]
        rec = self._decode_json(cur)
        if not isinstance(rec, dict):
            return [0, "corrupt", ""]
        exp = self._decode_json(args[0])
        if not isinstance(exp, list):
            return [0, "mismatch", ""]
        ids = rec.get("entry_ids")
        if not isinstance(ids, list) or ids != exp:
            return [0, "mismatch", ""]
        if len(args) < 4:
            return [0, "mismatch", ""]
        efp = self._decode_json(args[2])
        if not isinstance(efp, list) or rec.get("fingerprints") != efp:
            return [0, "mismatch", ""]
        if rec.get("plan_key") != args[3]:
            return [0, "mismatch", ""]
        if len(args) < 5 or rec.get("caller_nonce") != args[4]:
            return [0, "mismatch", ""]
        if rec.get("stage") in ("acknowledged", "trimmed"):
            return [1, "already", cur]
        if rec.get("stage") != "pending":
            return [0, "mismatch", ""]
        self.strings[keys[0]] = args[1]
        return [1, "acked", args[1]]

    def _caller_clear(self, keys: list[str], args: list[str]):
        cur = self._get_str(keys[0])
        if cur is None:
            return [1, "cleared", ""]
        rec = self._decode_json(cur)
        if not isinstance(rec, dict):
            return [0, "corrupt", ""]
        if rec.get("stage") not in ("acknowledged", "trimmed"):
            return [0, "not_acked", ""]
        exp = self._decode_json(args[0])
        if not isinstance(exp, list):
            return [0, "mismatch", ""]
        ids = rec.get("entry_ids")
        if not isinstance(ids, list) or ids != exp:
            return [0, "mismatch", ""]
        if len(args) < 2 or rec.get("caller_nonce") != args[1]:
            return [0, "mismatch", ""]
        del self.strings[keys[0]]
        return [1, "cleared", ""]

    def _caller_mark_trimmed(self, keys: list[str], args: list[str]):
        cur = self._get_str(keys[0])
        if cur is None:
            return [0, "missing", ""]
        rec = self._decode_json(cur)
        if not isinstance(rec, dict):
            return [0, "corrupt", ""]
        exp = self._decode_json(args[0])
        if not isinstance(exp, list):
            return [0, "mismatch", ""]
        if rec.get("entry_ids") != exp:
            return [0, "mismatch", ""]
        if len(args) < 2 or rec.get("caller_nonce") != args[1]:
            return [0, "mismatch", ""]
        if rec.get("stage") == "trimmed":
            return [1, "already", cur]
        if rec.get("stage") != "acknowledged":
            return [0, "mismatch", ""]
        rec["stage"] = "trimmed"
        nxt = json.dumps(rec, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        self.strings[keys[0]] = nxt
        return [1, "trimmed", nxt]
