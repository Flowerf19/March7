"""In-memory Redis double with WATCH/MULTI/EXEC semantics for T2 CAS tests.

TxFakeRedis emulates just enough of redis-py asyncio for DiaryWriter:
HASH/string storage with per-key versions, TTL recording, a live
FT.SEARCH merge-candidate reply built from current HASH state (so a
merge retry observes the rival's commit), and pipelines that raise
WatchError when a watched key's version moves before EXEC.

Not collected by pytest (filename matches no python_files pattern);
import via ``from unit.memory.redis_tx_fake import TxFakeRedis``.
"""
from __future__ import annotations

import struct
from typing import Any

from twin.shared.memory.diary.writer import WatchError


def _to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return str(value).encode("utf-8")


class TxFakePipeline:
    """Minimal WATCH/MULTI/EXEC pipeline over a TxFakeRedis."""

    def __init__(self, fake: "TxFakeRedis"):
        self._fake = fake
        self._watched: dict[str, int] = {}
        self._queued: list[tuple] = []
        self._in_multi = False

    async def watch(self, *keys: str) -> None:
        for key in keys:
            self._watched[str(key)] = self._fake._versions.get(str(key), 0)

    async def unwatch(self) -> None:
        self._watched = {}

    async def reset(self) -> None:
        self._watched = {}
        self._queued = []
        self._in_multi = False

    # Immediate reads (production code only calls these pre-MULTI).
    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return await self._fake.hgetall(key)

    async def get(self, key: str) -> bytes | None:
        return await self._fake.get(key)

    async def exists(self, key: str) -> int:
        return await self._fake.exists(key)

    def multi(self) -> None:
        self._in_multi = True

    # Writes queue post-MULTI, apply immediately pre-MULTI (redis-py parity).
    def hset(self, key: str, mapping: dict[str, Any] | None = None, **kwargs: Any) -> "TxFakePipeline":
        mapping = dict(mapping or {})
        mapping.update(kwargs)
        if self._in_multi:
            self._queued.append(("hset", str(key), mapping))
        else:
            self._fake._apply_hset(str(key), mapping)
        return self

    def expire(self, key: str, seconds: int) -> "TxFakePipeline":
        if self._in_multi:
            self._queued.append(("expire", str(key), int(seconds)))
        else:
            self._fake._apply_expire(str(key), int(seconds))
        return self

    def set(self, key: str, value: Any) -> "TxFakePipeline":
        if self._in_multi:
            self._queued.append(("set", str(key), value))
        else:
            self._fake._apply_set(str(key), value)
        return self

    async def execute(self) -> list[bool]:
        for hook in list(self._fake.pre_execute_hooks):
            await hook()
        for key, seen in self._watched.items():
            if self._fake._versions.get(key, 0) != seen:
                raise WatchError(f"WATCH conflict on {key}")
        batch = list(self._queued)
        self._queued = []
        for op in batch:
            if op[0] == "hset":
                self._fake._apply_hset(op[1], op[2])
            elif op[0] == "expire":
                self._fake._apply_expire(op[1], op[2])
            elif op[0] == "set":
                self._fake._apply_set(op[1], op[2])
        if batch:
            self._fake.exec_batches.append(batch)
        self._watched = {}
        self._in_multi = False
        return [True] * len(batch)


class TxFakeRedis:
    """Transactional in-memory stand-in for redis.asyncio.Redis (T2 subset)."""

    def __init__(self, *, dim: int = 8):
        self.dim = dim
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.strings: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}
        self._versions: dict[str, int] = {}
        self.exec_batches: list[list[tuple]] = []
        self.pre_execute_hooks: list[Any] = []
        # Live merge-candidate backing: FT.SEARCH KNN-1 replies reflect the
        # CURRENT hash at merge_target (None -> empty [0] reply).
        self.merge_target: str | None = None
        self.merge_score: float = 0.2
        self.search_calls: list[tuple] = []
        self.other_commands: list[tuple] = []

    # ------------------------------------------------------- versioned state

    def _bump(self, key: str) -> None:
        self._versions[key] = self._versions.get(key, 0) + 1

    def _apply_hset(self, key: str, mapping: dict[str, Any]) -> None:
        h = self.hashes.setdefault(key, {})
        for field, value in mapping.items():
            if isinstance(field, str):
                field = field.encode()
            h[field] = _to_bytes(value)
        self._bump(key)

    def _apply_set(self, key: str, value: Any) -> None:
        self.strings[key] = _to_bytes(value)
        self._bump(key)

    def _apply_expire(self, key: str, seconds: int) -> None:
        self.ttls[key] = seconds  # TTL-only: not a content modification

    # ------------------------------------------------------- direct commands

    async def hset(self, key: str, mapping: dict[str, Any] | None = None, **kwargs: Any) -> int:
        mapping = dict(mapping or {})
        mapping.update(kwargs)
        self._apply_hset(str(key), mapping)
        return len(mapping)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return dict(self.hashes.get(str(key), {}))

    async def expire(self, key: str, seconds: int) -> bool:
        self._apply_expire(str(key), int(seconds))
        return True

    async def get(self, key: str) -> bytes | None:
        return self.strings.get(str(key))

    async def set(self, key: str, value: Any) -> bool:
        self._apply_set(str(key), value)
        return True

    async def exists(self, key: str) -> int:
        key = str(key)
        return 1 if (key in self.hashes or key in self.strings) else 0

    def pipeline(self, transaction: bool = True) -> TxFakePipeline:
        return TxFakePipeline(self)

    # ------------------------------------------------------- FT.SEARCH stub

    async def execute_command(self, *args: Any) -> Any:
        if args[0] == "FT.SEARCH":
            self.search_calls.append(args)
            return self._merge_candidate_reply()
        if args[0] == "FT.INFO":
            raise Exception("Unknown index name")
        if args[0] == "FT.CREATE":
            return "OK"
        self.other_commands.append(args)
        return []

    def _merge_candidate_reply(self) -> list[Any]:
        """RESP2 flat-list KNN-1 reply from the LIVE target hash."""
        if self.merge_target is None or self.merge_target not in self.hashes:
            return [0]
        fields: list[Any] = []
        for k, v in self.hashes[self.merge_target].items():
            fields += [k, v]
        fields += [b"score", str(self.merge_score).encode()]
        return [1, self.merge_target.encode(), fields]

    # ------------------------------------------------------- test conveniences

    def summary_hashes(self, prefix: str = "timeline:summary:") -> dict[str, dict[str, str]]:
        """Decoded HASH state for assertions (embedding shown as dim count)."""
        out: dict[str, dict[str, str]] = {}
        for key, h in self.hashes.items():
            if not key.startswith(prefix):
                continue
            doc: dict[str, str] = {}
            for k, v in h.items():
                name = k.decode() if isinstance(k, bytes) else str(k)
                if name == "embedding" and isinstance(v, bytes) and len(v) % 4 == 0:
                    doc[name] = f"<{len(v) // 4}f>"
                else:
                    doc[name] = v.decode() if isinstance(v, bytes) else str(v)
            out[key] = doc
        return out

    @staticmethod
    def pack(values: list[float]) -> bytes:
        return struct.pack(f"{len(values)}f", *values)
