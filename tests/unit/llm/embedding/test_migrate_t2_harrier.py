"""Unit tests for scripts/migrate_t2_harrier.py (fake Redis, fake embedder)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import struct
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[4] / "scripts" / "migrate_t2_harrier.py"


def _load_migrate_module():
    spec = importlib.util.spec_from_file_location("migrate_t2_harrier", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _fake_embed(text: str) -> list[float]:
    rng = random.Random(hashlib.sha256(text.encode()).digest())
    vec = [rng.uniform(-1.0, 1.0) for _ in range(640)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


async def _embed_fn(text: str) -> list[float]:
    return _fake_embed(text)


class FakeRedis:
    """Minimal async Redis surface for the migration script."""

    def __init__(self):
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.hsets: list[tuple[str, dict]] = []
        self.commands: list[tuple] = []
        self.index_dim: int | None = 1024

    async def scan_iter(self, match: bytes = b"*", count: int = 500):
        prefix = match.decode().rstrip("*")
        for key in sorted(self.hashes):
            if key.startswith(prefix):
                yield key.encode()

    async def hgetall(self, key) -> dict[bytes, bytes]:
        k = key.decode() if isinstance(key, bytes) else str(key)
        return dict(self.hashes.get(k, {}))

    async def hset(self, key, mapping=None, **kwargs) -> int:
        k = key.decode() if isinstance(key, bytes) else str(key)
        assert mapping is not None and set(mapping) == {"embedding"}, (
            f"migration must rewrite ONLY the embedding field, got {set(mapping or {})}"
        )
        encoded = {name.encode() if isinstance(name, str) else name: v for name, v in mapping.items()}
        self.hashes.setdefault(k, {}).update(encoded)
        self.hsets.append((k, dict(encoded)))
        return 1

    async def expire(self, key, seconds) -> bool:
        return True

    async def execute_command(self, *args):
        self.commands.append(tuple(args))
        op = str(args[0]).upper()
        assert op not in {"DEL", "UNLINK", "FLUSHALL", "FLUSHDB"}, f"forbidden: {args}"
        if op == "FT.INFO":
            return [
                "index_name", args[1], "attributes",
                [["identifier", "embedding", "attribute", "embedding",
                  "type", "VECTOR", "dim", self.index_dim]],
            ]
        if op == "FT.DROPINDEX":
            assert "DD" not in [str(a).upper() for a in args], "DROPINDEX must not use DD"
            return "OK"
        if op == "FT.CREATE":
            self.index_dim = 640
            return "OK"
        raise AssertionError(f"unexpected command: {args}")

    async def aclose(self) -> None:
        pass


def _seed(r: FakeRedis) -> dict[str, dict[bytes, bytes]]:
    def rec(summary: str, dim: int, topic: str = "food") -> dict[bytes, bytes]:
        vec = [0.01 * ((i % 7) + 1) for i in range(dim)]
        return {
            b"user_id": b"u1",
            b"summary": summary.encode(),
            b"topic": topic.encode(),
            b"importance": b"3",
            b"created_at": b"1759240000.0",
            b"version": b"2",
            b"day": b"2026-09-30",
            b"source_entry_ids": b'["e1","e2"]',
            b"embedding": _pack(vec),
        }

    originals = {
        "timeline:summary:aaa": rec("Hòa nấu bún bò Huế sáng nay.", 1024),
        "timeline:summary:bbb": rec("Mèo Mun leo lên mái nhà.", 1024, "pet"),
        "timeline:summary:ccc": rec("Leo núi Bà Đen cuối tuần.", 1024, "hiking"),
        "timeline:summary:done": rec("Đã migrate rồi.", 640),
    }
    bad_no_emb = rec("Thiếu vector.", 1024)
    del bad_no_emb[b"embedding"]
    originals["timeline:summary:noemb"] = bad_no_emb
    bad_empty = rec("", 1024)
    originals["timeline:summary:empty"] = bad_empty
    for key, fields in originals.items():
        r.hashes[key] = dict(fields)
    return {k: dict(v) for k, v in originals.items()}


def _run_kwargs(r, tmp_path: Path, **over) -> dict:
    base = dict(
        redis_url="redis://fake:6379", db=0, prefix="timeline:summary",
        index="timeline_summaries", backup_path=None,
        checkpoint_path=tmp_path / "ckpt.json", limit=0,
        apply=False, rollback=False, recreate_index=False,
        redis_client=r, embed_fn=_embed_fn,
    )
    base.update(over)
    return base


class TestDryRun:
    async def test_dry_run_zero_mutations(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        before = _seed(r)

        rc = await mig.run(**_run_kwargs(r, tmp_path))

        assert rc == 0
        assert r.hashes == before
        assert r.hsets == []
        assert [c for c in r.commands if c[0] != "FT.INFO"] == []


class TestApply:
    async def test_apply_migrates_legacy_and_validates(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        originals = _seed(r)
        backup = tmp_path / "t2.bak.jsonl"

        rc = await mig.run(**_run_kwargs(r, tmp_path, apply=True, backup_path=backup))

        assert rc == 0
        # Backup verified on disk (covers every record incl. done/bad).
        assert json.loads(backup.read_text(encoding="utf-8").splitlines()[0])["key"]
        assert len(mig.read_backup(backup)) == len(originals)
        # Legacy keys now 640-dim with fresh vectors; everything else identical.
        for key in ("timeline:summary:aaa", "timeline:summary:bbb", "timeline:summary:ccc"):
            live = r.hashes[key]
            assert len(live[b"embedding"]) // 4 == 640
            for name, value in originals[key].items():
                if name != b"embedding":
                    assert live[name] == value, (key, name)
        # Already-640 and bad keys untouched.
        assert r.hashes["timeline:summary:done"] == originals["timeline:summary:done"]
        assert r.hashes["timeline:summary:noemb"] == originals["timeline:summary:noemb"]
        assert r.hashes["timeline:summary:empty"] == originals["timeline:summary:empty"]
        # Checkpoint lists exactly the migrated keys.
        ckpt = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
        assert sorted(ckpt["migrated"]) == [
            "timeline:summary:aaa", "timeline:summary:bbb", "timeline:summary:ccc",
        ]

    async def test_apply_requires_backup(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        _seed(r)

        rc = await mig.run(**_run_kwargs(r, tmp_path, apply=True, backup_path=None))

        assert rc == 2
        assert r.hsets == []

    async def test_failure_preserves_backup_and_resumes(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        originals = _seed(r)
        backup = tmp_path / "t2.bak.jsonl"

        async def flaky(text: str) -> list[float]:
            if "Mun" in text:
                raise RuntimeError("simulated embed crash")
            return _fake_embed(text)

        rc = await mig.run(**_run_kwargs(
            r, tmp_path, apply=True, backup_path=backup, embed_fn=flaky,
        ))

        assert rc == 1
        assert len(mig.read_backup(backup)) == len(originals)  # backup intact
        ckpt = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
        assert ckpt["migrated"] == ["timeline:summary:aaa"]  # progress kept
        assert len(r.hashes["timeline:summary:bbb"][b"embedding"]) // 4 == 1024

        rc = await mig.run(**_run_kwargs(r, tmp_path, apply=True, backup_path=backup))

        assert rc == 0  # resumed to completion (aaa skipped as done)
        for key in ("timeline:summary:aaa", "timeline:summary:bbb", "timeline:summary:ccc"):
            assert len(r.hashes[key][b"embedding"]) // 4 == 640

    async def test_staged_limit_resumes(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        _seed(r)
        backup = tmp_path / "t2.bak.jsonl"

        rc = await mig.run(**_run_kwargs(
            r, tmp_path, apply=True, backup_path=backup, limit=1,
        ))
        assert rc == 0
        first = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
        assert len(first["migrated"]) == 1

        rc = await mig.run(**_run_kwargs(r, tmp_path, apply=True, backup_path=backup))
        assert rc == 0
        final = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
        assert len(final["migrated"]) == 3


    async def test_existing_backup_reused_never_overwritten(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        originals = _seed(r)
        backup = tmp_path / "t2.bak.jsonl"
        assert await mig.run(**_run_kwargs(r, tmp_path, apply=True, backup_path=backup)) == 0
        first_bytes = backup.read_bytes()

        # Second run (e.g. --recreate-index pass) must keep the ORIGINAL backup.
        assert await mig.run(**_run_kwargs(
            r, tmp_path, apply=True, backup_path=backup, recreate_index=True,
        )) == 0
        assert backup.read_bytes() == first_bytes

        # ... so rollback still restores the pre-migration vectors.
        assert await mig.run(**_run_kwargs(
            r, tmp_path, rollback=True, backup_path=backup,
        )) == 0
        for key in ("timeline:summary:aaa", "timeline:summary:bbb", "timeline:summary:ccc"):
            assert r.hashes[key][b"embedding"] == originals[key][b"embedding"]

    async def test_stale_backup_missing_keys_aborts(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        _seed(r)
        backup = tmp_path / "t2.bak.jsonl"
        mig.write_backup(backup, [("timeline:summary:aaa", r.hashes["timeline:summary:aaa"])])

        rc = await mig.run(**_run_kwargs(r, tmp_path, apply=True, backup_path=backup))

        assert rc == 1
        assert r.hsets == []


class TestRollback:
    async def test_rollback_restores_original_vectors(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        originals = _seed(r)
        backup = tmp_path / "t2.bak.jsonl"
        assert await mig.run(**_run_kwargs(r, tmp_path, apply=True, backup_path=backup)) == 0

        rc = await mig.run(**_run_kwargs(
            r, tmp_path, rollback=True, backup_path=backup,
        ))

        assert rc == 0
        for key in ("timeline:summary:aaa", "timeline:summary:bbb", "timeline:summary:ccc"):
            assert r.hashes[key][b"embedding"] == originals[key][b"embedding"]
            for name, value in originals[key].items():
                assert r.hashes[key][name] == value, (key, name)
        ckpt = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
        assert ckpt["migrated"] == []


class TestBackupIntegrity:
    def test_corrupt_backup_detected(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        backup = tmp_path / "t2.bak.jsonl"
        mig.write_backup(backup, [("k1", {b"summary": b"x", b"embedding": b"1234"})])
        lines = backup.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[0])
        rec["fields"]["summary"] = "eQ=="  # tamper: 'y' instead of 'x'
        backup.write_text(json.dumps(rec) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="backup corrupt"):
            mig.read_backup(backup)


class TestIndexRecreate:
    async def test_no_recreate_by_default(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        _seed(r)
        backup = tmp_path / "t2.bak.jsonl"

        assert await mig.run(**_run_kwargs(r, tmp_path, apply=True, backup_path=backup)) == 0
        assert [c for c in r.commands if c[0] in {"FT.DROPINDEX", "FT.CREATE"}] == []

    async def test_recreate_after_full_validation(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        _seed(r)
        backup = tmp_path / "t2.bak.jsonl"

        rc = await mig.run(**_run_kwargs(
            r, tmp_path, apply=True, backup_path=backup, recreate_index=True,
        ))

        assert rc == 0
        assert ("FT.DROPINDEX", "timeline_summaries") in r.commands  # no DD
        assert any(c[0] == "FT.CREATE" for c in r.commands)
        assert r.index_dim == 640

    async def test_recreate_refused_with_legacy_remaining(self, tmp_path: Path) -> None:
        mig = _load_migrate_module()
        r = FakeRedis()
        _seed(r)
        backup = tmp_path / "t2.bak.jsonl"

        rc = await mig.run(**_run_kwargs(
            r, tmp_path, apply=True, backup_path=backup, recreate_index=True, limit=1,
        ))

        assert rc == 1
        assert [c for c in r.commands if c[0] in {"FT.DROPINDEX", "FT.CREATE"}] == []
