#!/usr/bin/env python3
"""migrate_t2_harrier.py — re-embed retained T2 HASHes from legacy 1024-dim
(Qwen) vectors to Harrier 640-dim vectors. CODE ONLY: dry-run by default.

Safety contract:
  - Dry-run unless --apply is passed; --apply requires --backup PATH.
  - The backup (full HASH content per key + SHA-256) is written and
    re-verified BEFORE any Redis write. Any failure stops the run with the
    backup intact and a checkpoint of completed keys (resumable: re-run).
  - Only the `embedding` field is ever rewritten; key names, IDs, summaries,
    topics, provenance (source_entry_ids), TTLs, and all other fields are
    preserved byte-for-byte and validated against the backup afterwards.
  - NEVER issues DEL, UNLINK, FLUSHALL, or DROPINDEX ... DD. Index recreation
    (DROPINDEX without DD + FT.CREATE) happens only with --recreate-index,
    only after every migrated record validates, and only when no legacy-dim
    vector remains.

Usage (owner-run, host with Redis + model access):
    python3 scripts/migrate_t2_harrier.py --dry-run --redis-url redis://localhost:6379
    python3 scripts/migrate_t2_harrier.py --apply --backup /safe/t2.bak.jsonl
    python3 scripts/migrate_t2_harrier.py --apply --backup /safe/t2.bak.jsonl --recreate-index
    python3 scripts/migrate_t2_harrier.py --rollback --backup /safe/t2.bak.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)

import redis.asyncio as aioredis  # noqa: E402

from twin.shared.config.settings import Config  # noqa: E402
from twin.shared.llm.embedding import HarrierEmbeddingService  # noqa: E402
from twin.shared.memory.diary.codec import pack_embedding, unpack_embedding  # noqa: E402
from twin.shared.memory.diary.schema import (  # noqa: E402
    create_timeline_index,
    extract_indexed_dim,
)

LEGACY_DIM = 1024
HARRIER_DIM = 640
CHECKPOINT_VERSION = 1

EmbedFn = Callable[[str], Awaitable[list[float]]]


# ------------------------------------------------------------------- backup

def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def record_sha256(key: str, fields: dict[bytes, bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(key.encode("utf-8"))
    for name in sorted(fields):
        digest.update(name)
        digest.update(b"\x00")
        digest.update(fields[name])
        digest.update(b"\x00")
    return digest.hexdigest()


def write_backup(path: Path, records: list[tuple[str, dict[bytes, bytes]]]) -> None:
    """Write full HASH content + per-record SHA-256 as JSONL (atomic)."""
    staging = path.with_suffix(path.suffix + ".tmp")
    with staging.open("w", encoding="utf-8") as fh:
        for key, fields in records:
            fh.write(json.dumps({
                "key": key,
                "sha256": record_sha256(key, fields),
                "fields": {k.decode("utf-8", "surrogateescape"): _b64(v) for k, v in fields.items()},
            }, ensure_ascii=False) + "\n")
    os.replace(staging, path)


def read_backup(path: Path) -> list[dict[str, Any]]:
    """Read + integrity-check a backup file; raises on any corruption."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            fields = {k.encode("utf-8", "surrogateescape"): _unb64(v) for k, v in rec["fields"].items()}
            if record_sha256(rec["key"], fields) != rec["sha256"]:
                raise ValueError(f"backup corrupt at line {lineno} (key {rec['key']})")
            rec["_raw"] = fields
            records.append(rec)
    return records


# --------------------------------------------------------------- checkpoint

def load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unknown checkpoint version in {path}")
    return set(data.get("migrated", []))


def save_checkpoint(path: Path, migrated: set[str], backup: str) -> None:
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(
        {"version": CHECKPOINT_VERSION, "backup": backup, "migrated": sorted(migrated)},
        ensure_ascii=False, indent=2,
    ) + "\n", encoding="utf-8")
    os.replace(staging, path)


# -------------------------------------------------------------------- redis

async def scan_keys(r: Any, prefix: str) -> list[str]:
    keys = []
    async for key in r.scan_iter(match=f"{prefix}:*".encode(), count=500):
        keys.append(key.decode() if isinstance(key, bytes) else str(key))
    return sorted(keys)


def embedding_dim_of(fields: dict[bytes, bytes]) -> int | None:
    raw = fields.get(b"embedding")
    if raw is None:
        return None
    return len(raw) // 4


def plan(records: list[tuple[str, dict[bytes, bytes]]]) -> dict[str, Any]:
    """Classify keys: migrate (legacy 1024) / done (already 640) / bad."""
    migrate, done, bad = [], [], []
    for key, fields in records:
        dim = embedding_dim_of(fields)
        if dim == HARRIER_DIM:
            done.append(key)
        elif dim == LEGACY_DIM:
            summary = fields.get(b"summary", b"").decode("utf-8", "replace").strip()
            if not summary:
                bad.append((key, "legacy vector but empty/missing summary: cannot re-embed"))
            else:
                migrate.append(key)
        elif dim is None:
            bad.append((key, "missing embedding field"))
        else:
            bad.append((key, f"unexpected embedding dim {dim} (want {LEGACY_DIM} or {HARRIER_DIM})"))
    return {"migrate": migrate, "done": done, "bad": bad}


def check_vector(vec: list[float]) -> str | None:
    """Return None when vec is a usable 640-dim Harrier vector, else a reason."""
    if len(vec) != HARRIER_DIM:
        return f"dim {len(vec)} != {HARRIER_DIM}"
    if not all(math.isfinite(x) for x in vec):
        return "non-finite values"
    norm = math.sqrt(sum(x * x for x in vec))
    if abs(norm - 1.0) > 1e-3:
        return f"not unit norm ({norm:.4f})"
    return None


async def migrate_one(r: Any, key: str, embed: EmbedFn) -> None:
    """Re-embed one HASH's summary; rewrites ONLY the embedding field."""
    fields = await r.hgetall(key)
    if embedding_dim_of(fields) != LEGACY_DIM:
        raise ValueError(f"{key}: no longer legacy-dim, refusing blind write")
    summary = fields.get(b"summary", b"").decode("utf-8")
    vec = await embed(f"{Config.EMBEDDING_PASSAGE_PREFIX}{summary}")
    problem = check_vector(vec)
    if problem is not None:
        raise ValueError(f"{key}: fresh embedding unusable ({problem})")
    packed = pack_embedding(vec)
    await r.hset(key, mapping={"embedding": packed})  # TTL + other fields untouched
    reread = await r.hgetall(key)
    if reread.get(b"embedding") != packed:
        raise ValueError(f"{key}: read-back mismatch after HSET")
    for name, value in fields.items():
        if name == b"embedding":
            continue
        if reread.get(name) != value:
            raise ValueError(f"{key}: field {name!r} changed during migration")


async def validate_against_backup(r: Any, backup: list[dict[str, Any]], keys: set[str]) -> list[str]:
    """Check migrated keys: other fields byte-identical, embedding now 640."""
    errors = []
    wanted = {rec["key"]: rec for rec in backup if rec["key"] in keys}
    for key, rec in sorted(wanted.items()):
        live = await r.hgetall(key)
        if not live:
            errors.append(f"{key}: key vanished")
            continue
        for name, value in rec["_raw"].items():
            if name == b"embedding":
                continue
            if live.get(name) != value:
                errors.append(f"{key}: field {name!r} differs from backup")
        if embedding_dim_of(live) != HARRIER_DIM:
            errors.append(f"{key}: embedding dim is not {HARRIER_DIM} after migration")
    return errors


# --------------------------------------------------------------------- flows

async def _real_embedder() -> tuple[EmbedFn, Any]:
    svc = HarrierEmbeddingService(
        model_dir=Config.HARRIER_MODEL_DIR, expected_dim=HARRIER_DIM,
    )

    async def _embed(text: str) -> list[float]:
        return await svc.get_embedding(text)

    return _embed, svc


async def run(
    *,
    redis_url: str,
    db: int,
    prefix: str,
    index: str,
    backup_path: Path | None,
    checkpoint_path: Path,
    limit: int,
    apply: bool,
    rollback: bool,
    recreate_index: bool,
    redis_client: Any = None,
    embed_fn: EmbedFn | None = None,
) -> int:
    if rollback:
        assert backup_path is not None, "--rollback requires --backup PATH"
        return await run_rollback(
            redis_url, db, backup_path, checkpoint_path, redis_client=redis_client,
        )
    if apply and backup_path is None:
        print("ERROR: --apply requires --backup PATH (verified backup first)", file=sys.stderr)
        return 2
    if recreate_index and not apply:
        print("ERROR: --recreate-index requires --apply", file=sys.stderr)
        return 2

    own_client = redis_client is None
    r = redis_client or aioredis.Redis.from_url(
        redis_url, db=db, password=Config.REDIS_PASSWORD, decode_responses=False,
    )
    svc = None
    try:
        keys = await scan_keys(r, prefix)
        records = [(key, await r.hgetall(key)) for key in keys]
        records = [(key, h) for key, h in records if h]
        report = plan(records)
        print(f"keys: {len(records)} total | to migrate: {len(report['migrate'])} | "
              f"already-640: {len(report['done'])} | bad: {len(report['bad'])}")
        for key, reason in report["bad"]:
            print(f"  BAD {key}: {reason} (skipped, never written)")

        if not apply:
            for key in report["migrate"][:20]:
                print(f"  would migrate {key}")
            if len(report["migrate"]) > 20:
                print(f"  ... and {len(report['migrate']) - 20} more (dry-run, zero writes)")
            print("dry-run: no writes performed (pass --apply --backup PATH to migrate)")
            return 0

        assert backup_path is not None
        # Backup EVERYTHING first, then verify before any write. An existing
        # backup is REUSED (never overwritten) so staged/resumed runs keep
        # the original pre-migration rollback point.
        if backup_path.exists():
            backup = read_backup(backup_path)  # raises if corrupt
            covered = {rec["key"] for rec in backup}
            missing = {key for key, _ in records} - covered
            if missing:
                print(f"ERROR: existing backup {backup_path} misses "
                      f"{len(missing)} current key(s); move it aside and re-run "
                      "to take a fresh backup", file=sys.stderr)
                return 1
            print(f"reusing verified backup: {len(backup)} records -> {backup_path}")
        else:
            write_backup(backup_path, records)
            backup = read_backup(backup_path)
            if len(backup) != len(records):
                print("ERROR: backup record count mismatch; aborting before any write",
                      file=sys.stderr)
                return 1
            print(f"backup verified: {len(backup)} records -> {backup_path}")

        migrated = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else set()
        todo = [k for k in report["migrate"] if k not in migrated]
        if limit > 0:
            todo = todo[:limit]
        print(f"checkpoint: {len(migrated)} already done, {len(todo)} to do this run")

        if embed_fn is None:
            embed_fn, svc = await _real_embedder()
        before = len(migrated)
        failed = False
        for key in todo:
            try:
                await migrate_one(r, key, embed_fn)
            except Exception as exc:
                print(f"ERROR: {key}: {exc} — stopping, progress kept in checkpoint",
                      file=sys.stderr)
                failed = True
                break
            migrated.add(key)
            save_checkpoint(checkpoint_path, migrated, str(backup_path))
        print(f"migrated this run: {len(migrated) - before} total={len(migrated)}")

        errors = await validate_against_backup(r, backup, migrated)
        if errors:
            print("VALIDATION FAILED:", file=sys.stderr)
            for err in errors[:20]:
                print(f"  {err}", file=sys.stderr)
            return 1
        print(f"validation OK: {len(migrated)} migrated record(s) match backup + 640-dim")

        remaining_legacy = [k for k in report["migrate"] if k not in migrated]
        if recreate_index:
            if remaining_legacy or failed:
                print(f"ERROR: refusing index recreate with {len(remaining_legacy)} "
                      "legacy key(s) unmigrated", file=sys.stderr)
                return 1
            return await recreate_index_if_needed(r, index, prefix)
        if remaining_legacy:
            print(f"staged run: {len(remaining_legacy)} legacy key(s) remain; re-run to resume")
        else:
            print("all legacy keys migrated; re-run with --recreate-index to rebuild "
                  "the RediSearch index (DROPINDEX without DD keeps all HASHes)")
        return 0 if not failed else 1
    finally:
        if svc is not None:
            await svc.close()
        if own_client:
            await r.aclose()


async def run_rollback(
    redis_url: str, db: int, backup_path: Path, checkpoint_path: Path,
    *, redis_client: Any = None,
) -> int:
    backup = read_backup(backup_path)
    own_client = redis_client is None
    r = redis_client or aioredis.Redis.from_url(
        redis_url, db=db, password=Config.REDIS_PASSWORD, decode_responses=False,
    )
    try:
        restored, skipped = 0, 0
        for rec in backup:
            key = rec["key"]
            live = await r.hgetall(key)
            if not live:
                print(f"  SKIP {key}: key absent, nothing to restore")
                skipped += 1
                continue
            orig = rec["_raw"].get(b"embedding")
            if orig is None or live.get(b"embedding") == orig:
                skipped += 1
                continue
            await r.hset(key, mapping={"embedding": orig})
            reread = await r.hgetall(key)
            if reread.get(b"embedding") != orig:
                print(f"ERROR: {key}: rollback read-back mismatch", file=sys.stderr)
                return 1
            restored += 1
        # Rolled-back keys are legacy again: drop them from the checkpoint.
        if checkpoint_path.exists():
            migrated = load_checkpoint(checkpoint_path)
            migrated -= {rec["key"] for rec in backup}
            save_checkpoint(checkpoint_path, migrated, str(backup_path))
        print(f"rollback done: {restored} restored, {skipped} unchanged/absent")
        return 0
    finally:
        if own_client:
            await r.aclose()


async def recreate_index_if_needed(r: Any, index: str, prefix: str) -> int:
    try:
        info = await r.execute_command("FT.INFO", index)
    except Exception:
        info = None
    dim = extract_indexed_dim(info) if info is not None else None
    if dim == HARRIER_DIM:
        print(f"index {index!r} already DIM={HARRIER_DIM}; nothing to do")
        return 0
    print(f"index {index!r} DIM={dim}: DROPINDEX (docs kept) + recreate at {HARRIER_DIM}")
    await r.execute_command("FT.DROPINDEX", index)  # no DD: HASHes are preserved
    await create_timeline_index(r, index, prefix, HARRIER_DIM)
    print(f"index {index!r} recreated at DIM={HARRIER_DIM}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redis-url", default="redis://localhost:6379")
    ap.add_argument("--db", type=int, default=Config.TIMELINE_REDIS_DB)
    ap.add_argument("--prefix", default="timeline:summary")
    ap.add_argument("--index", default="timeline_summaries")
    ap.add_argument("--backup", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=Path("migrate_t2_harrier.checkpoint.json"))
    ap.add_argument("--limit", type=int, default=0, help="cap migrated keys this run (0=all)")
    ap.add_argument("--apply", action="store_true", help="write (default is dry-run)")
    ap.add_argument("--rollback", action="store_true", help="restore embeddings from --backup")
    ap.add_argument("--recreate-index", action="store_true",
                    help="with --apply: DROPINDEX (no DD) + recreate at 640 after validation")
    args = ap.parse_args(argv)
    return asyncio.run(run(
        redis_url=args.redis_url, db=args.db, prefix=args.prefix, index=args.index,
        backup_path=args.backup, checkpoint_path=args.checkpoint, limit=args.limit,
        apply=args.apply, rollback=args.rollback, recreate_index=args.recreate_index,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
