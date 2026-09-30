"""Focused trust-boundary tests for ProfileFileIO (symlink/FIFO/atomicity).

Synthetic tmp fixtures only; dummy secret never leaves tmp_path.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import stat

import pytest

from twin.shared.memory.profile.file_io import ProfileFileIO
from twin.shared.memory.profile.markdown_store import MarkdownProfileStore, profile_hash


USER = "user_123"
DUMMY_SECRET = "DUMMY-PRIVATE-KEY-SYNTHETIC-9f3a7c1e-outside"
VICTIM_TEXT = "victim-original-synthetic-keep"
LEGIT_TEXT = "## Legit\n- synthetic-ok\n"


def _store(base) -> MarkdownProfileStore:
    return MarkdownProfileStore(base_path=str(base))


async def test_profile_symlink_to_outside_secret_rejected(tmp_path):
    base = tmp_path / "profiles"
    outside = tmp_path / "outside_secret.txt"
    outside.write_text(DUMMY_SECRET, encoding="utf-8")
    store = _store(base)
    target = base / f"{USER}.md"
    target.unlink(missing_ok=True)
    os.symlink(str(outside), str(target))

    with pytest.raises(OSError) as ei:
        await store.read_raw(USER)
    assert DUMMY_SECRET not in str(ei.value)

    with pytest.raises(OSError):
        ProfileFileIO.read_text_sync(target)
    # Skeleton path also rejects; must not return private bytes.
    with pytest.raises(OSError) as ei2:
        await store._io.ensure_file(target)
    assert DUMMY_SECRET not in str(ei2.value)
    assert outside.read_text(encoding="utf-8") == DUMMY_SECRET


async def test_predictable_tmp_symlink_cannot_truncate_victim(tmp_path):
    base = tmp_path / "profiles"
    victim = tmp_path / "victim.txt"
    victim.write_text(VICTIM_TEXT, encoding="utf-8")
    store = _store(base)
    assert await store.write_raw(USER, LEGIT_TEXT) is True
    old_tmp = base / f"{USER}.md.tmp"
    if old_tmp.exists() or old_tmp.is_symlink():
        old_tmp.unlink()
    os.symlink(str(victim), str(old_tmp))

    assert await store.write_raw(USER, LEGIT_TEXT + "- more\n") is True
    assert victim.read_text(encoding="utf-8") == VICTIM_TEXT
    assert old_tmp.is_symlink()
    assert await store.read_raw(USER) == LEGIT_TEXT + "- more\n"


async def test_lock_symlink_rejected(tmp_path):
    base = tmp_path / "profiles"
    outside = tmp_path / "outside_lock_target.txt"
    outside.write_text(DUMMY_SECRET, encoding="utf-8")
    store = _store(base)
    lock_path = base / ".locks" / f"{USER}.lock"
    if lock_path.exists() or lock_path.is_symlink():
        lock_path.unlink()
    os.symlink(str(outside), str(lock_path))

    with pytest.raises(OSError) as ei:
        async with store._io.locked(USER):
            pass
    assert DUMMY_SECRET not in str(ei.value)
    with pytest.raises(OSError):
        await store.read_raw(USER)
    assert outside.read_text(encoding="utf-8") == DUMMY_SECRET


async def test_redirected_locks_dir_rejected(tmp_path):
    base = tmp_path / "profiles"
    store = _store(base)
    evil_dir = tmp_path / "evil_locks"
    evil_dir.mkdir()
    (evil_dir / f"{USER}.lock").write_text("evil", encoding="utf-8")
    shutil.rmtree(base / ".locks")
    os.symlink(str(evil_dir), str(base / ".locks"))

    with pytest.raises(OSError):
        async with store._io.locked(USER):
            pass
    with pytest.raises(OSError):
        await store.read_raw(USER)


async def test_fifo_profile_rejected_without_hang(tmp_path):
    base = tmp_path / "profiles"
    store = _store(base)
    target = base / f"{USER}.md"
    target.unlink(missing_ok=True)
    os.mkfifo(str(target))

    with pytest.raises(OSError):
        ProfileFileIO.read_text_sync(target)
    with pytest.raises(OSError):
        await store.read_raw(USER)


async def test_legitimate_rw_and_concurrent_cas(tmp_path):
    base = tmp_path / "profiles"
    store = _store(base)
    assert await store.write_raw(USER, LEGIT_TEXT) is True
    assert await store.read_raw(USER) == LEGIT_TEXT
    mode = stat.S_IMODE(os.stat(base / f"{USER}.md").st_mode)
    assert mode == 0o600

    items = [f"cas-item-{i}" for i in range(5)]
    results = await asyncio.gather(
        *(store.append_raw(USER, "interest", it) for it in items)
    )
    assert all(results)
    assert sorted(await store.read_section(USER, "interest")) == sorted(items)

    before = await store.read_raw(USER)
    ok = await store.replace_section(
        USER, "basic", ["Cas: ok"], expected_profile_hash=profile_hash(before)
    )
    assert ok["ok"] is True and ok["conflict"] is False
    stale = await store.replace_section(
        USER, "basic", ["Cas: stale"], expected_profile_hash="stale"
    )
    assert stale["ok"] is False and stale["conflict"] is True
    assert await store.read_section(USER, "basic") == ["Cas: ok"]


async def test_write_failure_preserves_old_and_cleans_only_temp(tmp_path, monkeypatch):
    base = tmp_path / "profiles"
    store = _store(base)
    old = LEGIT_TEXT
    assert await store.write_raw(USER, old) is True
    keep = base / "keep.txt"
    keep.write_text("keep-synthetic", encoding="utf-8")

    def _fail_write(_fd, _data):
        raise OSError("injected write failure")

    monkeypatch.setattr(os, "write", _fail_write)
    with pytest.raises(OSError):
        ProfileFileIO.atomic_write_sync(base / f"{USER}.md", "new-bad-0")
    monkeypatch.undo()
    assert await store.read_raw(USER) == old
    assert keep.read_text(encoding="utf-8") == "keep-synthetic"
    assert [p for p in os.listdir(base) if ".tmp" in p] == []

    def _fail_fsync(_fd):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", _fail_fsync)
    with pytest.raises(OSError):
        ProfileFileIO.atomic_write_sync(base / f"{USER}.md", "new-bad")
    monkeypatch.undo()
    assert await store.read_raw(USER) == old
    assert keep.read_text(encoding="utf-8") == "keep-synthetic"
    assert [p for p in os.listdir(base) if ".tmp" in p] == []

    def _fail_replace(*_a, **_k):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(OSError):
        ProfileFileIO.atomic_write_sync(base / f"{USER}.md", "new-bad-2")
    monkeypatch.undo()
    assert await store.read_raw(USER) == old
    assert keep.read_text(encoding="utf-8") == "keep-synthetic"
    assert [p for p in os.listdir(base) if ".tmp" in p] == []
