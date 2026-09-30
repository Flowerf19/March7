"""Focused trust-boundary tests for UpdatePersonalityTool atomic write.

Synthetic tmp fixtures only; dummy secret never leaves tmp_path.
"""
from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from twin.shared.tools.modules.profile.update_personality_tool import (
    UpdatePersonalityTool,
)


DUMMY_SECRET = "DUMMY-PRIVATE-KEY-SYNTHETIC-9f3a7c1e-outside"
VICTIM_TEXT = "victim-original-synthetic-keep"
OLD_SOUL = "old soul synthetic\n"
NEW_SOUL = "new soul synthetic content"


def _tool(persona_dir: Path) -> UpdatePersonalityTool:
    return UpdatePersonalityTool(base_memory_path=str(persona_dir))


def _tmp_leftovers(persona_dir: Path) -> list[str]:
    return [p for p in os.listdir(persona_dir) if ".tmp" in p]


async def test_predictable_tmp_symlink_cannot_truncate_victim(tmp_path):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text(VICTIM_TEXT, encoding="utf-8")
    tool = _tool(persona_dir)
    await tool.execute(OLD_SOUL, target_file="SOUL.md")

    planted = persona_dir / "SOUL.md.tmp"
    if planted.exists() or planted.is_symlink():
        planted.unlink()
    os.symlink(str(victim), str(planted))

    result = await tool.execute(NEW_SOUL, target_file="SOUL.md")
    assert "Đã cập nhật SOUL.md" in result
    assert DUMMY_SECRET not in result
    assert victim.read_text(encoding="utf-8") == VICTIM_TEXT
    assert planted.is_symlink()
    assert (persona_dir / "SOUL.md").read_text(encoding="utf-8") == NEW_SOUL + "\n"


async def test_target_symlink_replaced_not_followed(tmp_path):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text(DUMMY_SECRET, encoding="utf-8")
    tool = _tool(persona_dir)
    await tool.execute(OLD_SOUL, target_file="SOUL.md")

    target = persona_dir / "SOUL.md"
    target.unlink()
    os.symlink(str(outside), str(target))

    result = await tool.execute(NEW_SOUL, target_file="SOUL.md")
    assert "Đã cập nhật SOUL.md" in result
    assert DUMMY_SECRET not in result
    assert outside.read_text(encoding="utf-8") == DUMMY_SECRET
    assert not target.is_symlink()
    assert stat.S_ISREG(os.stat(target).st_mode)
    assert target.read_text(encoding="utf-8") == NEW_SOUL + "\n"


async def test_fifo_target_replaced_without_hang(tmp_path):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    tool = _tool(persona_dir)
    await tool.execute(OLD_SOUL, target_file="SOUL.md")

    target = persona_dir / "SOUL.md"
    target.unlink()
    os.mkfifo(str(target))
    # Stale predictable tmp as FIFO must also be ignored, never opened.
    stale = persona_dir / "SOUL.md.tmp"
    if stale.exists() or stale.is_symlink():
        stale.unlink()
    os.mkfifo(str(stale))

    result = await tool.execute(NEW_SOUL, target_file="SOUL.md")
    assert "Đã cập nhật SOUL.md" in result
    assert not stat.S_ISFIFO(os.stat(target).st_mode)
    assert stat.S_ISREG(os.stat(target).st_mode)
    assert target.read_text(encoding="utf-8") == NEW_SOUL + "\n"
    assert stat.S_ISFIFO(os.stat(stale).st_mode)


def test_concurrent_writers_last_writer_wins_whole_payload(tmp_path):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    target = persona_dir / "SOUL.md"
    target.write_text(OLD_SOUL, encoding="utf-8")
    payload_a = ("A" * 200_000) + "\nA-END\n"
    payload_b = ("B" * 200_000) + "\nB-END\n"
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _write(payload: str) -> None:
        try:
            barrier.wait(timeout=10)
            UpdatePersonalityTool._atomic_write_sync(target, payload)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=_write, args=(payload_a,)),
        threading.Thread(target=_write, args=(payload_b,)),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=20)
    assert not any(th.is_alive() for th in threads)
    assert errors == []
    final = target.read_text(encoding="utf-8")
    assert final in (payload_a, payload_b)
    assert _tmp_leftovers(persona_dir) == []


async def test_write_failure_preserves_old_and_cleans_only_own_tmp(
    tmp_path, monkeypatch
):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text(VICTIM_TEXT, encoding="utf-8")
    keep = persona_dir / "keep.txt"
    keep.write_text("keep-synthetic", encoding="utf-8")
    tool = _tool(persona_dir)
    await tool.execute(OLD_SOUL, target_file="SOUL.md")
    target = persona_dir / "SOUL.md"

    planted = persona_dir / "SOUL.md.tmp"
    if planted.exists() or planted.is_symlink():
        planted.unlink()
    os.symlink(str(victim), str(planted))

    def _fail_write(_fd, _data):
        raise OSError("injected write failure")

    monkeypatch.setattr(os, "write", _fail_write)
    with pytest.raises(OSError):
        UpdatePersonalityTool._atomic_write_sync(target, "new-bad-0")
    monkeypatch.undo()
    assert target.read_text(encoding="utf-8") == OLD_SOUL
    assert keep.read_text(encoding="utf-8") == "keep-synthetic"
    assert victim.read_text(encoding="utf-8") == VICTIM_TEXT
    assert planted.is_symlink()
    assert _tmp_leftovers(persona_dir) == ["SOUL.md.tmp"]

    def _fail_fsync(_fd):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", _fail_fsync)
    with pytest.raises(OSError):
        UpdatePersonalityTool._atomic_write_sync(target, "new-bad")
    monkeypatch.undo()
    assert target.read_text(encoding="utf-8") == OLD_SOUL
    assert victim.read_text(encoding="utf-8") == VICTIM_TEXT
    assert planted.is_symlink()
    assert _tmp_leftovers(persona_dir) == ["SOUL.md.tmp"]

    def _fail_replace(*_a, **_k):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(OSError):
        UpdatePersonalityTool._atomic_write_sync(target, "new-bad-2")
    monkeypatch.undo()
    assert target.read_text(encoding="utf-8") == OLD_SOUL
    assert victim.read_text(encoding="utf-8") == VICTIM_TEXT
    assert planted.is_symlink()
    assert _tmp_leftovers(persona_dir) == ["SOUL.md.tmp"]

    # Fault through execute() still preserves old content and victim.
    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(Exception):
        await tool.execute("new-bad-3", target_file="SOUL.md")
    monkeypatch.undo()
    assert target.read_text(encoding="utf-8") == OLD_SOUL
    assert victim.read_text(encoding="utf-8") == VICTIM_TEXT
