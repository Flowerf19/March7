"""Secure key-file provisioning (private tmp paths, deliberate regeneration)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from system_gateway import keyfile
from system_gateway.cli.main import _cmd_pair


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _mode(path: Path) -> str:
    return oct(path.stat().st_mode)[-3:]


def test_write_new_creates_0600(tmp_path: Path):
    target = tmp_path / "sub" / "approval_secret"
    keyfile.write_new_secret_file(target, "owner-key-value")
    assert target.read_text(encoding="utf-8") == "owner-key-value"
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert _mode(target) == "600"
        assert _mode(target.parent) == "700"


def test_write_new_never_silently_replaces(tmp_path: Path):
    target = tmp_path / "approval_secret"
    keyfile.write_new_secret_file(target, "first")
    with pytest.raises(FileExistsError):
        keyfile.write_new_secret_file(target, "second")
    assert target.read_text(encoding="utf-8") == "first"


def test_overwrite_atomic_replaces_for_explicit_regeneration(tmp_path: Path):
    target = tmp_path / "approval_secret"
    keyfile.write_new_secret_file(target, "first")
    keyfile.overwrite_secret_file_atomic(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert _mode(target) == "600"


def test_read_missing_returns_none(tmp_path: Path):
    assert keyfile.read_secret_file(tmp_path / "missing") is None


@pytest.mark.skipif(
    not hasattr(__import__("os"), "O_NOFOLLOW"), reason="O_NOFOLLOW unavailable"
)
def test_read_refuses_symlink(tmp_path: Path):
    real = tmp_path / "real"
    real.write_text("secret-value", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert keyfile.read_secret_file(link) is None


def test_pair_provisions_both_0600_without_leak(tmp_path: Path, capsys):
    secret_file = tmp_path / "secret"
    approval_file = tmp_path / "approval_secret"
    args = _Namespace(
        secret_file=str(secret_file),
        approval_secret_file=str(approval_file),
        force=False,
    )
    assert _cmd_pair(args) == 0
    approval = approval_file.read_text(encoding="utf-8").strip()
    assert approval and len(approval) >= 32
    out = capsys.readouterr().out
    assert approval not in out
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert _mode(secret_file) == "600"
        assert _mode(approval_file) == "600"


def test_pair_keeps_existing_approval_without_force(tmp_path: Path, capsys):
    secret_file = tmp_path / "secret"
    approval_file = tmp_path / "approval_secret"
    approval_file.write_text("keep-approval-key", encoding="utf-8")
    args = _Namespace(
        secret_file=str(secret_file),
        approval_secret_file=str(approval_file),
        force=False,
    )
    assert _cmd_pair(args) == 0
    assert approval_file.read_text(encoding="utf-8") == "keep-approval-key"
    assert "keep-approval-key" not in capsys.readouterr().out


def test_pair_force_regenerates_both_deliberately(tmp_path: Path, capsys):
    secret_file = tmp_path / "secret"
    approval_file = tmp_path / "approval_secret"
    secret_file.write_text("old-request", encoding="utf-8")
    approval_file.write_text("old-approval", encoding="utf-8")
    args = _Namespace(
        secret_file=str(secret_file),
        approval_secret_file=str(approval_file),
        force=True,
    )
    assert _cmd_pair(args) == 0
    assert secret_file.read_text(encoding="utf-8") != "old-request"
    new_approval = approval_file.read_text(encoding="utf-8")
    assert new_approval != "old-approval"
    out = capsys.readouterr().out
    assert "old-approval" not in out
    assert new_approval.strip() not in out
