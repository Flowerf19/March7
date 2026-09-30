"""Secure host-private key file I/O (stdlib only).

Creates secrets with restrictive mode AT CREATION (no 0644 window), refuses
symlinks on read, and never silently replaces an existing owner key: callers
must choose exclusive-create vs explicit atomic overwrite.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


class KeyFileError(OSError):
    """Raised for key-file I/O failures (never carries secret values)."""


def ensure_private_dir(path: Path, mode: int = 0o700) -> None:
    """Create a directory with owner-only permissions (best-effort on Win)."""

    Path(path).mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except OSError:  # pragma: no cover - Windows ACLs
        pass


def read_secret_file(path: str | Path) -> str | None:
    """Read a secret file, refusing symlinks; None when missing/unreadable."""

    target = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(target, flags)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return None
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def write_new_secret_file(path: str | Path, secret: str) -> None:
    """Create a NEW secret file with 0600 at creation; never overwrite.

    Raises FileExistsError when the path already exists so concurrent
    provisioning cannot silently replace the owner key.
    """

    target = Path(path)
    ensure_private_dir(target.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secret)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
    except BaseException:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    try:
        os.chmod(target, 0o600)
    except OSError:  # pragma: no cover - Windows ACLs
        pass


def overwrite_secret_file_atomic(path: str | Path, secret: str) -> None:
    """Atomically replace a secret file (explicit regeneration only).

    Writes a 0600 temp file in the same directory then publishes with
    os.replace so readers never see a partial file or a 0644 window.
    """

    target = Path(path)
    ensure_private_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=".tmp-secret-"
    )
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError:  # pragma: no cover
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secret)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_name, target)
        try:
            os.chmod(target, 0o600)
        except OSError:  # pragma: no cover
            pass
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
