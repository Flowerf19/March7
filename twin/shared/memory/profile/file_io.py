"""Profile (T3) file I/O: paths, cross-process locks, atomic writes.

Security: T3 dir is shared-writable, so symlinks/FIFOs planted inside must
never redirect reads/writes/locks outside. All opens use O_NOFOLLOW with a
pinned dir_fd plus fstat S_ISREG/S_ISDIR checks; no symlink-following fallback.
"""
from __future__ import annotations

import asyncio
import errno
import fcntl
import logging
import os
import secrets
import stat
from contextlib import asynccontextmanager
from pathlib import Path

from twin.shared.memory.profile.codec import default_skeleton, sanitize_user_id
from twin.shared.memory.profile.constants import DEFAULT_PROFILE_DIR

logger = logging.getLogger(__name__)

_READ_CHUNK = 65536
_TEMP_RETRIES = 5


def _split_parent_name(path: Path) -> tuple[Path, str]:
    name = path.name
    if not name or name in (".", "..") or "/" in name or "\x00" in name:
        raise OSError(errno.EINVAL, "invalid profile filename")
    return path.parent, name


def _open_dir_fd(dir_path: Path | str) -> int:
    fd = os.open(
        str(dir_path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.ENOTDIR, "not a directory")
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return fd


def _create_temp_exclusive(dir_fd: int, target_name: str) -> tuple[int, str]:
    last: OSError | None = None
    for _ in range(_TEMP_RETRIES):
        tmp_name = f".{target_name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        try:
            fd = os.open(
                tmp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=dir_fd,
            )
        except FileExistsError as exc:
            last = exc  # type: ignore[assignment]
            continue
        except OSError as exc:
            if exc.errno == errno.ELOOP:  # colliding symlink, retry random name
                last = exc
                continue
            raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError(errno.EINVAL, "temp is not a regular file")
            os.fchmod(fd, 0o600)
        except OSError:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except OSError:
                pass
            raise
        return fd, tmp_name
    assert last is not None
    raise last


class ProfileFileIO:
    """Per-user markdown file I/O with in-process + cross-process locks."""

    def __init__(
        self,
        base_path: str = DEFAULT_PROFILE_DIR,
        *,
        enable_file_lock: bool = True,
        file_lock_poll_seconds: float = 0.05,
    ) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._base_path / ".locks"
        self._lock_path.mkdir(parents=True, exist_ok=True)
        # Fail fast if attacker pre-planted symlink/file; runtime pinning is authoritative.
        base_fd = _open_dir_fd(self._base_path)
        try:
            locks_fd = os.open(
                ".locks",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=base_fd,
            )
        finally:
            try:
                os.close(base_fd)
            except OSError:
                pass
        try:
            if not stat.S_ISDIR(os.fstat(locks_fd).st_mode):
                raise OSError(errno.ENOTDIR, "locks path is not a directory")
        finally:
            try:
                os.close(locks_fd)
            except OSError:
                pass
        self._enable_file_lock = enable_file_lock
        self._file_lock_poll_seconds = file_lock_poll_seconds
        self._locks: dict[str, asyncio.Lock] = {}

    def path_for(self, user_id: str) -> Path:
        return self._base_path / f"{sanitize_user_id(user_id)}.md"

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        cleaned = sanitize_user_id(user_id)
        lock = self._locks.get(cleaned)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[cleaned] = lock
        return lock

    def _file_lock_path_for(self, user_id: str) -> Path:
        return self._lock_path / f"{sanitize_user_id(user_id)}.lock"

    @asynccontextmanager
    async def locked(self, user_id: str):
        """Per-user lock shared by March7/Evernight processes."""
        async_lock = self._lock_for(user_id)
        async with async_lock:
            if not self._enable_file_lock:
                yield
                return
            cleaned = sanitize_user_id(user_id)
            lock_name = f"{cleaned}.lock"
            base_fd = _open_dir_fd(self._base_path)
            try:
                try:
                    locks_fd = os.open(
                        ".locks",
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=base_fd,
                    )
                except FileNotFoundError:
                    try:
                        os.mkdir(".locks", 0o700, dir_fd=base_fd)
                    except FileExistsError:
                        pass
                    locks_fd = os.open(
                        ".locks",
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=base_fd,
                    )
                if not stat.S_ISDIR(os.fstat(locks_fd).st_mode):
                    try:
                        os.close(locks_fd)
                    except OSError:
                        pass
                    raise OSError(errno.ENOTDIR, "locks path is not a directory")
            finally:
                try:
                    os.close(base_fd)
                except OSError:
                    pass
            try:
                # O_NOFOLLOW: symlink lock is rejected, never followed.
                lock_fd = os.open(
                    lock_name,
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                    0o600,
                    dir_fd=locks_fd,
                )
            except OSError:
                try:
                    os.close(locks_fd)
                except OSError:
                    pass
                raise
            try:
                if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                    raise OSError(errno.EINVAL, "lock file is not a regular file")
                while True:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        await asyncio.sleep(self._file_lock_poll_seconds)
                try:
                    yield
                finally:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
            finally:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
                try:
                    os.close(locks_fd)
                except OSError:
                    pass

    @staticmethod
    def read_text_sync(path: Path) -> str | None:
        parent, name = _split_parent_name(path)
        try:
            dir_fd = _open_dir_fd(parent)
        except FileNotFoundError:
            return None
        try:
            try:
                # O_NONBLOCK avoids FIFO hang; fstat below rejects non-regular.
                fd = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                    dir_fd=dir_fd,
                )
            except FileNotFoundError:
                return None
        finally:
            try:
                os.close(dir_fd)
            except OSError:
                pass
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError(errno.EINVAL, "profile is not a regular file")
            chunks: list[bytes] = []
            while True:
                data = os.read(fd, _READ_CHUNK)
                if not data:
                    break
                chunks.append(data)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            return b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OSError(errno.EINVAL, "profile is not valid utf-8") from exc

    @staticmethod
    def atomic_write_sync(path: Path, content: str) -> None:
        if not isinstance(content, str):
            raise TypeError("content must be str")
        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("content is not encodable") from exc
        parent, name = _split_parent_name(path)
        dir_fd = _open_dir_fd(parent)
        try:
            fd, tmp_name = _create_temp_exclusive(dir_fd, name)
            try:
                view = memoryview(data)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
                os.fsync(fd)
            except OSError:
                try:
                    os.close(fd)
                except OSError:
                    pass
                try:
                    os.unlink(tmp_name, dir_fd=dir_fd)
                except OSError:
                    pass
                raise
            try:
                os.close(fd)
            except OSError:
                try:
                    os.unlink(tmp_name, dir_fd=dir_fd)
                except OSError:
                    pass
                raise
            try:
                # Pinned rename replaces a symlink itself, never follows it.
                os.replace(tmp_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            except OSError:
                try:
                    os.unlink(tmp_name, dir_fd=dir_fd)
                except OSError:
                    pass
                raise
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
        finally:
            try:
                os.close(dir_fd)
            except OSError:
                pass

    async def ensure_file(self, path: Path) -> str:
        """Return current text; create default skeleton if missing."""
        text = self.read_text_sync(path)
        if text is not None:
            return text
        skeleton = default_skeleton()
        try:
            self.atomic_write_sync(path, skeleton)
        except OSError as exc:
            logger.warning("profile skeleton write failed for %s: %s", path, exc)
            raise
        return skeleton
