"""Durable single-use approval ledger (stdlib SQLite).

Stores ONLY nonce + expiry + approval-key fingerprint + authority; never the
raw token or payload. Claim is an atomic INSERT under a PRIMARY KEY so two
processes/instances racing the same nonce yield exactly one winner. Expired
rows are pruned on claim to bound size. All I/O uses a bounded timeout and
raises LedgerUnavailable so callers fail closed (deny, never execute).
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path

LEDGER_TIMEOUT_SECONDS = 5.0
MEMORY_LEDGER = ":memory:"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consumed_approvals (
  nonce TEXT PRIMARY KEY,
  expiry INTEGER NOT NULL,
  key_fp TEXT NOT NULL,
  authority TEXT NOT NULL DEFAULT 'owner',
  claimed_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consumed_expiry ON consumed_approvals(expiry);
"""


class LedgerUnavailable(OSError):
    """Raised when the durable ledger cannot be read/written (fail closed)."""


def approval_key_fingerprint(secret: str) -> str:
    """Non-reversible scope for the approval-key generation (hex digest)."""

    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class ApprovalLedger:
    """SQLite-backed single-use claim store. ``:memory:`` is test-only."""

    def __init__(
        self,
        path: str | Path,
        *,
        timeout: float = LEDGER_TIMEOUT_SECONDS,
        authority: str = "owner",
    ) -> None:
        self._memory = str(path) == MEMORY_LEDGER
        self._path = Path(str(path)) if not self._memory else None
        self._timeout = timeout
        self._authority = authority
        if self._memory:
            self._mem_conn = sqlite3.connect(
                MEMORY_LEDGER, timeout=timeout, check_same_thread=False
            )
            with self._mem_conn:
                self._mem_conn.executescript(_SCHEMA)
            return
        assert self._path is not None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self._path.parent, 0o700)
            except OSError:
                pass
            if not self._path.exists():
                # Pre-create 0600 so SQLite never inherits a lax umask mode.
                try:
                    fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    os.close(fd)
                except FileExistsError:
                    # Lost initial-create race; existing DB is legitimate.
                    pass
            conn = self._connect()
            try:
                with conn:
                    conn.executescript(_SCHEMA)
            finally:
                conn.close()
        except OSError as exc:
            raise LedgerUnavailable("approval ledger unavailable") from exc
        except sqlite3.Error as exc:
            raise LedgerUnavailable("approval ledger unavailable") from exc

    @property
    def path(self) -> Path | str:
        return MEMORY_LEDGER if self._memory else self._path  # type: ignore[return-value]

    def _connect(self) -> sqlite3.Connection:
        if self._memory:
            return self._mem_conn
        assert self._path is not None
        conn = sqlite3.connect(str(self._path), timeout=self._timeout)
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)}")
        except sqlite3.Error:
            pass
        return conn

    def claim(self, *, nonce: str, expiry: int, key_fp: str) -> bool:
        """Atomically claim a nonce; False means consumed, expired, or invalid.

        Expired/malformed inputs fail closed (no insert). Validity is rechecked
        after acquiring the write lock so busy-wait crossing expiry denies.
        Raises LedgerUnavailable on any I/O failure so callers fail closed.
        """

        if not nonce or not isinstance(nonce, str):
            return False
        if not key_fp or not isinstance(key_fp, str):
            return False
        if isinstance(expiry, bool):
            return False
        try:
            exp = int(expiry)
        except (TypeError, ValueError):
            return False
        if exp <= 0:
            return False
        # Float compare: valid iff now < exp, consistent with prune/verify.
        if time.time() >= exp:
            return False
        try:
            conn = self._connect()
        except (OSError, sqlite3.Error) as exc:
            raise LedgerUnavailable("approval ledger unavailable") from exc
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise LedgerUnavailable("approval ledger unavailable") from exc
            try:
                now_locked = time.time()
                if now_locked >= exp:
                    conn.execute("ROLLBACK")
                    return False
                claimed_at = int(now_locked)
                try:
                    conn.execute(
                        "INSERT INTO consumed_approvals"
                        " (nonce, expiry, key_fp, authority, claimed_at)"
                        " VALUES (?,?,?,?,?)",
                        (nonce, exp, key_fp, self._authority, claimed_at),
                    )
                except sqlite3.IntegrityError:
                    conn.execute("ROLLBACK")
                    return False
                # Prune others only; winner is never deleted in its own txn.
                conn.execute(
                    "DELETE FROM consumed_approvals"
                    " WHERE expiry <= ? AND nonce != ?",
                    (int(now_locked), nonce),
                )
                conn.execute("COMMIT")
                return True
            except LedgerUnavailable:
                raise
            except sqlite3.IntegrityError:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                return False
            except (OSError, sqlite3.Error) as exc:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise LedgerUnavailable("approval ledger unavailable") from exc
        finally:
            if not self._memory:
                try:
                    conn.close()
                except (OSError, sqlite3.Error):
                    pass

    def prune_expired(self, *, now: float | None = None) -> int:
        """Delete expired rows; returns the number removed."""

        cutoff = int(now if now is not None else time.time())
        try:
            conn = self._connect()
            try:
                if self._memory:
                    with self._mem_conn:
                        cur = self._mem_conn.execute(
                            "DELETE FROM consumed_approvals WHERE expiry <= ?",
                            (cutoff,),
                        )
                        return cur.rowcount if cur.rowcount > 0 else 0
                with conn:
                    cur = conn.execute(
                        "DELETE FROM consumed_approvals WHERE expiry <= ?",
                        (cutoff,),
                    )
                    return cur.rowcount if cur.rowcount > 0 else 0
            finally:
                if not self._memory:
                    conn.close()
        except (OSError, sqlite3.Error) as exc:
            raise LedgerUnavailable("approval ledger unavailable") from exc
