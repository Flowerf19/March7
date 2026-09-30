"""Unit tests for the durable approval ledger (private tmp paths only)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from system_gateway.approval_ledger import (
    ApprovalLedger,
    LedgerUnavailable,
    approval_key_fingerprint,
)

AFP = approval_key_fingerprint("owner-key-for-ledger-tests")


def test_claim_then_replay_denied(tmp_path: Path):
    ledger = ApprovalLedger(tmp_path / "ledger.db")
    assert ledger.claim(nonce="n1", expiry=9_999_999_999, key_fp=AFP) is True
    assert ledger.claim(nonce="n1", expiry=9_999_999_999, key_fp=AFP) is False


def test_two_stores_same_file_only_one_wins(tmp_path: Path):
    path = tmp_path / "shared.db"
    first = ApprovalLedger(path)
    second = ApprovalLedger(path)
    results = sorted(
        [
            first.claim(nonce="race-nonce", expiry=9_999_999_999, key_fp=AFP),
            second.claim(nonce="race-nonce", expiry=9_999_999_999, key_fp=AFP),
        ]
    )
    assert results == [False, True]


def test_expired_rows_pruned_and_reclaimable(tmp_path: Path):
    import time

    ledger = ApprovalLedger(tmp_path / "ledger.db")
    now = int(time.time())
    # Expired claims fail closed and insert nothing.
    assert ledger.claim(nonce="old", expiry=now - 100, key_fp=AFP) is False
    assert ledger.prune_expired(now=now) == 0
    assert ledger.claim(nonce="old", expiry=now + 600, key_fp=AFP) is True
    # Valid row later expires, is prunable, then reclaimable.
    assert ledger.prune_expired(now=now + 601) >= 1
    assert ledger.claim(nonce="old", expiry=now + 1200, key_fp=AFP) is True


def test_claim_prunes_expired_opportunistically(tmp_path: Path):
    import time
    from unittest.mock import patch

    ledger = ApprovalLedger(tmp_path / "ledger.db")
    now = int(time.time())
    # Expired direct claim fails closed, inserts nothing.
    assert ledger.claim(nonce="stale", expiry=now - 5, key_fp=AFP) is False
    with patch(
        "system_gateway.approval_ledger.time.time", return_value=float(now)
    ):
        assert ledger.claim(nonce="short", expiry=now + 10, key_fp=AFP) is True
    with patch(
        "system_gateway.approval_ledger.time.time", return_value=float(now + 20)
    ):
        assert ledger.claim(nonce="fresh", expiry=now + 600, key_fp=AFP) is True
        # Short row expired and was pruned by the fresh claim.
        assert ledger.claim(nonce="short", expiry=now + 600, key_fp=AFP) is True
    assert ledger.claim(nonce="stale", expiry=now + 600, key_fp=AFP) is True


def test_blocked_path_raises_fail_closed(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not-a-dir", encoding="utf-8")
    with pytest.raises(LedgerUnavailable):
        ApprovalLedger(blocker / "ledger.db")


def test_directory_path_raises_fail_closed(tmp_path: Path):
    target = tmp_path / "as-dir"
    target.mkdir()
    with pytest.raises(LedgerUnavailable):
        ApprovalLedger(target)


def test_explicit_memory_ledger_isolated():
    ledger = ApprovalLedger(":memory:")
    assert ledger.claim(nonce="m1", expiry=9_999_999_999, key_fp=AFP) is True
    assert ledger.claim(nonce="m1", expiry=9_999_999_999, key_fp=AFP) is False


def test_ledger_file_created_private(tmp_path: Path):
    path = tmp_path / "sub" / "ledger.db"
    ApprovalLedger(path)
    assert path.exists()
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert oct(path.stat().st_mode)[-3:] == "600"
        assert oct(path.parent.stat().st_mode)[-3:] == "700"


def test_no_token_stored_in_db(tmp_path: Path):
    import sqlite3

    path = tmp_path / "ledger.db"
    ledger = ApprovalLedger(path)
    assert ledger.claim(nonce="abc123", expiry=9_999_999_999, key_fp=AFP) is True
    conn = sqlite3.connect(str(path))
    try:
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(consumed_approvals)").fetchall()
        }
        assert "nonce" in cols and "expiry" in cols
        assert "token" not in cols and "payload" not in cols and "approval_id" not in cols
    finally:
        conn.close()
