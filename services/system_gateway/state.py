"""Runtime state for the System Gateway service.

The native service holds shared state (uptime, durable approval ledger, seen
nonces, audit log buffer) here so handlers and tests can reach it without
singletons.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from twin.shared.system_gateway import NonceStore

from .approval_ledger import (
    ApprovalLedger,
    LedgerUnavailable,
    approval_key_fingerprint,
)

logger = logging.getLogger(__name__)

SERVICE_VERSION = "0.1.0"


@dataclass
class GatewayState:
    """Mutable runtime state owned by the gateway service."""

    started_at: float = field(default_factory=time.time)
    raw_shell_enabled: bool = False
    shared_secret: Optional[str] = None
    approval_secret: Optional[str] = None
    # Durable single-use store. None fails closed (deny); there is no
    # unbounded in-memory consumed set and no silent memory fallback.
    approval_ledger: Optional[ApprovalLedger] = None
    nonce_store: NonceStore = field(default_factory=NonceStore)
    audit_log: list[dict[str, object]] = field(default_factory=list)

    def uptime_seconds(self) -> int:
        return int(time.time() - self.started_at)

    def record_audit(self, event: dict[str, object]) -> None:
        self.audit_log.append(event)
        if len(self.audit_log) > 500:
            self.audit_log = self.audit_log[-500:]
        # approval_id is a short-lived bearer token; keep it in the in-memory
        # audit_log buffer above but never write it to the log stream.
        logger.info(
            "system_gateway audit %s",
            {k: v for k, v in event.items() if k not in {"details", "approval_id"}},
        )

    def consume_approval(self, nonce: str | None, *, expiry: int | None = None) -> bool:
        """Atomically claim a single-use approval nonce in the durable ledger.

        Returns True if the nonce was unused (and is now durably consumed),
        False if it had already been consumed (replay). Raises
        LedgerUnavailable when the ledger is missing/unwritable so callers
        fail closed (deny, never execute). Scoped by approval-key
        fingerprint so key generations are distinguishable.
        """

        if not nonce or not isinstance(nonce, str):
            return False
        if self.approval_ledger is None:
            raise LedgerUnavailable("approval ledger unavailable")
        if not self.approval_secret:
            return False
        # Malformed/expired expiries fail closed; never synthesize now+TTL.
        # Ledger rechecks now < exp after acquiring the write lock.
        if isinstance(expiry, bool):
            return False
        try:
            exp = int(expiry) if expiry is not None else None
        except (TypeError, ValueError):
            return False
        if not exp or exp <= 0:
            return False
        key_fp = approval_key_fingerprint(self.approval_secret)
        return self.approval_ledger.claim(nonce=nonce, expiry=exp, key_fp=key_fp)
