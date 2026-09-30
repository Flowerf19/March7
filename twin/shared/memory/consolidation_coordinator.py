"""Consolidation coordination: dispatch + acknowledgement-gated T1 trim."""
from __future__ import annotations

import logging
from typing import Any, Callable

from twin.shared.memory.consolidation_ack import ack_trim_ids as _pure_ack_trim_ids
from twin.shared.memory.consolidation_journal import (
    CallerJournal,
    ReceiverJournal,
    batch_id_of_plan_key,
    select_caller_batch,
    validate_ack_generation,
)
from twin.shared.observability.langsmith import add_current_run_metadata

logger = logging.getLogger(__name__)


def _redis_from(getter: Callable[[], Any] | None) -> Any | None:
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


def normalize_role(role: str) -> str:
    if role == "assistant":
        return "assistant"
    if role == "system":
        return "system"
    return "user"


def entry_to_snapshot(entry: Any) -> dict:
    return {
        "entry_id": entry.entry_id,
        "message_id": entry.message_id,
        "role": entry.role,
        "content": entry.content,
        "author_id": entry.author_id,
        "author_name": entry.author_name,
        "timestamp": entry.created_at.isoformat(),
    }


class ConsolidationCoordinator:
    """Dispatch consolidation and trim T1 only on validated acknowledgement.

    The A2A client / local consolidator are read through providers on every
    call, so SharedMemoryManager's attributes stay the mutable source of
    truth (containers and tests reassign them after construction).
    """

    def __init__(
        self,
        t1: Any,
        get_client: Callable[[], Any],
        get_local: Callable[[], Any],
        *,
        get_caller_redis: Callable[[], Any] | None = None,
        get_receiver_redis: Callable[[], Any] | None = None,
    ) -> None:
        self._t1 = t1
        self._get_client = get_client
        self._get_local = get_local
        self._get_caller_redis = get_caller_redis
        self._get_receiver_redis = get_receiver_redis

    def _caller_journal(self) -> CallerJournal | None:
        return CallerJournal.from_redis(_redis_from(self._get_caller_redis))

    def _receiver_journal(self) -> ReceiverJournal | None:
        return ReceiverJournal.from_redis(_redis_from(self._get_receiver_redis))

    async def consolidate_scope(
        self,
        scope: str,
        scope_id: str,
        entries: list[dict] | None = None,
    ) -> dict:
        """Consolidate T1 messages, then trim on success.

        Uses the remote A2A client if configured (March7 → Evernight), else a
        local consolidator if set (Evernight worker), else fails.

        On the A2A path the caller's OWN T1 entries are shipped over the wire so
        Evernight consolidates this agent's messages (not its own T1, which is
        empty for this agent's scopes). ``entries`` may be supplied directly by a
        payload that already carries them; otherwise the caller journal pins
        the first fresh batch (pending-first, exact ID fetch on retry) and we
        ship that. A validated ACK is journaled before trim; trim, receiver
        release and pending clear are all retry-safe from the ACK record.
        """
        client = self._get_client()
        local = self._get_local()
        if client is None and local is None:
            logger.error("No consolidator configured (neither client nor local)")
            add_current_run_metadata({
                "entries_shipped": len(entries or []),
                "trimmed": 0,
                "trim_skipped_reason": "no_consolidator_configured",
            })
            return {"status": "failed", "scope": scope, "scope_id": scope_id, "error": "consolidator not configured"}
        if entries is not None and not isinstance(entries, list):
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": "invalid_entries"}
        selection = await select_caller_batch(
            t1=self._t1, journal=self._caller_journal(), scope=scope,
            scope_id=scope_id, explicit=entries, limit=200,
        )
        if selection.status == "empty":
            return {"status": "skipped", "reason": "no_messages",
                    "messages_summarized": 0, "entry_ids": []}
        if selection.status == "acknowledged":
            assert selection.pending is not None
            return await self._finish_acknowledged(scope, scope_id, selection.pending)
        if selection.status != "ready":
            add_current_run_metadata({
                "entries_shipped": 0, "trimmed": 0,
                "trim_skipped_reason": selection.reason or "selection_failed",
            })
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": selection.reason or "selection_failed",
                    "detail": selection.detail}
        ship = selection.entries
        if entries is None:
            ship = [entry_to_snapshot(e) for e in ship]
        if client is not None:
            logger.info(
                "Consolidating via A2A client scope=%s/%s entries=%d",
                scope, scope_id, len(ship),
            )
            result_dict = await client.consolidate_scope(
                scope=scope,
                scope_id=scope_id,
                reason="auto",
                entries=ship,
            )
        else:
            logger.info(
                "Consolidating locally scope=%s/%s entries=%d",
                scope, scope_id, len(ship),
            )
            result_dict = await local(
                scope=scope,
                scope_id=scope_id,
                reason="auto",
                entries=ship,
            )
        shipped_ids = {str(e.get("entry_id")) for e in ship if e.get("entry_id")}

        # Trim T1 by the EXACT entry_ids the consolidator acknowledged.
        # Only a fully successful ok (no failed topics/profile writes) with a
        # non-empty entry_ids list may trim; anything else keeps T1 for retry.
        trim_ids, trim_skipped_reason = self._ack_trim_ids(result_dict, shipped_ids)
        trimmed = 0
        if trim_ids is not None:
            if selection.pending is None:
                trim_skipped_reason = "foreign_entry_ids"
            else:
                return await self._complete_validated_ack(
                    scope, scope_id, selection.pending, trim_ids, result_dict,
                    shipped=len(ship),
                )
        if trim_skipped_reason == "no_entry_ids":
            logger.warning(
                "Consolidation ok but no entry_ids returned scope=%s/%s — skipping trim",
                scope, scope_id,
            )
        elif trim_skipped_reason == "unacknowledged_failures":
            logger.warning(
                "Consolidation ok with failures scope=%s/%s — skipping trim",
                scope, scope_id,
            )
        elif trim_skipped_reason in (
            "foreign_entry_ids",
            "partial_ack",
            "no_meaningful_writes",
            "malformed_result",
            "malformed_counts",
            "missing_meaningful_flag",
            "contradictory_noise_ack",
        ):
            logger.warning(
                "Consolidation ack rejected (%s) scope=%s/%s — skipping trim",
                trim_skipped_reason, scope, scope_id,
            )

        add_current_run_metadata({
            "entries_shipped": len(ship),
            "trimmed": trimmed,
            "trim_skipped_reason": trim_skipped_reason,
        })
        return result_dict

    async def _complete_validated_ack(self, scope: str, scope_id: str,
                                       pending: dict[str, Any], trim_ids: list[str],
                                       result: dict[str, Any], *, shipped: int) -> dict:
        """Journal the ACK, then trim + release + clear (each step retry-safe)."""
        pinned = set(pending.get("entry_ids") or [])
        valid, reason = self._ack_trim_ids(result, pinned)
        if valid is None or set(trim_ids) != set(valid) or set(valid) != pinned:
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": reason or "partial_ack",
                    "detail": "incomplete batch receipt"}
        ok_gen, gen_reason = validate_ack_generation(result)
        if not ok_gen:
            return {"status": "failed", "scope": scope, "scope_id": scope_id, "error": gen_reason or "bad_generation", "detail": "invalid generation"}
        receiver_pre = self._receiver_journal()
        if receiver_pre is None:
            return {"status": "failed", "scope": scope, "scope_id": scope_id, "error": "journal_unavailable"}
        try:
            ok_own, why_own = await receiver_pre.check_owners(scope, scope_id, list(pending.get("entry_ids") or []), batch_id_of_plan_key(str(pending.get("plan_key") or "")), str(result.get("receiver_generation") or ""))
        except Exception as exc:
            return {"status": "failed", "scope": scope, "scope_id": scope_id, "error": "journal_error", "detail": f"owner check failed: {exc}"}
        if not ok_own:
            return {"status": "failed", "scope": scope, "scope_id": scope_id, "error": "journal_error", "Detail": f"stale generation: {why_own}"}
        journal = self._caller_journal()
        if journal is None:
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": "journal_unavailable"}
        marked, stored = await journal.mark_acknowledged(
            scope, scope_id, pending, trim_ids, result)
        if marked not in ("acked", "already") or not isinstance(stored, dict):
            detail = stored if isinstance(stored, str) else "stored ACK unreadable"
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": "journal_error", "detail": detail}
        # Always finalize the stored winner (never loser's differing ACK).
        final = await self._finish_acknowledged(scope, scope_id, stored)
        winner_n = len(stored.get("trim_ids") or trim_ids)
        if final.get("status") == "ok":
            add_current_run_metadata({
                "entries_shipped": shipped, "trimmed": winner_n,
                "trim_skipped_reason": None,
            })
            logger.info("Trimmed T1 after consolidation: %d entries", winner_n)
        return final

    async def _finish_acknowledged(self, scope: str, scope_id: str,
                                   record: dict[str, Any]) -> dict:
        """Trim (once) + generation-bound release + nonce-bound clear."""
        trim_ids = list(record.get("trim_ids") or [])
        pending_ids = list(record.get("entry_ids") or [])
        plan_key = str(record.get("plan_key") or "")
        nonce = str(record.get("caller_nonce") or "")
        gen = str(record.get("receiver_generation") or "")
        stage = str(record.get("stage") or "")
        ack_raw = record.get("ack")
        if not isinstance(ack_raw, dict):
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": "journal_error", "detail": "stored ACK corrupt"}
        ack = dict(ack_raw)
        pending_set = set(pending_ids)
        if (
            not pending_ids or not trim_ids
            or any(type(i) is not str or not i.strip() for i in trim_ids)
            or len(trim_ids) != len(set(trim_ids)) or set(trim_ids) != pending_set
            or not plan_key or len(nonce) != 32
            or stage not in ("acknowledged", "trimmed")
        ):
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": "journal_error", "detail": "stored ACK coverage invalid"}
        meaningful = ack.get("has_meaningful_content")
        if (meaningful is True and len(gen) != 32) or (meaningful is False and gen != "") or meaningful not in (True, False):
            return {"status": "failed", "scope": scope, "scope_id": scope_id, "error": "journal_error", "detail": "bad generation"}
        valid, reason = _pure_ack_trim_ids(ack, pending_set)
        if valid is None or set(valid) != set(trim_ids):
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": "journal_error",
                    "detail": f"stored ACK proof invalid: {reason}"}
        journal = self._caller_journal()
        receiver = self._receiver_journal()
        if journal is None or receiver is None:
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": "journal_unavailable", "ack_durable": True}
        if stage != "trimmed":
            ok, _why = await receiver.check_owners(scope, scope_id, pending_ids, batch_id_of_plan_key(plan_key), gen)
            if not ok:
                return {"status": "failed", "scope": scope, "scope_id": scope_id, "error": "journal_error", "Detail": "stale generation"}
            try:
                guarded = self._t1.trim_consolidated_batch
            except AttributeError:
                return {"status": "failed", "scope": scope, "scope_id": scope_id,
                        "error": "journal_error", "detail": "T1 lacks guarded trim"}
            try:
                gout = await guarded(scope, scope_id, record)
            except Exception as exc:
                logger.warning("consolidation trim failed, keeping ACK for retry: %s", exc)
                return {"status": "failed", "scope": scope, "scope_id": scope_id,
                        "error": "trim_failed", "ack_durable": True, "detail": str(exc)}
            if not isinstance(gout, dict) or gout.get("status") not in ("trimmed", "already"):
                why = gout.get("reason") if isinstance(gout, dict) else None
                return {"status": "failed", "scope": scope, "scope_id": scope_id,
                        "error": "journal_error", "ack_durable": True,
                        "detail": f"T1 changed, stale: {why}"}
        released = await receiver.release(
            scope, scope_id, pending_ids, batch_id_of_plan_key(plan_key),
            plan_key, gen)
        if not released.ok:
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": "release_failed", "ack_durable": True,
                    "detail": released.detail}
        cleared, detail = await journal.clear(scope, scope_id, pending_ids, nonce)
        if cleared != "cleared":
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": "journal_error", "ack_durable": True, "detail": detail}
        return ack if ack.get("status") == "ok" else {
            "status": "failed", "scope": scope, "scope_id": scope_id,
            "error": "journal_error", "detail": "stored ACK not ok",
        }

    async def consolidate_snapshot(
        self,
        user_id: str,
        snapshot: list[dict],
        reason: str = "manual",
    ) -> bool:
        del reason
        for item in snapshot or []:
            role = normalize_role(str(item.get("role") or "user"))
            content = str(item.get("content") or "")
            if not content.strip():
                continue
            await self._t1.observe(
                "user",
                str(user_id),
                role,
                content,
                author_id=str(user_id) if role == "user" else None,
                author_name=str(user_id) if role == "user" else None,
                message_id=str(item.get("message_id") or item.get("id") or "") or None,
            )
        result = await self.consolidate_scope("user", str(user_id))
        return result.get("status") in {"ok", "skipped"}

    @staticmethod
    def _ack_trim_ids(
        result: dict, shipped_ids: set[str] | None,
    ) -> tuple[list[str] | None, str | None]:
        """Alias to pure ACK proof (single impl in consolidation_ack)."""
        return _pure_ack_trim_ids(result, shipped_ids)
