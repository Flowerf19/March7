"""Pure ACK durability proof (shared by coordinator + journal revalidation).

No Redis, no I/O. Single implementation: coordinator keeps a staticmethod
alias for backwards compat; journal parse + finish re-run this before any
destructive trim/release/clear. Exact batch coverage required (no subset).
"""
from __future__ import annotations


def ack_trim_ids(
    result: dict, shipped_ids: set[str] | None,
) -> tuple[list[str] | None, str | None]:
    """Trim IDs for a fully successful ack, else (None, skip reason)."""
    if not isinstance(result, dict):
        return None, "malformed_result"
    if result.get("status") != "ok":
        return None, f"status_{result.get('status')}"
    entry_ids = result.get("entry_ids")
    valid_ids = (
        isinstance(entry_ids, list)
        and len(entry_ids) > 0
        and all(isinstance(x, str) and x.strip() for x in entry_ids)
    )
    if not valid_ids:
        return None, "no_entry_ids"
    topics_failed = result.get("topics_failed")
    if type(topics_failed) is not int or topics_failed < 0:
        return None, "malformed_counts"
    if (
        topics_failed > 0
        or bool(result.get("profile_failed"))
        or bool(result.get("profile_conflict"))
        or bool(result.get("conflict"))
    ):
        return None, "unacknowledged_failures"
    if shipped_ids is not None and any(i not in shipped_ids for i in entry_ids):
        return None, "foreign_entry_ids"
    if len(entry_ids) != len(set(entry_ids)):
        return None, "partial_ack"
    if shipped_ids is not None and set(entry_ids) != set(shipped_ids):
        return None, "partial_ack"
    meaningful = result.get("has_meaningful_content")
    topics_stored = result.get("topics_stored")
    if meaningful is True:
        if type(topics_stored) is not int or topics_stored <= 0:
            return None, "no_meaningful_writes"
        return list(entry_ids), None
    if meaningful is False:
        if type(topics_stored) is not int or topics_stored != 0:
            return None, "contradictory_noise_ack"
        return list(entry_ids), None
    return None, "missing_meaningful_flag"
