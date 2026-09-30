"""Harrier token budgets (ported from another-brain).

Locked product contract, counted with the real Harrier tokenizer:
- document payload (with special tokens): at most 256
- prompted query (with special tokens): at most 128

another-brain *rejects* over-budget input. March7 T2 summaries can reach
T2_MERGE_MAX_CHARS (1500 chars) and the Harrier graph itself handles long
context, so here over-budget input only logs a warning and still embeds —
same validator, warn-only policy.
"""
from __future__ import annotations

BUDGET_DOCUMENT_TOKENS = 256
BUDGET_QUERY_TOKENS = 128


def check_budget(*, tokens: int, is_query: bool) -> str | None:
    """Return a warning message when over budget, else None."""
    limit = BUDGET_QUERY_TOKENS if is_query else BUDGET_DOCUMENT_TOKENS
    if tokens > limit:
        kind = "query" if is_query else "document"
        return (
            f"harrier {kind} uses {tokens} tokens, budget is {limit} "
            "(embedding anyway; consider shortening)"
        )
    return None
