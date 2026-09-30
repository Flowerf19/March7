"""User-facing formatting for T2 recall results (pure functions, no I/O).

Owns the shared dict-or-object field accessor plus the numbered memory
list, per-hit metadata, and the VN "N ngày trước — DD/MM" date display.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from twin.shared.memory.vn_time import VN_TZ, vn_now


def memory_field(memory: Any, field_name: str, default: Any = None) -> Any:
    """Read a field off a recall hit (dict from the store, or a legacy
    object shape). Shared here so ranking/format/trace agree."""
    if isinstance(memory, dict):
        return memory.get(field_name, default)
    return getattr(memory, field_name, default)


def days_ago_display(memory: Any, *, now: Optional[datetime] = None) -> Optional[str]:
    """P3.4: "<N> ngày trước — DD/MM" in VN tz from `period_end`
    (fallback `created_at`). None when neither field is usable."""
    ts = memory_field(memory, "period_end")
    if ts is None:
        ts = memory_field(memory, "created_at")
    try:
        if isinstance(ts, datetime):
            ts = ts.timestamp()
        ts = float(ts)  # type: ignore[arg-type]
        dt_vn = datetime.fromtimestamp(ts, tz=VN_TZ)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    now = now or vn_now()
    delta_days = (now.date() - dt_vn.date()).days
    when = "hôm nay" if delta_days <= 0 else f"{delta_days} ngày trước"
    return f"{when} — {dt_vn.strftime('%d/%m')}"


def format_memories(
    memories: Any, *, widen_label: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    """Render the numbered recall list (or the empty/pass-through cases)."""
    if isinstance(memories, str):
        return memories
    if not memories:
        return "Không tìm thấy ký ức phù hợp."

    header = f"Tìm thấy {len(memories)} ký ức:"
    if widen_label:
        header += f" {widen_label}"
    now = now or vn_now()
    lines = [header]
    for index, memory in enumerate(memories, start=1):
        content = (
            memory_field(memory, "summary")
            or memory_field(memory, "content")
            or str(memory)
        )
        content = str(content).strip()
        topic_val = memory_field(memory, "topic")
        topic_display = memory_field(memory, "topic_display")
        label = topic_display or topic_val
        date_display = days_ago_display(memory, now=now)

        line = str(index) + "."
        if label:
            line += f" [{label}]"
        if date_display:
            line += f" ({date_display})"
        line += f" {content}"
        lines.append(line)

        metadata = memory_metadata(memory)
        if metadata:
            lines.append(f"   ({'; '.join(metadata)})")
    return "\n".join(lines)


def memory_metadata(memory: Any) -> list[str]:
    """Per-hit metadata lines (relevance, ids, timestamps)."""
    metadata: list[str] = []
    # KNN `score` is COSINE DISTANCE (1 - similarity) — surface it to the
    # LLM as a 0-1 relevance so weak hits can be treated with caution.
    score = memory_field(memory, "score")
    if score is not None:
        try:
            metadata.append(f"relevance={1.0 - float(score):.2f}")
        except (TypeError, ValueError):
            pass
    elif memory_field(memory, "_score") is not None:
        metadata.append("match=bm25")
    for field_name in ("memory_id", "summary_id", "speaker"):
        value = memory_field(memory, field_name)
        if value:
            metadata.append(f"{field_name}={value}")

    catalogs = memory_field(memory, "catalogs")
    if catalogs:
        metadata.append(f"catalogs={','.join(catalogs)}")

    created_at = memory_field(memory, "created_at")
    if isinstance(created_at, datetime):
        metadata.append(f"created_at={created_at.isoformat()}")
    elif created_at:
        try:
            ts = float(created_at)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            metadata.append(f"created_at={dt.isoformat()}")
        except (TypeError, ValueError, OverflowError, OSError):
            metadata.append(f"created_at={created_at}")

    return metadata
