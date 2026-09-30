"""Consolidation LLM-output validation policy (fail-closed)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from twin.shared.memory.profile import SECTIONS

_TOPIC_RE = re.compile(r"^[a-z0-9_]{1,64}$")


@dataclass(frozen=True, slots=True)
class TopicPlan:
    topic: str
    topic_display: str
    summary: str
    importance: int


@dataclass(frozen=True, slots=True)
class ConsolidationPlan:
    has_meaningful: bool
    topics: tuple[TopicPlan, ...] = ()
    profile_updates: dict[str, list[str]] = field(default_factory=dict)
    profile_rewrites: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlanError:
    reason: str
    detail: str


def _normalize_topic(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    slug = raw.strip().lower().replace(" ", "_").replace("-", "_")
    slug = re.sub(r"__+", "_", slug).strip("_")
    if not slug or not _TOPIC_RE.match(slug):
        return None
    return slug


def _check_bullet_text(value: object) -> str | None:
    """Return stripped bullet or None when invalid for LLM contract."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if "\n" in text or "\r" in text:
        return None
    if text.startswith("- "):
        return None
    for ch in text:
        o = ord(ch)
        if o == 0x09:
            continue
        if o < 0x20 or o == 0x7F:
            return None
    return text


def _validate_profile_map(
    raw: object, *, field_name: str, allow_empty_lists: bool,
) -> tuple[dict[str, list[str]] | None, PlanError | None]:
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return None, PlanError("invalid_schema", f"{field_name} must be an object")
    cleaned: dict[str, list[str]] = {}
    for section, bullets in raw.items():
        if section not in SECTIONS:
            return None, PlanError("invalid_schema", f"{field_name}.{section}: invalid section")
        if not isinstance(bullets, list):
            return None, PlanError("invalid_schema", f"{field_name}.{section}: must be a list")
        if not bullets:
            if allow_empty_lists:
                continue
            cleaned[section] = []
            continue
        items: list[str] = []
        for item in bullets:
            text = _check_bullet_text(item)
            if text is None:
                return None, PlanError("invalid_schema", f"{field_name}.{section}: invalid bullet")
            items.append(text)
        if items:
            cleaned[section] = items
    return cleaned, None


class ConsolidationPlanValidator:
    """Strict validator: exact bool, complete topics, consistent noise."""

    @staticmethod
    def validate(data: object) -> tuple[ConsolidationPlan | None, PlanError | None]:
        if not isinstance(data, dict):
            return None, PlanError("invalid_schema", "top-level JSON must be an object")
        if "has_meaningful_content" not in data:
            return None, PlanError("incomplete_content", "missing has_meaningful_content")
        flag = data.get("has_meaningful_content")
        if type(flag) is not bool:
            return None, PlanError("invalid_schema", "has_meaningful_content must be exactly true/false")
        has_meaningful: bool = flag

        raw_topics = data.get("topics", [])
        if not isinstance(raw_topics, list):
            return None, PlanError("invalid_schema", "topics must be a list")

        updates, err = _validate_profile_map(
            data.get("profile_updates", {}), field_name="profile_updates", allow_empty_lists=True,
        )
        if err is not None:
            return None, err
        rewrites, err = _validate_profile_map(
            data.get("profile_rewrites", {}), field_name="profile_rewrites", allow_empty_lists=True,
        )
        if err is not None:
            return None, err
        assert updates is not None and rewrites is not None

        if not has_meaningful:
            if raw_topics:
                return None, PlanError("inconsistent_noise", "noise must have empty topics")
            legacy = data.get("timeline_summary")
            if isinstance(legacy, str) and legacy.strip():
                return None, PlanError("inconsistent_noise", "noise must not carry timeline_summary")
            if updates or rewrites:
                return None, PlanError("inconsistent_noise", "noise must have empty profile writes")
            return ConsolidationPlan(False, (), {}, {}), None

        topics_raw: list[object] = list(raw_topics)
        if not topics_raw and isinstance(data.get("timeline_summary"), str):
            legacy_summary = str(data.get("timeline_summary") or "").strip()
            if legacy_summary:
                importance = data.get("importance", 3)
                if type(importance) is not int or not 1 <= importance <= 5:
                    return None, PlanError("invalid_topic", "legacy importance must be int 1-5")
                topics_raw = [{
                    "topic": "general",
                    "topic_display": "Tổng hợp",
                    "summary": legacy_summary,
                    "importance": importance,
                }]
        if not topics_raw:
            return None, PlanError("incomplete_content", "meaningful content requires non-empty topics")

        topics: list[TopicPlan] = []
        seen: set[str] = set()
        for idx, item in enumerate(topics_raw):
            label = f"topics[{idx}]"
            if not isinstance(item, dict):
                return None, PlanError("invalid_topic", f"{label} must be an object")
            slug = _normalize_topic(item.get("topic"))
            if slug is None:
                return None, PlanError("invalid_topic", f"{label}.topic invalid slug")
            if slug in seen:
                return None, PlanError("invalid_topic", f"{label}.topic duplicate: {slug}")
            seen.add(slug)
            display_raw = item.get("topic_display", "")
            if display_raw is None:
                display_raw = ""
            if not isinstance(display_raw, str):
                return None, PlanError("invalid_topic", f"{label}.topic_display must be a string")
            display = display_raw.strip()
            summary_raw = item.get("summary")
            if not isinstance(summary_raw, str) or not summary_raw.strip():
                return None, PlanError("invalid_topic", f"{label}.summary must be non-empty")
            importance = item.get("importance")
            if type(importance) is not int or not 1 <= importance <= 5:
                return None, PlanError("invalid_topic", f"{label}.importance must be int 1-5")
            topics.append(TopicPlan(slug, display, summary_raw.strip(), importance))

        return ConsolidationPlan(True, tuple(topics), updates, rewrites), None
