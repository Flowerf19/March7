"""Profile (T3) codec: parse/render/hash/validate markdown bullets.

Pure policy helpers extracted from MarkdownProfileStore so the store
orchestrator stays under the class-size cap. No file I/O here.
"""
from __future__ import annotations

import hashlib
import re

from twin.shared.memory.profile.constants import (
    EMPTY_PLACEHOLDER,
    SECTION_HEADERS,
    SECTIONS,
)

_USER_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_HEADER_TO_SECTION = {header: key for key, header in SECTION_HEADERS.items()}


def sanitize_user_id(user_id: str) -> str:
    """Validate user_id is filename-safe; return stripped value or raise."""
    if not isinstance(user_id, str):
        raise ValueError("user_id must be a string")
    cleaned = user_id.strip()
    if not cleaned:
        raise ValueError("user_id must not be empty")
    if not _USER_ID_RE.match(cleaned):
        raise ValueError(
            f"invalid user_id {user_id!r}: only [A-Za-z0-9_-] allowed"
        )
    return cleaned


def empty_section_map() -> dict[str, list[str]]:
    """Fresh map: section_key -> [] (empty list = render placeholder)."""
    return {key: [] for key in SECTIONS}


def parse_markdown(text: str) -> dict[str, list[str]]:
    """Parse markdown into {section_key: [bullet, ...]} (placeholder dropped)."""
    sections = empty_section_map()
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            header = line[3:].strip()
            current = _HEADER_TO_SECTION.get(header)
            continue
        if current is None:
            continue
        if not line.startswith("- "):
            continue
        bullet = line[2:].strip()
        if not bullet:
            continue
        if bullet == EMPTY_PLACEHOLDER:
            continue
        sections[current].append(bullet)
    return sections


def render_markdown(sections: dict[str, list[str]]) -> str:
    """Render canonical markdown: 8 headers in SECTIONS order."""
    parts: list[str] = []
    for idx, key in enumerate(SECTIONS):
        parts.append(f"## {SECTION_HEADERS[key]}")
        bullets = sections.get(key) or []
        if bullets:
            for b in bullets:
                parts.append(f"- {b}")
        else:
            parts.append(f"- {EMPTY_PLACEHOLDER}")
        if idx != len(SECTIONS) - 1:
            parts.append("")
    return "\n".join(parts) + "\n"


def profile_hash(text: str) -> str:
    """SHA256 of the raw profile markdown."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def default_skeleton() -> str:
    return render_markdown(empty_section_map())


def _check_single_line(value: str, label: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be a single line")
    # Reject other C0 controls + DEL; allow TAB (0x09) inside text.
    for ch in value:
        o = ord(ch)
        if o == 0x09:
            continue
        if (o < 0x20) or o == 0x7F:
            raise ValueError(f"{label} must not contain control characters")


def sanitize_bullets(bullets: list[str]) -> list[str]:
    """Validate replacement bullets; return stripped copies or raise."""
    if not isinstance(bullets, list):
        raise ValueError("bullets must be a list")
    cleaned: list[str] = []
    for idx, bullet in enumerate(bullets):
        label = f"bullet #{idx + 1}"
        if not isinstance(bullet, str):
            raise ValueError(f"{label} must be a string")
        value = bullet.strip()
        if not value:
            raise ValueError(f"{label} must not be empty")
        _check_single_line(value, label)
        if value.startswith("- "):
            raise ValueError(f"{label} must not include '- ' prefix")
        cleaned.append(value)
    return cleaned


def clean_append_bullet(content: object) -> str | None:
    """Validate one append bullet.

    Returns stripped bullet, None for empty (caller skips with False),
    raises ValueError on non-string/multiline/control/"- " prefix to match
    replacement validation and avoid lossy parser truncation.
    """
    if content is None:
        return None
    if not isinstance(content, str):
        raise ValueError("bullet must be a string")
    value = content.strip()
    if not value:
        return None
    _check_single_line(value, "bullet")
    if value.startswith("- "):
        raise ValueError("bullet must not include '- ' prefix")
    return value


# Back-compat aliases for the previous private helpers.
_sanitize_user_id = sanitize_user_id
_empty_section_map = empty_section_map
_parse_markdown = parse_markdown
_render_markdown = render_markdown
_default_skeleton = default_skeleton
_sanitize_bullets = sanitize_bullets
