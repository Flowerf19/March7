"""T3 markdown profile store: per-user `.md` file with 8 fixed sections."""
from __future__ import annotations

import logging
from typing import Any

from twin.shared.memory.profile.codec import (
    clean_append_bullet,
    empty_section_map,
    parse_markdown,
    profile_hash,
    render_markdown,
    sanitize_bullets,
)
from twin.shared.memory.profile.constants import (
    DEFAULT_PROFILE_DIR,
    PROFILE_CURATION_MIN_BULLETS,
    PROFILE_FOOTER_HINT,
    PROFILE_HEADER,
    SECTION_HEADERS,
    SECTIONS,
)
from twin.shared.memory.profile.file_io import ProfileFileIO
from twin.shared.memory.profile.mutations import ProfileMutationPolicy

logger = logging.getLogger(__name__)

__all__ = ["MarkdownProfileStore", "profile_hash"]


class MarkdownProfileStore:
    """Per-user markdown profile store (T3).

    File: ``<base_path>/<user_id>.md`` with 8 fixed sections.
    Orchestrator over ProfileFileIO (I/O) + codec/mutations (policy).
    """

    def __init__(
        self,
        base_path: str = DEFAULT_PROFILE_DIR,
        *,
        enable_file_lock: bool = True,
        file_lock_poll_seconds: float = 0.05,
    ) -> None:
        self._io = ProfileFileIO(
            base_path,
            enable_file_lock=enable_file_lock,
            file_lock_poll_seconds=file_lock_poll_seconds,
        )

    async def read_raw(self, user_id: str) -> str:
        """Return full markdown text, auto-creating skeleton on first read."""
        path = self._io.path_for(user_id)
        async with self._io.locked(user_id):
            return await self._io.ensure_file(path)

    async def read_raw_hash(self, user_id: str) -> str:
        return profile_hash(await self.read_raw(user_id))

    async def read_section(self, user_id: str, section: str) -> list[str]:
        if section not in SECTIONS:
            raise ValueError(f"invalid section: {section!r}")
        return list(parse_markdown(await self.read_raw(user_id)).get(section, []))

    async def read_section_if_exists(self, user_id: str, section: str) -> list[str]:
        if section not in SECTIONS:
            raise ValueError(f"invalid section: {section!r}")
        text = self._io.read_text_sync(self._io.path_for(user_id))
        if text is None:
            return []
        return list(parse_markdown(text).get(section, []))

    async def get_system_prompt_context(self, user_id: str) -> str:
        parsed = parse_markdown(await self.read_raw(user_id))
        non_empty = [(k, parsed[k]) for k in SECTIONS if parsed.get(k)]
        if not non_empty:
            return ""
        lines: list[str] = [PROFILE_HEADER]
        for idx, (key, bullets) in enumerate(non_empty):
            lines.append(f"## {SECTION_HEADERS[key]}")
            for b in bullets:
                lines.append(f"- {b}")
            if idx != len(non_empty) - 1:
                lines.append("")
        lines.append("")
        lines.append(PROFILE_FOOTER_HINT)
        return "\n".join(lines)

    async def append_raw(
        self,
        user_id: str,
        section: str,
        content: str,
        source_memory_id: str | None = None,
    ) -> bool:
        """Append a bullet; False on empty/duplicate. Raises on invalid."""
        del source_memory_id
        if section not in SECTIONS:
            raise ValueError(f"invalid section: {section!r}")
        bullet = clean_append_bullet(content)
        if bullet is None:
            return False
        path = self._io.path_for(user_id)
        async with self._io.locked(user_id):
            parsed = parse_markdown(await self._io.ensure_file(path))
            existing = parsed.get(section, [])
            if bullet.lower() in {b.strip().lower() for b in existing}:
                return False
            parsed[section] = [*existing, bullet]
            try:
                self._io.atomic_write_sync(path, render_markdown(parsed))
            except OSError as exc:
                logger.warning("profile append failed for %s: %s", path, exc)
                raise
            return True

    async def write_raw(self, user_id: str, new_content: str) -> bool:
        path = self._io.path_for(user_id)
        async with self._io.locked(user_id):
            try:
                self._io.atomic_write_sync(path, new_content)
            except OSError as exc:
                logger.warning("profile write failed for %s: %s", path, exc)
                raise
            return True

    async def replace_section(
        self,
        user_id: str,
        section: str,
        bullets: list[str],
        expected_profile_hash: str | None = None,
    ) -> dict[str, Any]:
        """Replace one section; conflict when expected hash is stale."""
        if section not in SECTIONS:
            raise ValueError(f"invalid section: {section!r}")
        cleaned = sanitize_bullets(bullets)
        expected = (expected_profile_hash or "").strip() or None
        path = self._io.path_for(user_id)
        async with self._io.locked(user_id):
            text = await self._io.ensure_file(path)
            current_hash = profile_hash(text)
            if expected is not None and expected != current_hash:
                return {
                    "ok": False, "conflict": True, "section": section,
                    "profile_hash": current_hash,
                    "expected_profile_hash": expected, "written": False,
                }
            parsed = parse_markdown(text)
            before = list(parsed.get(section, []))
            parsed[section] = cleaned
            new_text = render_markdown(parsed)
            new_hash = profile_hash(new_text)
            if new_text != text:
                try:
                    self._io.atomic_write_sync(path, new_text)
                except OSError as exc:
                    logger.warning("profile section replace failed for %s: %s", path, exc)
                    raise
            return {
                "ok": True, "conflict": False, "section": section,
                "profile_hash": new_hash, "previous_profile_hash": current_hash,
                "written": new_text != text,
                "old_count": len(before), "new_count": len(cleaned),
            }

    async def replace_all(
        self,
        user_id: str,
        sections: dict[str, list[str]],
        expected_profile_hash: str | None = None,
        *,
        allow_shrink: bool = False,
    ) -> dict[str, Any]:
        """Rewrite whole profile; absent sections render empty."""
        if not isinstance(sections, dict):
            raise ValueError("sections must be a dict")
        cleaned: dict[str, list[str]] = empty_section_map()
        for key, bullets in sections.items():
            if key not in SECTIONS:
                raise ValueError(f"invalid section: {key!r}")
            cleaned[key] = sanitize_bullets(bullets)
        expected = (expected_profile_hash or "").strip() or None
        path = self._io.path_for(user_id)
        async with self._io.locked(user_id):
            text = await self._io.ensure_file(path)
            current_hash = profile_hash(text)
            if expected is not None and expected != current_hash:
                return {
                    "ok": False, "conflict": True,
                    "profile_hash": current_hash,
                    "expected_profile_hash": expected, "written": False,
                }
            parsed = parse_markdown(text)
            old_total = sum(len(parsed.get(s, [])) for s in SECTIONS)
            new_total = sum(len(cleaned[s]) for s in SECTIONS)
            if (
                old_total >= PROFILE_CURATION_MIN_BULLETS
                and new_total < old_total * 0.5
                and not allow_shrink
            ):
                return {
                    "ok": False, "shrink_blocked": True,
                    "old_total": old_total, "new_total": new_total,
                    "profile_hash": current_hash, "written": False,
                }
            new_text = render_markdown(cleaned)
            new_hash = profile_hash(new_text)
            if new_text != text:
                try:
                    self._io.atomic_write_sync(path, new_text)
                except OSError as exc:
                    logger.warning("profile replace_all failed for %s: %s", path, exc)
                    raise
            changed = [s for s in SECTIONS if parsed.get(s, []) != cleaned[s]]
            return {
                "ok": True, "conflict": False, "profile_hash": new_hash,
                "previous_profile_hash": current_hash,
                "written": new_text != text,
                "old_total": old_total, "new_total": new_total,
                "sections_changed": changed,
            }

    async def apply_consolidation_updates(
        self,
        user_id: str,
        rewrites: dict[str, list[str]] | None,
        appends: dict[str, list[str]] | None,
        expected_profile_hash: str | None,
    ) -> dict[str, Any]:
        """Apply rewrites+appends atomically under one lock/hash check.

        expected_profile_hash must be the hash of the profile text used in
        the LLM prompt. Stale hash => conflict, no write, new facts kept.
        """
        expected = (expected_profile_hash or "").strip() or None
        path = self._io.path_for(user_id)
        async with self._io.locked(user_id):
            text = await self._io.ensure_file(path)
            current_hash = profile_hash(text)
            if expected is not None and expected != current_hash:
                return {
                    "ok": False, "conflict": True,
                    "profile_hash": current_hash,
                    "expected_profile_hash": expected, "written": False,
                    "updated_sections": [], "rewritten_sections": [],
                }
            parsed = parse_markdown(text)
            new_sections, rewritten, updated = ProfileMutationPolicy.plan_consolidation(
                parsed, rewrites or {}, appends or {},
            )
            new_text = render_markdown(new_sections)
            new_hash = profile_hash(new_text)
            if new_text != text:
                try:
                    self._io.atomic_write_sync(path, new_text)
                except OSError as exc:
                    logger.warning("profile consolidation write failed for %s: %s", path, exc)
                    raise
            return {
                "ok": True, "conflict": False,
                "profile_hash": new_hash, "previous_profile_hash": current_hash,
                "written": new_text != text,
                "updated_sections": updated, "rewritten_sections": rewritten,
            }
