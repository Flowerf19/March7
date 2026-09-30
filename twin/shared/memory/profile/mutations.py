"""Profile (T3) mutation policy: atomic rewrite+append planning."""
from __future__ import annotations

from twin.shared.memory.profile.codec import clean_append_bullet, sanitize_bullets
from twin.shared.memory.profile.constants import SECTIONS


class ProfileMutationPolicy:
    """Pure policy: apply rewrites/appends to parsed sections."""

    @staticmethod
    def plan_consolidation(
        parsed: dict[str, list[str]],
        rewrites: dict[str, list[str]] | None,
        appends: dict[str, list[str]] | None,
    ) -> tuple[dict[str, list[str]], list[str], list[str]]:
        """Return (new_sections, rewritten, updated) or raise ValueError.

        Empty rewrite lists are skipped (never wipe a section by accident).
        Empty append bullets are skipped; duplicates (case-insensitive)
        are skipped. Multiline/control/"- " bullets raise.
        """
        working: dict[str, list[str]] = {k: list(v) for k, v in parsed.items()}
        for key in SECTIONS:
            working.setdefault(key, [])
        rewritten: list[str] = []
        updated: list[str] = []

        for section, bullets in (rewrites or {}).items():
            if section not in SECTIONS:
                raise ValueError(f"invalid section: {section!r}")
            if not isinstance(bullets, list):
                raise ValueError(f"rewrite bullets for {section!r} must be a list")
            if not bullets:
                continue
            cleaned = sanitize_bullets(bullets)
            working[section] = cleaned
            rewritten.append(section)

        for section, bullets in (appends or {}).items():
            if section not in SECTIONS:
                raise ValueError(f"invalid section: {section!r}")
            if not isinstance(bullets, list):
                raise ValueError(f"append bullets for {section!r} must be a list")
            if section in rewritten:
                continue
            if not bullets:
                continue
            existing = list(working.get(section, []))
            lowered = {b.strip().lower() for b in existing}
            changed = False
            for raw in bullets:
                bullet = clean_append_bullet(raw)
                if bullet is None:
                    continue
                if bullet.lower() in lowered:
                    continue
                existing.append(bullet)
                lowered.add(bullet.lower())
                changed = True
            if changed:
                working[section] = existing
                updated.append(section)

        return working, rewritten, updated
