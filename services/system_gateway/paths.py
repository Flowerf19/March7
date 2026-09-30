"""Host-private path resolvers for the native System Gateway service.

Single source of truth for request-secret, approval-key and state/ledger
defaults, shared by the foreground CLI (``run``/``pair``), the installed
service and the bootstrap script. Explicit env overrides win on all OSes;
otherwise UID/platform decides. No hardcoded home paths. Stdlib only so
bootstrap tooling can reuse the same rules without an installed package.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APPROVAL_SECRET_FILE_ENV = "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE"
SHARED_SECRET_FILE_ENV = "SYSTEM_GATEWAY_SHARED_SECRET_FILE"
STATE_DIR_ENV = "SYSTEM_GATEWAY_STATE_DIR"
LEDGER_FILE_ENV = "SYSTEM_GATEWAY_APPROVAL_LEDGER_FILE"
LEDGER_FILENAME = "approval_ledger.db"


def is_root() -> bool:
    """Return True when running with UID 0 (POSIX only)."""

    try:
        return hasattr(os, "geteuid") and os.geteuid() == 0
    except OSError:
        return False


def _is_windows_admin() -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _windows_base_dir() -> Path:
    # Installed service (admin) uses machine-wide ProgramData; interactive
    # user runs use per-user LOCALAPPDATA. Never fall back to a baked path.
    if _is_windows_admin():
        return Path(os.getenv("ProgramData", r"C:\ProgramData"))
    return Path(os.getenv("LOCALAPPDATA") or str(Path.home()))


def _etc_approval_file() -> Path:
    return Path("/etc/system-gateway/approval_secret")


def _etc_shared_file() -> Path:
    return Path("/etc/system-gateway/secret")


def _etc_config_dir() -> Path:
    return Path("/etc/system-gateway")


def _user_config_dir() -> Path:
    return Path.home() / ".config" / "system-gateway"


def _mac_config_dir() -> Path:
    return (
        Path.home() / "Library" / "Application Support" / "system-gateway"
    )


def _win_config_dir() -> Path:
    return _windows_base_dir() / "system-gateway"


def _etc_usable_for_state() -> bool:
    """True when the root state dir exists+writable or /etc can create it."""

    etc_dir = Path("/etc/system-gateway")
    try:
        if etc_dir.is_dir():
            return os.access(etc_dir, os.W_OK | os.X_OK)
        return os.access("/etc", os.W_OK | os.X_OK)
    except OSError:
        return False


def default_config_dir() -> Path:
    """Base host-private config/state dir (platform + UID aware, no file env)."""

    if sys.platform.startswith("win"):
        return _win_config_dir()
    if sys.platform == "darwin":
        return _mac_config_dir()
    if is_root():
        # Matches systemd ReadWritePaths so the hardened unit can write.
        # Fall back to per-user state when /etc is not writable (tests).
        if _etc_usable_for_state():
            return _etc_config_dir()
    return _user_config_dir()


def default_shared_secret_file() -> Path:
    """Host-private default path for the shared request secret (outside repo)."""

    override = os.getenv(SHARED_SECRET_FILE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        return _win_config_dir() / "secret"
    if sys.platform == "darwin":
        return _mac_config_dir() / "secret"
    if is_root():
        # Mirror the approval resolver: an existing /etc secret wins so a
        # provisioned root service keeps reading it; otherwise /etc only
        # when writable, else per-user fallback for test/non-root coherence.
        try:
            if _etc_shared_file().exists():
                return _etc_shared_file()
        except OSError:
            pass
        if _etc_usable_for_state():
            return _etc_shared_file()
    return _user_config_dir() / "secret"


def default_approval_secret_file() -> Path:
    """Host-private default path for the owner approval key (outside repo)."""

    override = os.getenv(APPROVAL_SECRET_FILE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        return _win_config_dir() / "approval_secret"
    if sys.platform == "darwin":
        return _mac_config_dir() / "approval_secret"
    if is_root():
        # Installed root service uses /etc (the systemd unit also declares
        # it explicitly via env). When /etc is not usable (read-only test
        # containers), fall back to the per-user path so CLI and service
        # stay consistent in that environment.
        try:
            if _etc_approval_file().exists():
                return _etc_approval_file()
        except OSError:
            pass
        if _etc_usable_for_state():
            return _etc_approval_file()
    return _user_config_dir() / "approval_secret"


def default_state_dir() -> Path:
    """Private writable dir for native service state (ledger)."""

    override = os.getenv(STATE_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return default_config_dir()


def default_approval_ledger_file() -> Path:
    """Durable single-use ledger file (SQLite, host-private)."""

    override = os.getenv(LEDGER_FILE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return default_state_dir() / LEDGER_FILENAME
