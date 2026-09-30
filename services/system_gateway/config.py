"""Runtime configuration for the System Gateway scaffold."""
from __future__ import annotations

from dataclasses import dataclass
import hmac
import os
from pathlib import Path

from . import keyfile as _keyfile
from .paths import (
    default_approval_ledger_file as _default_ledger_file,
)
from .paths import (
    default_approval_secret_file as _unified_approval_file,
)
from .paths import default_shared_secret_file as _unified_shared_file


def default_approval_secret_file() -> Path:
    """Host-private default path for the owner approval key (outside repo)."""

    # Single resolver shared with the CLI/service (env override wins on all
    # OSes; Linux UID 0 uses /etc, otherwise per-user config; no baked home).
    return _unified_approval_file()


def default_shared_secret_file() -> Path:
    """Host-private default path for the shared request secret (outside repo)."""

    # Same UID/platform rules as the approval resolver so foreground `run`
    # reads the file `pair`/bootstrap provisioned.
    return _unified_shared_file()


@dataclass(frozen=True)
class GatewayConfig:
    """Small configuration object for the native service."""

    host: str = "127.0.0.1"
    # Port 8380 matches SYSTEM_GATEWAY_URL in twin/shared/config/settings.py and
    # the SYSTEM_GATEWAY_URL wired into the march7/evernight containers.
    port: int = 8380
    # Generic shell execution is the gateway's one path; owner approval is the
    # control. Raw shell is denied by default and requires explicit opt-in
    # (SYSTEM_GATEWAY_RAW_SHELL=true); approval is still required when enabled.
    raw_shell_enabled: bool = False
    shared_secret: str | None = None
    # Separate owner approval key (Evernight/owner CLI only). Loaded from a
    # host-private file, never from inline env or the shared repo .env.
    approval_secret: str | None = None
    # Durable single-use ledger file (SQLite). None means the unified
    # default (env override or private state dir); ":memory:" is an
    # explicit test-only opt-in, never the production default.
    approval_ledger_file: Path | str | None = None

    def __post_init__(self) -> None:
        # The request signer must never become the issuer by
        # misconfiguration: equal credentials are rejected in both the
        # env-loaded and injected-config paths. No secret values in errors.
        if self.shared_secret and self.approval_secret:
            try:
                equal = hmac.compare_digest(
                    self.shared_secret.encode("utf-8"),
                    self.approval_secret.encode("utf-8"),
                )
            except (TypeError, ValueError):
                equal = self.shared_secret == self.approval_secret
            if equal:
                raise ValueError(
                    "approval credentials must differ from request credentials"
                )

    @classmethod
    def from_env(cls) -> GatewayConfig:
        """Load config from environment variables."""

        shared_secret = os.getenv("SYSTEM_GATEWAY_SHARED_SECRET") or None
        if not shared_secret:
            secret_file = os.getenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE") or str(
                default_shared_secret_file()
            )
            shared_secret = _read_secret_file(secret_file)

        approval_file = os.getenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE") or str(
            default_approval_secret_file()
        )
        approval_secret = _read_secret_file(approval_file)

        return cls(
            host=os.getenv("SYSTEM_GATEWAY_HOST", cls.host),
            port=_parse_port(os.getenv("SYSTEM_GATEWAY_PORT"), cls.port),
            raw_shell_enabled=_parse_bool(
                os.getenv("SYSTEM_GATEWAY_RAW_SHELL"), cls.raw_shell_enabled
            ),
            shared_secret=shared_secret,
            approval_secret=approval_secret,
            approval_ledger_file=_default_ledger_file(),
        )


def _parse_port(raw_port: str | None, default: int) -> int:
    if raw_port is None or raw_port == "":
        return default
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("SYSTEM_GATEWAY_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("SYSTEM_GATEWAY_PORT must be between 1 and 65535")
    return port


def _read_secret_file(path: str) -> str | None:
    # Symlink-safe read; missing/unreadable fails closed to None.
    return _keyfile.read_secret_file(path)


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
