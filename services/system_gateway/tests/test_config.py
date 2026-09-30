"""Tests for System Gateway runtime configuration."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from system_gateway.config import GatewayConfig


def test_from_env_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_RAW_SHELL", raising=False)
    config = GatewayConfig.from_env()
    assert config.host == "127.0.0.1"
    assert config.port == 8380
    # Raw shell is denied by default; explicit opt-in required.
    assert config.raw_shell_enabled is False
    assert config.shared_secret is None
    assert config.approval_secret is None


def test_from_env_requires_explicit_raw_shell_opt_in(monkeypatch):
    monkeypatch.setenv("SYSTEM_GATEWAY_RAW_SHELL", "true")
    config = GatewayConfig.from_env()
    assert config.raw_shell_enabled is True


def test_from_env_reads_secret_file(monkeypatch, tmp_path: Path):
    secret_file = tmp_path / "secret"
    secret_file.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", str(secret_file))
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET", raising=False)
    config = GatewayConfig.from_env()
    assert config.shared_secret == "file-secret"


def test_env_secret_overrides_file_secret(monkeypatch, tmp_path: Path):
    secret_file = tmp_path / "secret"
    secret_file.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", str(secret_file))
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET", "env-secret")
    config = GatewayConfig.from_env()
    assert config.shared_secret == "env-secret"


def test_missing_secret_file_falls_back_to_none(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", str(tmp_path / "missing"))
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET", raising=False)
    config = GatewayConfig.from_env()
    assert config.shared_secret is None


def test_from_env_custom_host_port(monkeypatch):
    monkeypatch.setenv("SYSTEM_GATEWAY_HOST", "0.0.0.0")
    monkeypatch.setenv("SYSTEM_GATEWAY_PORT", "9999")
    monkeypatch.setenv("SYSTEM_GATEWAY_RAW_SHELL", "true")
    config = GatewayConfig.from_env()
    assert config.host == "0.0.0.0"
    assert config.port == 9999
    assert config.raw_shell_enabled is True


def test_from_env_loads_approval_secret_from_file(monkeypatch, tmp_path: Path):
    approval_file = tmp_path / "approval_secret"
    approval_file.write_text("approval-key\n", encoding="utf-8")
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(approval_file))
    config = GatewayConfig.from_env()
    assert config.approval_secret == "approval-key"


def test_from_env_uses_host_private_approval_default(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    default_file = tmp_path / ".config" / "system-gateway" / "approval_secret"
    default_file.parent.mkdir(parents=True)
    default_file.write_text("default-approval-key", encoding="utf-8")
    assert GatewayConfig.from_env().approval_secret == "default-approval-key"


def test_from_env_ignores_inline_approval_secret_env(monkeypatch, tmp_path: Path):
    # Inline approval secrets are forbidden (shared .env is bind-mounted into
    # both agents); only the host-private file is read.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_SECRET", "inline-forbidden")
    assert GatewayConfig.from_env().approval_secret is None


def test_missing_approval_file_falls_back_to_none(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(tmp_path / "missing")
    )
    assert GatewayConfig.from_env().approval_secret is None


def test_parse_port_invalid(monkeypatch):
    monkeypatch.setenv("SYSTEM_GATEWAY_PORT", "not-a-port")
    with pytest.raises(ValueError, match="integer"):
        GatewayConfig.from_env()


def test_parse_port_out_of_range(monkeypatch):
    monkeypatch.setenv("SYSTEM_GATEWAY_PORT", "70000")
    with pytest.raises(ValueError, match="between 1 and 65535"):
        GatewayConfig.from_env()


def test_from_env_uses_host_private_shared_default(monkeypatch, tmp_path: Path):
    # Foreground `run` reads the same unified default `pair` provisions.
    from system_gateway import paths as gateway_paths

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", raising=False)
    monkeypatch.setattr(gateway_paths, "is_root", lambda: False)
    default_file = tmp_path / ".config" / "system-gateway" / "secret"
    default_file.parent.mkdir(parents=True)
    default_file.write_text("default-shared-key", encoding="utf-8")
    assert GatewayConfig.from_env().shared_secret == "default-shared-key"


def test_from_env_shared_env_overrides_default_file(monkeypatch, tmp_path: Path):
    from system_gateway import paths as gateway_paths

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(gateway_paths, "is_root", lambda: False)
    default_file = tmp_path / ".config" / "system-gateway" / "secret"
    default_file.parent.mkdir(parents=True)
    default_file.write_text("default-shared-key", encoding="utf-8")
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET", "env-wins")
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", raising=False)
    assert GatewayConfig.from_env().shared_secret == "env-wins"
