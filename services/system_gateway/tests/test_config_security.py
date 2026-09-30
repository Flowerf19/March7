"""Config security: equal-credential rejection and unified path resolvers."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from system_gateway import paths as gateway_paths
from system_gateway.cli.main import _default_approval_secret_file as cli_default
from system_gateway.cli.main import _default_secret_file as cli_shared_default
from system_gateway.config import (
    GatewayConfig,
    default_approval_secret_file as config_default,
)
from system_gateway.config import default_shared_secret_file as config_shared_default


def test_injected_equal_keys_rejected():
    with pytest.raises(ValueError) as excinfo:
        GatewayConfig(shared_secret="same-value", approval_secret="same-value")
    assert "same-value" not in str(excinfo.value)


def test_env_equal_keys_rejected(monkeypatch, tmp_path: Path):
    shared = tmp_path / "secret"
    approval = tmp_path / "approval"
    shared.write_text("equal-both\n", encoding="utf-8")
    approval.write_text("equal-both\n", encoding="utf-8")
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", str(shared))
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET", raising=False)
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(approval))
    with pytest.raises(ValueError) as excinfo:
        GatewayConfig.from_env()
    assert "equal-both" not in str(excinfo.value)


def test_distinct_keys_accepted():
    config = GatewayConfig(shared_secret="request-key", approval_secret="owner-key")
    assert config.shared_secret == "request-key"
    assert config.approval_secret == "owner-key"


def test_missing_owner_key_allowed_in_config_but_none(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(tmp_path / "missing")
    )
    assert GatewayConfig.from_env().approval_secret is None


def test_resolvers_agree_on_env_override(monkeypatch, tmp_path: Path):
    target = tmp_path / "custom" / "approval_secret"
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(target))
    assert config_default() == target
    assert cli_default() == target
    assert gateway_paths.default_approval_secret_file() == target


def test_resolvers_agree_without_override(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    # Force the non-root branch deterministically regardless of test UID.
    monkeypatch.setattr(gateway_paths, "is_root", lambda: False)
    expected = tmp_path / ".config" / "system-gateway" / "approval_secret"
    assert config_default() == expected
    assert cli_default() == expected


def test_no_baked_home_in_resolver(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(gateway_paths, "is_root", lambda: False)
    resolved = str(config_default())
    assert "/home/flowerf" not in resolved
    assert str(tmp_path) in resolved


def test_root_uses_etc_when_writable(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(gateway_paths, "is_root", lambda: True)
    monkeypatch.setattr(gateway_paths, "_etc_usable_for_state", lambda: True)
    assert config_default() == Path("/etc/system-gateway/approval_secret")
    assert gateway_paths.default_state_dir() == Path("/etc/system-gateway")


def test_root_falls_back_when_etc_unusable(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(gateway_paths, "is_root", lambda: True)
    monkeypatch.setattr(gateway_paths, "_etc_usable_for_state", lambda: False)
    assert config_default() == tmp_path / ".config" / "system-gateway" / "approval_secret"


def test_ledger_env_override_wins(monkeypatch, tmp_path: Path):
    target = tmp_path / "custom.db"
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_LEDGER_FILE", str(target))
    assert gateway_paths.default_approval_ledger_file() == target
    assert GatewayConfig.from_env().approval_ledger_file == target


def test_ledger_default_is_private_db(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_LEDGER_FILE", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_STATE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(gateway_paths, "is_root", lambda: False)
    ledger = gateway_paths.default_approval_ledger_file()
    assert ledger.name == "approval_ledger.db"
    assert str(tmp_path) in str(ledger)


def test_shared_resolvers_agree_on_env_override(monkeypatch, tmp_path: Path):
    target = tmp_path / "custom" / "secret"
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", str(target))
    assert config_shared_default() == target
    assert cli_shared_default() == target
    assert gateway_paths.default_shared_secret_file() == target


def test_shared_resolvers_agree_without_override(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(gateway_paths, "is_root", lambda: False)
    expected = tmp_path / ".config" / "system-gateway" / "secret"
    assert config_shared_default() == expected
    assert cli_shared_default() == expected
    assert gateway_paths.default_shared_secret_file() == expected


def test_shared_and_approval_share_base_dir(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(gateway_paths, "is_root", lambda: False)
    assert (
        gateway_paths.default_shared_secret_file().parent
        == gateway_paths.default_approval_secret_file().parent
        == gateway_paths.default_config_dir()
        == gateway_paths.default_state_dir()
    )
