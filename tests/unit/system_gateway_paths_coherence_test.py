"""Coherent native path resolvers across bootstrap/CLI/Config (simulated only).

Simulates sys.platform/admin/user/env with fake TMP paths and mocks; no live
/etc, Library, ProgramData, launchctl, systemctl, or real .env reads.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from system_gateway import paths as gateway_paths
from system_gateway.cli import main as cli_main
from system_gateway.config import (
    default_approval_secret_file as config_approval,
)
from system_gateway.config import (
    default_shared_secret_file as config_shared,
)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_system_gateway.py"


def _load_bootstrap():
    # Import gateway first so bootstrap reuses the same paths instance when
    # present (keeps UID/platform mocks coherent); file-load covers the
    # pre-install case where the package is not importable.
    assert "system_gateway.paths" in sys.modules
    spec = importlib.util.spec_from_file_location("bootstrap_coherence", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._PATHS_CACHE = None
    return mod


def _clear_overrides(monkeypatch):
    for key in (
        "SYSTEM_GATEWAY_SHARED_SECRET_FILE",
        "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE",
        "SYSTEM_GATEWAY_STATE_DIR",
        "SYSTEM_GATEWAY_APPROVAL_LEDGER_FILE",
    ):
        monkeypatch.delenv(key, raising=False)


def _mock_both(monkeypatch, bootstrap, *, is_root=None, etc_usable=None, win_admin=None):
    b_paths = bootstrap._unified_paths()
    if is_root is not None:
        monkeypatch.setattr(gateway_paths, "is_root", lambda: is_root)
        if b_paths is not gateway_paths:
            monkeypatch.setattr(b_paths, "is_root", lambda: is_root)
    if etc_usable is not None:
        monkeypatch.setattr(
            gateway_paths, "_etc_usable_for_state", lambda: etc_usable
        )
        if b_paths is not gateway_paths:
            monkeypatch.setattr(b_paths, "_etc_usable_for_state", lambda: etc_usable)
    if win_admin is not None:
        monkeypatch.setattr(
            gateway_paths, "_is_windows_admin", lambda: win_admin
        )
        if b_paths is not gateway_paths:
            monkeypatch.setattr(b_paths, "_is_windows_admin", lambda: win_admin)


def test_linux_nonroot_all_agree_tmp_not_etc(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _clear_overrides(monkeypatch)
    _mock_both(monkeypatch, bootstrap, is_root=False, etc_usable=False)

    base = tmp_path / ".config" / "system-gateway"
    assert gateway_paths.default_config_dir() == base
    assert gateway_paths.default_shared_secret_file() == base / "secret"
    assert gateway_paths.default_approval_secret_file() == base / "approval_secret"
    assert gateway_paths.default_state_dir() == base
    assert cli_main._default_config_dir() == base
    assert cli_main._default_secret_file() == base / "secret"
    assert cli_main._default_approval_secret_file() == base / "approval_secret"
    assert config_shared() == base / "secret"
    assert config_approval() == base / "approval_secret"
    assert bootstrap.config_dir() == base
    assert bootstrap.secret_file_path() == base / "secret"
    assert bootstrap.approval_secret_file_path() == base / "approval_secret"
    assert "/etc" not in str(bootstrap.secret_file_path())


def test_linux_root_etc_usable_all_agree(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _clear_overrides(monkeypatch)
    _mock_both(monkeypatch, bootstrap, is_root=True, etc_usable=True)

    # Path comparison only; never writes to real /etc.
    assert gateway_paths.default_config_dir() == Path("/etc/system-gateway")
    assert gateway_paths.default_shared_secret_file() == Path(
        "/etc/system-gateway/secret"
    )
    assert gateway_paths.default_approval_secret_file() == Path(
        "/etc/system-gateway/approval_secret"
    )
    assert cli_main._default_secret_file() == Path("/etc/system-gateway/secret")
    assert config_shared() == Path("/etc/system-gateway/secret")
    assert bootstrap.secret_file_path() == Path("/etc/system-gateway/secret")
    assert bootstrap.approval_secret_file_path() == Path(
        "/etc/system-gateway/approval_secret"
    )


def test_linux_root_fallback_all_agree(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _clear_overrides(monkeypatch)
    _mock_both(monkeypatch, bootstrap, is_root=True, etc_usable=False)

    base = tmp_path / ".config" / "system-gateway"
    assert gateway_paths.default_shared_secret_file() == base / "secret"
    assert cli_main._default_secret_file() == base / "secret"
    assert config_shared() == base / "secret"
    assert bootstrap.secret_file_path() == base / "secret"
    assert bootstrap.approval_secret_file_path() == base / "approval_secret"


def test_macos_all_agree_library_not_config(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _clear_overrides(monkeypatch)
    _mock_both(monkeypatch, bootstrap, is_root=False, etc_usable=False)

    base = tmp_path / "Library" / "Application Support" / "system-gateway"
    assert gateway_paths.default_shared_secret_file() == base / "secret"
    assert gateway_paths.default_approval_secret_file() == base / "approval_secret"
    assert cli_main._default_secret_file() == base / "secret"
    assert config_shared() == base / "secret"
    assert bootstrap.secret_file_path() == base / "secret"
    # Regression: bootstrap previously returned ~/.config on macOS.
    assert bootstrap.approval_secret_file_path() == base / "approval_secret"
    assert ".config" not in str(bootstrap.approval_secret_file_path())


def test_windows_admin_all_agree_programdata(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pd = tmp_path / "ProgramData"
    la = tmp_path / "LocalAppData"
    monkeypatch.setenv("ProgramData", str(pd))
    monkeypatch.setenv("LOCALAPPDATA", str(la))
    _clear_overrides(monkeypatch)
    _mock_both(monkeypatch, bootstrap, win_admin=True)

    base = pd / "system-gateway"
    assert gateway_paths.default_shared_secret_file() == base / "secret"
    assert gateway_paths.default_approval_secret_file() == base / "approval_secret"
    assert cli_main._default_secret_file() == base / "secret"
    assert config_shared() == base / "secret"
    assert bootstrap.secret_file_path() == base / "secret"
    assert bootstrap.approval_secret_file_path() == base / "approval_secret"


def test_windows_user_all_agree_localappdata(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pd = tmp_path / "ProgramData"
    la = tmp_path / "LocalAppData"
    monkeypatch.setenv("ProgramData", str(pd))
    monkeypatch.setenv("LOCALAPPDATA", str(la))
    _clear_overrides(monkeypatch)
    _mock_both(monkeypatch, bootstrap, win_admin=False)

    base = la / "system-gateway"
    assert gateway_paths.default_shared_secret_file() == base / "secret"
    assert cli_main._default_secret_file() == base / "secret"
    assert config_shared() == base / "secret"
    assert bootstrap.secret_file_path() == base / "secret"
    assert bootstrap.approval_secret_file_path() == base / "approval_secret"


def test_explicit_override_wins_all(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    shared = tmp_path / "custom" / "secret"
    approval = tmp_path / "custom" / "approval_secret"
    state = tmp_path / "custom-state"
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", str(shared))
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(approval))
    monkeypatch.setenv("SYSTEM_GATEWAY_STATE_DIR", str(state))
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_LEDGER_FILE", raising=False)
    _mock_both(monkeypatch, bootstrap, is_root=False, etc_usable=False)

    assert gateway_paths.default_shared_secret_file() == shared
    assert gateway_paths.default_approval_secret_file() == approval
    assert gateway_paths.default_state_dir() == state
    assert cli_main._default_secret_file() == shared
    assert cli_main._default_approval_secret_file() == approval
    assert config_shared() == shared
    assert config_approval() == approval
    assert bootstrap.secret_file_path() == shared
    assert bootstrap.approval_secret_file_path() == approval


def test_bootstrap_file_load_without_installed_package(monkeypatch, tmp_path: Path):
    # Simulate pre-install: system_gateway.paths not yet importable; bootstrap
    # must still resolve via repo-relative stdlib file load (no aiohttp).
    spec = importlib.util.spec_from_file_location(
        "bootstrap_preinstall", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._PATHS_CACHE = None
    monkeypatch.delitem(sys.modules, "system_gateway.paths", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    shared = tmp_path / "override" / "secret"
    approval = tmp_path / "override" / "approval_secret"
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", str(shared))
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(approval))

    # Env overrides win without needing the installed package.
    assert mod.secret_file_path() == shared
    assert mod.approval_secret_file_path() == approval
    # No separate resolver duplication: module delegates to loaded helpers.
    assert mod._unified_paths().default_shared_secret_file() == shared
