"""Tests for the one-command System Gateway bootstrap script."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_system_gateway.py"


def _load_bootstrap_module():
    spec = importlib.util.spec_from_file_location("bootstrap_system_gateway", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ensure_venv_uses_system_site_packages(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap_module()
    monkeypatch.setattr(bootstrap, "VENV", str(tmp_path / "venv"))
    created = []

    def fake_create(path, *, with_pip, clear, system_site_packages):
        created.append(
            {
                "path": path,
                "with_pip": with_pip,
                "clear": clear,
                "system_site_packages": system_site_packages,
            }
        )
        bootstrap.venv_python(path).parent.mkdir(parents=True, exist_ok=True)
        bootstrap.venv_python(path).write_text("python", encoding="utf-8")

    monkeypatch.setattr(bootstrap.venv, "create", fake_create)

    assert bootstrap.ensure_venv() == tmp_path / "venv"
    assert created == [
        {
            "path": tmp_path / "venv",
            "with_pip": True,
            "clear": False,
            "system_site_packages": True,
        }
    ]


def test_gateway_install_on_windows_prints_foreground_command(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap_module()
    monkeypatch.setattr(bootstrap, "IS_WIN", True)
    monkeypatch.setattr(bootstrap, "IS_LINUX", False)
    monkeypatch.setattr(bootstrap, "IS_MAC", False)
    calls = []
    messages = []
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(bootstrap, "info", messages.append)

    bootstrap.gateway_install(tmp_path / "venv")

    assert calls == []
    assert any("foreground" in message for message in messages)


def test_gateway_install_on_linux_runs_cli_install(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap_module()
    monkeypatch.setattr(bootstrap, "IS_WIN", False)
    monkeypatch.setattr(bootstrap, "IS_LINUX", True)
    monkeypatch.setattr(bootstrap, "IS_MAC", False)
    venv_python = bootstrap.venv_python(tmp_path / "venv")
    ran = []

    class _Result:
        returncode = 0

    def fake_run(cmd, env):
        ran.append((cmd, env))
        return _Result()

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    bootstrap.gateway_install(tmp_path / "venv")

    assert ran[0][0] == [str(venv_python), "-m", "system_gateway", "install"]
    assert ran[0][1]["SYSTEM_GATEWAY_HOST"] == bootstrap.HOST
    assert ran[0][1]["SYSTEM_GATEWAY_PORT"] == bootstrap.PORT


def test_approval_secret_file_respects_env_override(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap_module()
    target = tmp_path / "custom" / "approval_secret"
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(target))
    assert bootstrap.approval_secret_file_path() == target


def test_secret_file_path_respects_shared_env_override(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap_module()
    target = tmp_path / "custom" / "secret"
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", str(target))
    assert bootstrap.secret_file_path() == target


def test_bootstrap_macos_uses_library_not_config(monkeypatch, tmp_path: Path):
    from system_gateway import paths as gateway_paths

    bootstrap = _load_bootstrap_module()
    bootstrap._PATHS_CACHE = None
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    base = tmp_path / "Library" / "Application Support" / "system-gateway"
    assert bootstrap.secret_file_path() == base / "secret"
    assert bootstrap.approval_secret_file_path() == base / "approval_secret"
    assert bootstrap.config_dir() == base
    assert gateway_paths.default_shared_secret_file() == base / "secret"


def test_bootstrap_linux_nonroot_uses_home_not_etc(monkeypatch, tmp_path: Path):
    from system_gateway import paths as gateway_paths

    bootstrap = _load_bootstrap_module()
    bootstrap._PATHS_CACHE = None
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    monkeypatch.setattr(gateway_paths, "is_root", lambda: False)
    monkeypatch.setattr(gateway_paths, "_etc_usable_for_state", lambda: False)
    b_paths = bootstrap._unified_paths()
    if b_paths is not gateway_paths:
        monkeypatch.setattr(b_paths, "is_root", lambda: False)
        monkeypatch.setattr(b_paths, "_etc_usable_for_state", lambda: False)
    base = tmp_path / ".config" / "system-gateway"
    assert bootstrap.config_dir() == base
    assert bootstrap.secret_file_path() == base / "secret"
    assert bootstrap.approval_secret_file_path() == base / "approval_secret"


def test_bootstrap_windows_admin_vs_user(monkeypatch, tmp_path: Path):
    from system_gateway import paths as gateway_paths

    bootstrap = _load_bootstrap_module()
    bootstrap._PATHS_CACHE = None
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pd = tmp_path / "ProgramData"
    la = tmp_path / "LocalAppData"
    monkeypatch.setenv("ProgramData", str(pd))
    monkeypatch.setenv("LOCALAPPDATA", str(la))
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", raising=False)
    b_paths = bootstrap._unified_paths()
    monkeypatch.setattr(gateway_paths, "_is_windows_admin", lambda: True)
    if b_paths is not gateway_paths:
        monkeypatch.setattr(b_paths, "_is_windows_admin", lambda: True)
    assert bootstrap.secret_file_path() == pd / "system-gateway" / "secret"
    monkeypatch.setattr(gateway_paths, "_is_windows_admin", lambda: False)
    if b_paths is not gateway_paths:
        monkeypatch.setattr(b_paths, "_is_windows_admin", lambda: False)
    assert bootstrap.secret_file_path() == la / "system-gateway" / "secret"


def test_ensure_approval_secret_creates_restricted_file(monkeypatch, tmp_path: Path):
    bootstrap = _load_bootstrap_module()
    target = tmp_path / "host-private" / "approval_secret"
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(target))
    messages: list[str] = []
    monkeypatch.setattr(bootstrap, "info", messages.append)

    returned = bootstrap.ensure_approval_secret()

    assert returned == target
    secret = target.read_text(encoding="utf-8").strip()
    assert secret
    assert len(secret) >= 32
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert oct(target.stat().st_mode)[-3:] == "600"
    # Value never logged; path is outside the repo.
    assert all(secret not in message for message in messages)
    assert str(bootstrap.REPO_ROOT) not in str(target)


def test_ensure_approval_secret_keeps_existing_and_ignores_env(
    monkeypatch, tmp_path: Path
):
    bootstrap = _load_bootstrap_module()
    target = tmp_path / "approval_secret"
    target.write_text("keep-approval", encoding="utf-8")
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(target))
    monkeypatch.setenv("SYSTEM_GATEWAY_APPROVAL_SECRET", "inline-forbidden")
    fake_env = tmp_path / ".env"
    monkeypatch.setattr(bootstrap, "ENV_FILE", fake_env)
    messages: list[str] = []
    monkeypatch.setattr(bootstrap, "info", messages.append)

    assert bootstrap.ensure_approval_secret() == target
    assert target.read_text(encoding="utf-8") == "keep-approval"
    # Never writes the approval key into the shared repo .env.
    assert not fake_env.exists()
    assert all("keep-approval" not in message for message in messages)
