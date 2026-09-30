"""Tests for the native System Gateway CLI."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from system_gateway import paths as gateway_paths
from system_gateway.cli.main import (
    _cmd_capabilities,
    _cmd_doctor,
    _cmd_install,
    _cmd_pair,
    _cmd_status,
    _cmd_update,
    _default_approval_secret_file,
    _default_config_dir,
    _default_secret_file,
    _ensure_secret_file,
    _read_secret_file,
    _render_darwin_plist,
    _render_linux_service,
    _service_file_path,
    build_parser,
    main,
)
from system_gateway.state import SERVICE_VERSION


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_build_parser_default_run():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command is None


def test_build_parser_subcommands():
    parser = build_parser()
    for cmd in ("status", "doctor", "capabilities", "logs", "pair", "install", "uninstall", "update", "run"):
        args = parser.parse_args([cmd])
        assert args.command == cmd


def test_read_secret_file_missing(tmp_path: Path):
    assert _read_secret_file(tmp_path / "does-not-exist") is None


def test_read_secret_file_present(tmp_path: Path):
    path = tmp_path / "secret"
    path.write_text(" s3cret ", encoding="utf-8")
    assert _read_secret_file(path) == "s3cret"


def test_ensure_secret_file_restricts_permissions(tmp_path: Path):
    path = tmp_path / "secret"
    _ensure_secret_file(path, "s3cret")
    assert path.read_text(encoding="utf-8") == "s3cret"
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert oct(path.stat().st_mode)[-3:] == "600"


def test_render_linux_service_uses_current_python_and_env(monkeypatch, tmp_path: Path):
    raw = "\n".join(
        [
            "[Service]",
            "ExecStart=/usr/local/bin/system-gateway run",
            "Environment=SYSTEM_GATEWAY_HOST=127.0.0.1",
            "Environment=SYSTEM_GATEWAY_PORT=8380",
            "Environment=SYSTEM_GATEWAY_SHARED_SECRET_FILE=/etc/system-gateway/secret",
            "Environment=SYSTEM_GATEWAY_APPROVAL_SECRET_FILE=/etc/system-gateway/approval_secret",
            "ProtectHome=true",
        ]
    )
    monkeypatch.setenv("SYSTEM_GATEWAY_HOST", "0.0.0.0")
    monkeypatch.setenv("SYSTEM_GATEWAY_PORT", "9999")

    rendered = _render_linux_service(
        raw,
        secret_file=tmp_path / "secret",
        approval_secret_file=tmp_path / "approval_secret",
    )

    assert f"ExecStart={sys.executable} -m system_gateway run" in rendered
    assert "Environment=SYSTEM_GATEWAY_HOST=0.0.0.0" in rendered
    assert "Environment=SYSTEM_GATEWAY_PORT=9999" in rendered
    assert f"Environment=SYSTEM_GATEWAY_SHARED_SECRET_FILE={tmp_path / 'secret'}" in rendered
    assert (
        f"Environment=SYSTEM_GATEWAY_APPROVAL_SECRET_FILE={tmp_path / 'approval_secret'}"
        in rendered
    )
    assert "Environment=PYTHONPATH=" not in rendered
    assert "ProtectHome=true" in rendered


def test_pair_generates_secret_file(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET", "")
    secret_file = tmp_path / "secret"
    approval_file = tmp_path / "approval_secret"
    args = _Namespace(
        secret_file=str(secret_file),
        approval_secret_file=str(approval_file),
        force=False,
    )
    assert _cmd_pair(args) == 0
    secret = _read_secret_file(secret_file)
    assert secret
    assert len(secret) >= 32
    # Approval key is created alongside, 0600, and its value is never printed.
    approval = _read_secret_file(approval_file)
    assert approval
    assert len(approval) >= 32
    assert approval != secret
    out = capsys.readouterr().out
    assert approval not in out
    assert str(approval_file) in out
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert oct(approval_file.stat().st_mode)[-3:] == "600"


def test_pair_refuses_to_overwrite_without_force(tmp_path: Path):
    secret_file = tmp_path / "secret"
    secret_file.write_text("existing", encoding="utf-8")
    args = _Namespace(
        secret_file=str(secret_file),
        approval_secret_file=str(tmp_path / "approval_secret"),
        force=False,
    )
    assert _cmd_pair(args) == 2


def test_pair_overwrites_with_force(tmp_path: Path):
    secret_file = tmp_path / "secret"
    secret_file.write_text("existing", encoding="utf-8")
    args = _Namespace(
        secret_file=str(secret_file),
        approval_secret_file=str(tmp_path / "approval_secret"),
        force=True,
    )
    assert _cmd_pair(args) == 0
    assert _read_secret_file(secret_file) != "existing"


def test_pair_keeps_existing_approval_without_force(tmp_path: Path, capsys):
    secret_file = tmp_path / "secret"
    approval_file = tmp_path / "approval_secret"
    approval_file.write_text("keep-me", encoding="utf-8")
    args = _Namespace(
        secret_file=str(secret_file),
        approval_secret_file=str(approval_file),
        force=False,
    )
    assert _cmd_pair(args) == 0
    assert _read_secret_file(approval_file) == "keep-me"
    out = capsys.readouterr().out
    assert "keep-me" not in out
    assert "kept" in out


def test_status_reports_health(monkeypatch):
    health = {
        "status": "ok",
        "service": "system_gateway",
        "version": SERVICE_VERSION,
        "uptime": 123,
        "platform": "linux",
    }
    monkeypatch.setattr(
        "system_gateway.cli.main._http_get_json", lambda url: health
    )
    args = _Namespace(url="http://gw")
    assert _cmd_status(args) == 0


def test_capabilities_reports_json(monkeypatch, capsys):
    caps = {"platform": "linux", "structured_actions": ["system.status"]}
    monkeypatch.setattr(
        "system_gateway.cli.main._http_get_json", lambda url: caps
    )
    args = _Namespace(url="http://gw")
    assert _cmd_capabilities(args) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == caps


def test_doctor_warns_without_secret(monkeypatch, capsys):
    health = {"status": "ok", "version": SERVICE_VERSION, "platform": "linux", "uptime": 1}
    caps = {"platform": "linux", "raw_shell": True, "shells": ["/bin/sh"], "structured_actions": []}
    call_count = {"count": 0}

    def fake_get(url):
        call_count["count"] += 1
        if "health" in url:
            return health
        return caps

    monkeypatch.setattr("system_gateway.cli.main._http_get_json", fake_get)
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", raising=False)
    args = _Namespace(url="http://gw")
    assert _cmd_doctor(args) == 2
    captured = capsys.readouterr()
    assert "No SYSTEM_GATEWAY_SHARED_SECRET" in captured.out
    assert "approval key" in captured.out


def test_update_rejects_without_secret(monkeypatch, capsys):
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET", raising=False)
    monkeypatch.delenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", raising=False)
    args = _Namespace(
        url="http://gw",
        target_version=None,
        from_version=None,
        approval_secret_file=None,
    )
    assert _cmd_update(args) == 1
    captured = capsys.readouterr()
    assert "SYSTEM_GATEWAY_SHARED_SECRET is required" in captured.err


def test_update_refuses_without_approval_key(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET", "request-secret")
    monkeypatch.setenv(
        "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(tmp_path / "missing")
    )
    args = _Namespace(
        url="http://gw",
        target_version=None,
        from_version=SERVICE_VERSION,
        approval_secret_file=None,
    )
    assert _cmd_update(args) == 1
    assert "approval key" in capsys.readouterr().err.lower()


def test_update_sends_approved_request(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET", "test-request-secret")
    approval_file = tmp_path / "approval_secret"
    approval_file.write_text("test-approval-secret", encoding="utf-8")
    response = {"ok": True, "message": "queued"}

    class _FakeResponse:
        def read(self):
            return json.dumps(response).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    opened = []

    def fake_urlopen(req, timeout):
        opened.append({"url": req.full_url, "data": req.data, "req": req})
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("urllib.request.Request", lambda *a, **kw: _Namespace(full_url=a[0], data=kw.get("data"), headers=kw.get("headers", {})))

    args = _Namespace(
        url="http://gw",
        target_version="0.2.0",
        from_version=SERVICE_VERSION,
        approval_secret_file=str(approval_file),
    )
    assert _cmd_update(args) == 0
    assert len(opened) == 1
    sent_headers = opened[0]["req"].headers
    assert "X-System-Gateway-Signature" in sent_headers
    assert "X-System-Gateway-Timestamp" in sent_headers
    assert "X-System-Gateway-Nonce" in sent_headers
    assert sent_headers["X-System-Gateway-Actor"] == "owner-cli"
    payload = json.loads(opened[0]["data"])
    assert payload["from_version"] == SERVICE_VERSION
    assert payload["to_version"] == "0.2.0"
    assert payload["approval_id"]
    # The approval is minted with the separate approval key over the
    # canonical update payload, and the request key cannot verify it.
    from twin.shared.system_gateway.auth import (
        canonical_approval_action,
        verify_approval_token,
    )

    canonical = canonical_approval_action(
        "self.update",
        {"from_version": SERVICE_VERSION, "to_version": "0.2.0"},
    )
    assert verify_approval_token(
        secret="test-approval-secret",
        token=payload["approval_id"],
        action=canonical,
        actor="owner-cli",
    ).valid is True
    assert verify_approval_token(
        secret="test-request-secret",
        token=payload["approval_id"],
        action=canonical,
        actor="owner-cli",
    ).valid is False
    # Approval key value never appears on stdout.
    assert "test-approval-secret" not in capsys.readouterr().out


def test_main_runs_subcommand(monkeypatch, capsys):
    """``main`` dispatches to subcommands and returns their exit code."""

    health = {"status": "ok", "service": "system_gateway", "version": SERVICE_VERSION, "uptime": 1, "platform": "linux"}
    monkeypatch.setattr("system_gateway.cli.main._http_get_json", lambda url: health)
    assert main(["--url", "http://gw", "status"]) == 0
    assert "Status:" in capsys.readouterr().out


def test_main_default_is_run(monkeypatch):
    """With no args, ``main`` defaults to ``run`` and creates the app."""

    run_called = []

    def fake_run_app(app, host, port):
        run_called.append((host, port))
        raise SystemExit(0)

    monkeypatch.setattr("system_gateway.cli.main.web.run_app", fake_run_app)
    with pytest.raises(SystemExit):
        main([])
    assert run_called == [("127.0.0.1", 8380)]


def test_main_run_uses_config_env(monkeypatch):
    monkeypatch.setenv("SYSTEM_GATEWAY_HOST", "0.0.0.0")
    monkeypatch.setenv("SYSTEM_GATEWAY_PORT", "9999")
    run_called = []

    def fake_run_app(app, host, port):
        run_called.append((host, port))
        raise SystemExit(0)

    monkeypatch.setattr("system_gateway.cli.main.web.run_app", fake_run_app)
    with pytest.raises(SystemExit):
        main(["run"])
    assert run_called == [("0.0.0.0", 9999)]


def _clear_secret_env(monkeypatch):
    for key in (
        "SYSTEM_GATEWAY_SHARED_SECRET_FILE",
        "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE",
        "SYSTEM_GATEWAY_STATE_DIR",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_config_dir_nonroot_linux_uses_home(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _clear_secret_env(monkeypatch)
    monkeypatch.setattr(gateway_paths, "is_root", lambda: False)
    monkeypatch.setattr(gateway_paths, "_etc_usable_for_state", lambda: False)
    assert _default_config_dir() == tmp_path / ".config" / "system-gateway"
    assert _default_secret_file() == tmp_path / ".config" / "system-gateway" / "secret"
    assert "/etc" not in str(_default_secret_file())


def test_default_secret_file_respects_shared_env(monkeypatch, tmp_path: Path):
    target = tmp_path / "custom" / "secret"
    monkeypatch.setenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE", str(target))
    assert _default_secret_file() == target


def test_default_secret_file_macos_library(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _clear_secret_env(monkeypatch)
    expected = tmp_path / "Library" / "Application Support" / "system-gateway" / "secret"
    assert _default_secret_file() == expected


def test_default_secret_file_windows_admin_vs_user(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pd = tmp_path / "ProgramData"
    la = tmp_path / "LocalAppData"
    monkeypatch.setenv("ProgramData", str(pd))
    monkeypatch.setenv("LOCALAPPDATA", str(la))
    _clear_secret_env(monkeypatch)
    monkeypatch.setattr(gateway_paths, "_is_windows_admin", lambda: True)
    assert _default_secret_file() == pd / "system-gateway" / "secret"
    monkeypatch.setattr(gateway_paths, "_is_windows_admin", lambda: False)
    assert _default_secret_file() == la / "system-gateway" / "secret"


def test_pair_defaults_to_home_not_etc_for_nonroot(
    monkeypatch, tmp_path: Path, capsys
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _clear_secret_env(monkeypatch)
    monkeypatch.setattr(gateway_paths, "is_root", lambda: False)
    monkeypatch.setattr(gateway_paths, "_etc_usable_for_state", lambda: False)
    args = _Namespace(secret_file=None, approval_secret_file=None, force=False)
    assert _cmd_pair(args) == 0
    secret = tmp_path / ".config" / "system-gateway" / "secret"
    approval = tmp_path / ".config" / "system-gateway" / "approval_secret"
    assert secret.exists() and approval.exists()
    assert _read_secret_file(secret)
    assert _read_secret_file(approval)
    out = capsys.readouterr().out
    assert str(secret) in out
    assert _read_secret_file(approval) not in out
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert oct(secret.stat().st_mode)[-3:] == "600"
        assert oct(approval.stat().st_mode)[-3:] == "600"


def test_render_darwin_plist_uses_venv_python_and_env(
    monkeypatch, tmp_path: Path
):
    import plistlib

    raw = _service_file_path("darwin").read_bytes()
    secret = tmp_path / "secret"
    approval = tmp_path / "approval_secret"
    state = tmp_path / "state"
    monkeypatch.setenv("SYSTEM_GATEWAY_HOST", "0.0.0.0")
    monkeypatch.setenv("SYSTEM_GATEWAY_PORT", "9999")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    rendered = _render_darwin_plist(
        raw, secret_file=secret, approval_secret_file=approval, state_dir=state
    )
    data = plistlib.loads(rendered)
    assert data["ProgramArguments"] == [sys.executable, "-m", "system_gateway", "run"]
    assert "/usr/local/bin/system-gateway" not in rendered.decode("utf-8")
    env = data["EnvironmentVariables"]
    assert env["SYSTEM_GATEWAY_HOST"] == "0.0.0.0"
    assert env["SYSTEM_GATEWAY_PORT"] == "9999"
    assert env["SYSTEM_GATEWAY_SHARED_SECRET_FILE"] == str(secret)
    assert env["SYSTEM_GATEWAY_APPROVAL_SECRET_FILE"] == str(approval)
    assert env["SYSTEM_GATEWAY_STATE_DIR"] == str(state)
    assert set(env) == {
        "SYSTEM_GATEWAY_HOST",
        "SYSTEM_GATEWAY_PORT",
        "SYSTEM_GATEWAY_SHARED_SECRET_FILE",
        "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE",
        "SYSTEM_GATEWAY_STATE_DIR",
    }
    assert data["Label"] == "com.twin.system-gateway"
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert str(tmp_path) in data["StandardOutPath"]
    assert "/Users/me" not in data["StandardOutPath"]


def test_install_windows_foreground_only_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    assert _cmd_install(_Namespace(approval_secret_file=None)) == 1
    out = capsys.readouterr().out
    assert "foreground-only" in out.lower() or "foreground" in out.lower()
    assert "Linux" not in out or "Windows" in out


def test_install_darwin_renders_plist_mocked(monkeypatch, tmp_path: Path):
    import plistlib

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _clear_secret_env(monkeypatch)
    monkeypatch.setenv("SYSTEM_GATEWAY_HOST", "127.0.0.1")
    monkeypatch.setenv("SYSTEM_GATEWAY_PORT", "8380")
    dst = tmp_path / "LaunchAgents" / "com.twin.system-gateway.plist"
    monkeypatch.setattr("system_gateway.cli.main._launchd_plist_path", lambda: dst)
    ran: list[list[str]] = []
    monkeypatch.setattr(
        "system_gateway.cli.main._run_or_warn", lambda cmd: ran.append(cmd)
    )
    assert _cmd_install(_Namespace(approval_secret_file=None)) == 0
    assert dst.exists()
    data = plistlib.loads(dst.read_bytes())
    assert data["ProgramArguments"] == [sys.executable, "-m", "system_gateway", "run"]
    env = data["EnvironmentVariables"]
    assert set(env) == {
        "SYSTEM_GATEWAY_HOST",
        "SYSTEM_GATEWAY_PORT",
        "SYSTEM_GATEWAY_SHARED_SECRET_FILE",
        "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE",
        "SYSTEM_GATEWAY_STATE_DIR",
    }
    base = tmp_path / "Library" / "Application Support" / "system-gateway"
    assert env["SYSTEM_GATEWAY_SHARED_SECRET_FILE"] == str(base / "secret")
    assert env["SYSTEM_GATEWAY_APPROVAL_SECRET_FILE"] == str(
        base / "approval_secret"
    )
    # Mocked launchctl only; no real service execution.
    assert ran == [["launchctl", "unload", str(dst)], ["launchctl", "load", "-w", str(dst)]]
    assert (tmp_path / "Library" / "Logs").is_dir()
    assert _default_approval_secret_file() == base / "approval_secret"
