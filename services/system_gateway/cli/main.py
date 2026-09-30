"""Command-line interface for the native System Gateway service.

The CLI is the owner-facing control plane on the host. It can:

* run the gateway service (`system-gateway run`, default)
* report local status/capabilities (`status`, `doctor`, `capabilities`)
* stream service logs (`logs`)
* generate and store shared auth material (`pair`)
* install/uninstall the native service (`install`, `uninstall`)
* request an in-place update (`update`)

Trust model (plan A): request signing uses the shared request secret
(``SYSTEM_GATEWAY_SHARED_SECRET`` env or secret file, shared with agent
containers for HMAC). Owner approvals use a SEPARATE approval key stored in a
host-private file (``SYSTEM_GATEWAY_APPROVAL_SECRET_FILE`` or
``~/.config/system-gateway/approval_secret``), held only by the gateway
service, Evernight (owner-trusted issuer, via a dedicated mount), and this
owner CLI. The approval key value is never printed and never written to the
shared repo ``.env``. Missing approval key fails closed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import plistlib
import secrets
import shlex
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

from aiohttp import web

from .. import keyfile as _keyfile
from ..config import GatewayConfig
from ..paths import default_approval_secret_file as _unified_approval_file
from ..paths import default_config_dir as _unified_config_dir
from ..paths import default_shared_secret_file as _unified_shared_file
from ..paths import default_state_dir as _unified_state_dir
from ..server import create_app
from ..state import SERVICE_VERSION

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8380"


def _default_config_dir() -> Path:
    """Return the host config directory used by ``pair``."""

    # Single UID/platform-aware resolver shared with the service and
    # bootstrap (Linux root uses /etc when writable, otherwise per-user;
    # macOS Library; Windows admin ProgramData else LOCALAPPDATA).
    return _unified_config_dir()


def _default_secret_file() -> Path:
    # SYSTEM_GATEWAY_SHARED_SECRET_FILE override wins; otherwise the
    # unified platform/UID default so non-root pair never targets /etc.
    return _unified_shared_file()


def _default_approval_secret_file() -> Path:
    """Host-private default path for the owner approval key (outside repo)."""

    # Single resolver shared with the service (env override wins on all
    # OSes; Linux UID 0 uses /etc, otherwise per-user config).
    return _unified_approval_file()


def _read_approval_secret(explicit: str | None = None) -> str | None:
    """Read the owner approval key from its host-private file only."""

    path = Path(explicit) if explicit else _default_approval_secret_file()
    return _keyfile.read_secret_file(path)


def _read_secret_file(path: Path) -> str | None:
    return _keyfile.read_secret_file(path)


def _ensure_secret_file(path: Path, secret: str) -> None:
    # Explicit write path (pair --force or fresh provisioning): atomic 0600
    # temp publication, no 0644 window. Fresh owner-key creation without
    # --force uses exclusive-create in _cmd_pair so it never replaces.
    _keyfile.overwrite_secret_file_atomic(path, secret)


def _http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cmd_status(args: argparse.Namespace) -> int:
    url = (args.url or DEFAULT_URL).rstrip("/")
    try:
        data = _http_get_json(f"{url}/health")
    except Exception as exc:  # pragma: no cover - exercised manually
        print(f"❌ System Gateway unreachable at {url}: {exc}", file=sys.stderr)
        return 1

    print(f"Status: {data.get('status')}")
    print(f"Service: {data.get('service')}")
    print(f"Version: {data.get('version')}")
    print(f"Uptime: {data.get('uptime')}s")
    print(f"Platform: {data.get('platform')}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    url = (args.url or DEFAULT_URL).rstrip("/")
    ok = True
    try:
        health = _http_get_json(f"{url}/health")
    except Exception as exc:  # pragma: no cover - exercised manually
        print(f"❌ /health unreachable: {exc}")
        return 1

    print("Health:")
    for key in ("status", "version", "platform", "uptime"):
        print(f"  {key}: {health.get(key)}")

    try:
        caps = _http_get_json(f"{url}/capabilities")
    except Exception as exc:  # pragma: no cover - exercised manually
        print(f"❌ /capabilities unreachable: {exc}")
        return 1

    print("Capabilities:")
    for key in ("platform", "shells", "features", "raw_shell", "structured_actions"):
        print(f"  {key}: {caps.get(key)}")

    if health.get("version") != SERVICE_VERSION:
        print(f"⚠️  Service reports {health.get('version')}; CLI is {SERVICE_VERSION}.")
        ok = False

    secret = os.getenv("SYSTEM_GATEWAY_SHARED_SECRET")
    secret_file = os.getenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE")
    if not secret and not secret_file:
        print("⚠️  No SYSTEM_GATEWAY_SHARED_SECRET or SYSTEM_GATEWAY_SHARED_SECRET_FILE set.")
        print("   Run `system-gateway pair` to provision one.")
        ok = False

    approval_path = _default_approval_secret_file()
    if not approval_path.exists():
        print(f"⚠️  No owner approval key at {approval_path}.")
        print("   Run `system-gateway pair` to provision one (fail-closed without it).")
        ok = False

    return 0 if ok else 2


def _cmd_capabilities(args: argparse.Namespace) -> int:
    url = (args.url or DEFAULT_URL).rstrip("/")
    try:
        data = _http_get_json(f"{url}/capabilities")
    except Exception as exc:  # pragma: no cover - exercised manually
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    print(json.dumps(data, indent=2))
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    """Stream native service logs using the platform's log tool."""

    if sys.platform.startswith("linux"):
        cmd = ["journalctl", "-u", "system-gateway", "-f"]
    elif sys.platform == "darwin":
        cmd = ["log", "stream", "--predicate", 'process == "system-gateway"']
    else:  # pragma: no cover
        print("❌ `logs` is not implemented for this platform.")
        print("   On Windows, use Event Viewer or the console output of the service.")
        return 1

    try:
        os.execvp(cmd[0], cmd)
    except FileNotFoundError as exc:  # pragma: no cover - exercised manually
        print(f"❌ Could not start log viewer: {exc}", file=sys.stderr)
        return 1


def _cmd_pair(args: argparse.Namespace) -> int:
    """Generate and store shared auth material for pairing with a container."""

    secret_file = Path(args.secret_file) if args.secret_file else _default_secret_file()
    approval_file = (
        Path(getattr(args, "approval_secret_file", None))
        if getattr(args, "approval_secret_file", None)
        else _default_approval_secret_file()
    )

    existing = _read_secret_file(secret_file)
    if existing and not args.force:
        print(f"⚠️  Secret file already exists at {secret_file}.")
        print("   Use --force to regenerate (this will invalidate existing clients).")
        return 2

    secret = secrets.token_urlsafe(32)
    if args.force:
        _ensure_secret_file(secret_file, secret)
    else:
        try:
            _keyfile.write_new_secret_file(secret_file, secret)
        except FileExistsError:
            print(f"⚠️  Secret file already exists at {secret_file}.")
            print("   Use --force to regenerate (this will invalidate existing clients).")
            return 2

    print("🔐 Generated shared secret for System Gateway pairing.")
    print(f"   Stored at: {secret_file}")
    print("   Permissions: owner-read only")
    print()
    print("Add this secret to the agent containers via ONE of these methods:")
    print()
    print("  1. Environment variable on the host that launches the containers:")
    print(f"     export SYSTEM_GATEWAY_SHARED_SECRET={secret}")
    print()
    print("  2. Point the service at the file (host path must be readable by service):")
    print(f"     export SYSTEM_GATEWAY_SHARED_SECRET_FILE={secret_file}")
    print()
    print("  3. In container orchestration, set SYSTEM_GATEWAY_SHARED_SECRET from a secret.")
    print()
    print("⚠️  Do not commit this value. Treat it like a password.")

    # Separate owner approval key: host-private file, never printed, never in
    # the shared repo .env. Mount read-only into Evernight only.
    if args.force:
        _ensure_secret_file(approval_file, secrets.token_urlsafe(32))
        print()
        print("🔑 Generated owner approval key (value NOT shown).")
        print(f"   Stored at: {approval_file}")
        print("   Permissions: owner-read only")
        print("   Mount read-only into Evernight only; never into March7.")
        print("   Never copy this value into the shared repo .env.")
    else:
        try:
            _keyfile.write_new_secret_file(approval_file, secrets.token_urlsafe(32))
        except FileExistsError:
            print()
            print(f"🔑 Owner approval key already exists at {approval_file} (kept).")
        else:
            print()
            print("🔑 Generated owner approval key (value NOT shown).")
            print(f"   Stored at: {approval_file}")
            print("   Permissions: owner-read only")
            print("   Mount read-only into Evernight only; never into March7.")
            print("   Never copy this value into the shared repo .env.")

    if not secret_file.parent.exists():
        secret_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return 0


def _service_file_path(platform: str) -> Path:
    here = Path(__file__).resolve().parent.parent / "packaging"
    if platform == "linux":
        return here / "linux" / "system-gateway.service"
    if platform == "darwin":
        return here / "macos" / "com.twin.system-gateway.plist"
    raise RuntimeError(f"unsupported platform: {platform}")


def _systemd_unit_path() -> Path:
    return Path("/etc/systemd/system/system-gateway.service")


def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "com.twin.system-gateway.plist"


def _is_root() -> bool:
    return os.geteuid() == 0 if hasattr(os, "geteuid") else False


def _render_linux_service(
    raw: str, *, secret_file: Path, approval_secret_file: Path | None = None
) -> str:
    """Render the packaged Linux unit for the current Python environment."""

    host = os.getenv("SYSTEM_GATEWAY_HOST", GatewayConfig.host)
    port = os.getenv("SYSTEM_GATEWAY_PORT", str(GatewayConfig.port))
    exec_start = f"{shlex.quote(sys.executable)} -m system_gateway run"
    rendered_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("ExecStart="):
            rendered_lines.append(f"ExecStart={exec_start}")
        elif line.startswith("Environment=SYSTEM_GATEWAY_HOST="):
            rendered_lines.append(f"Environment=SYSTEM_GATEWAY_HOST={host}")
        elif line.startswith("Environment=SYSTEM_GATEWAY_PORT="):
            rendered_lines.append(f"Environment=SYSTEM_GATEWAY_PORT={port}")
        elif line.startswith("Environment=SYSTEM_GATEWAY_SHARED_SECRET_FILE="):
            rendered_lines.append(
                f"Environment=SYSTEM_GATEWAY_SHARED_SECRET_FILE={secret_file}"
            )
        elif line.startswith("Environment=SYSTEM_GATEWAY_APPROVAL_SECRET_FILE="):
            if approval_secret_file is not None:
                rendered_lines.append(
                    "Environment=SYSTEM_GATEWAY_APPROVAL_SECRET_FILE="
                    f"{approval_secret_file}"
                )
            else:
                rendered_lines.append(line)
        else:
            rendered_lines.append(line)
    return "\n".join(rendered_lines) + "\n"


def _render_darwin_plist(
    raw: bytes,
    *,
    secret_file: Path,
    approval_secret_file: Path,
    state_dir: Path | None = None,
) -> bytes:
    """Render the packaged macOS plist for the current venv python."""

    data = plistlib.loads(raw)
    host = os.getenv("SYSTEM_GATEWAY_HOST", GatewayConfig.host)
    port = os.getenv("SYSTEM_GATEWAY_PORT", str(GatewayConfig.port))
    # Never leave the /usr/local/bin symlink hope: bootstrap installs a venv
    # under ~/Library/... so the agent must exec the current venv python.
    data["ProgramArguments"] = [sys.executable, "-m", "system_gateway", "run"]
    env = dict(data.get("EnvironmentVariables") or {})
    env["SYSTEM_GATEWAY_HOST"] = host
    env["SYSTEM_GATEWAY_PORT"] = port
    env["SYSTEM_GATEWAY_SHARED_SECRET_FILE"] = str(secret_file)
    env["SYSTEM_GATEWAY_APPROVAL_SECRET_FILE"] = str(approval_secret_file)
    if state_dir is not None:
        env["SYSTEM_GATEWAY_STATE_DIR"] = str(state_dir)
    data["EnvironmentVariables"] = env
    home = str(Path.home())
    for key in ("StandardOutPath", "StandardErrorPath"):
        val = data.get(key)
        if isinstance(val, str) and "/Users/me" in val:
            data[key] = val.replace("/Users/me", home)
    # Label/RunAtLoad/KeepAlive/logs are retained from the template.
    return plistlib.dumps(data, fmt=plistlib.FMT_XML)


def _ensure_darwin_log_dir(rendered: bytes) -> None:
    """Create the user log dir when the plist logs to the standard path."""

    try:
        data = plistlib.loads(rendered)
    except Exception:
        return
    allowed = Path.home() / "Library" / "Logs"
    for key in ("StandardOutPath", "StandardErrorPath"):
        val = data.get(key)
        if not isinstance(val, str):
            continue
        try:
            parent = Path(val).expanduser().parent
        except Exception:
            continue
        try:
            if parent == allowed or parent.is_relative_to(allowed):
                parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def _cmd_install(args: argparse.Namespace) -> int:
    """Install and start the native service."""

    if sys.platform.startswith("win"):
        print("❌ Native install is not supported on Windows (foreground-only).")
        print("   See services/system_gateway/packaging/windows/README.md.")
        print(f"   Run in foreground: {sys.executable} -m system_gateway run")
        return 1
    if sys.platform == "darwin":
        platform = "darwin"
    elif sys.platform.startswith("linux"):
        platform = "linux"
    else:  # pragma: no cover
        print("❌ Native install is only supported on Linux and macOS.")
        print("   On Windows see packaging/windows/README.md for manual setup.")
        return 1

    src = _service_file_path(platform)
    if not src.exists():
        print(f"❌ Packaging file not found: {src}", file=sys.stderr)
        return 1

    # Ensure secrets exist before installing; otherwise the service would refuse
    # mutating requests and most admin operations would fail.
    secret_file = _default_secret_file()
    approval_file = getattr(args, "approval_secret_file", None)
    approval_path = (
        Path(approval_file) if approval_file else _default_approval_secret_file()
    )
    if not secret_file.exists() or not approval_path.exists():
        print("🔐 No shared/approval secret found. Generating one first...")
        _cmd_pair(
            argparse.Namespace(
                secret_file=str(secret_file),
                approval_secret_file=str(approval_path),
                force=False,
            )
        )

    if platform == "linux":
        if not _is_root():
            print("❌ Linux install requires root (systemctl/systemd).", file=sys.stderr)
            return 1
        dst = _systemd_unit_path()
        raw = src.read_text(encoding="utf-8")
        dst.write_text(
            _render_linux_service(
                raw, secret_file=secret_file, approval_secret_file=approval_path
            ),
            encoding="utf-8",
        )
        _run_or_warn(["systemctl", "daemon-reload"])
        _run_or_warn(["systemctl", "enable", "--now", "system-gateway"])
        print(f"✅ Installed {dst} and started system-gateway.service")
        print("   Check status: systemctl status system-gateway")
        print(f"   Secret: {secret_file}")
        print(f"   Approval key: {approval_path} (never printed)")
        return 0

    if platform == "darwin":
        dst = _launchd_plist_path()
        raw_bytes = src.read_bytes()
        state_dir = _unified_state_dir()
        rendered = _render_darwin_plist(
            raw_bytes,
            secret_file=secret_file,
            approval_secret_file=approval_path,
            state_dir=state_dir,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        _ensure_darwin_log_dir(rendered)
        dst.write_bytes(rendered)
        _run_or_warn(["launchctl", "unload", str(dst)])
        _run_or_warn(["launchctl", "load", "-w", str(dst)])
        print(f"✅ Installed {dst} and loaded com.twin.system-gateway")
        print("   Check status: launchctl list | grep com.twin.system-gateway")
        print(f"   Secret: {secret_file}")
        print(f"   Approval key: {approval_path} (never printed)")
        print(f"   State: {state_dir}")
        return 0

    return 1  # pragma: no cover


def _cmd_uninstall(args: argparse.Namespace) -> int:
    """Stop and remove the native service."""

    if sys.platform.startswith("linux"):
        if not _is_root():
            print("❌ Linux uninstall requires root.", file=sys.stderr)
            return 1
        _run_or_warn(["systemctl", "stop", "system-gateway"])
        _run_or_warn(["systemctl", "disable", "system-gateway"])
        dst = _systemd_unit_path()
        if dst.exists():
            dst.unlink()
        _run_or_warn(["systemctl", "daemon-reload"])
        print("✅ Removed system-gateway.service")
        return 0

    if sys.platform == "darwin":
        dst = _launchd_plist_path()
        if dst.exists():
            _run_or_warn(["launchctl", "unload", str(dst)])
            dst.unlink()
        print("✅ Removed com.twin.system-gateway launch agent")
        return 0

    print("❌ Native uninstall is only supported on Linux and macOS.", file=sys.stderr)
    return 1


def _run_or_warn(cmd: list[str]) -> None:
    try:
        result = os.spawnvpe(os.P_WAIT, cmd[0], cmd, os.environ)
        if result != 0:
            logger.warning("Command %s exited with %d", cmd, result)
    except FileNotFoundError:
        logger.warning("Command not found: %s", cmd[0])


def _cmd_update(args: argparse.Namespace) -> int:
    """Request an in-place update from the running service."""

    url = (args.url or DEFAULT_URL).rstrip("/")
    secret = os.getenv("SYSTEM_GATEWAY_SHARED_SECRET")
    if not secret:
        secret_file = os.getenv("SYSTEM_GATEWAY_SHARED_SECRET_FILE")
        if secret_file:
            secret = _read_secret_file(Path(secret_file))
    if not secret:
        print("❌ SYSTEM_GATEWAY_SHARED_SECRET is required to request an update.", file=sys.stderr)
        return 1
    approval_secret = _read_approval_secret(
        getattr(args, "approval_secret_file", None)
    )
    if not approval_secret:
        print(
            "❌ Owner approval key is missing; refusing update. "
            f"Provision {_default_approval_secret_file()} via `system-gateway pair`.",
            file=sys.stderr,
        )
        return 1

    # Import here so the CLI can still print help without importing auth helpers.
    from twin.shared.system_gateway.auth import (
        canonical_approval_action,
        headers_from_signed,
        mint_approval_token,
        sign_request,
    )

    actor = "owner-cli"
    exec_payload: dict[str, object] = {
        "from_version": args.from_version or SERVICE_VERSION,
    }
    if args.target_version:
        exec_payload["to_version"] = args.target_version
    try:
        canonical = canonical_approval_action("self.update", exec_payload)
    except ValueError as exc:
        print(f"❌ Invalid update request: {exc}", file=sys.stderr)
        return 1
    approval_id = mint_approval_token(
        secret=approval_secret,
        action=canonical,
        actor=actor,
    )
    payload = dict(exec_payload)
    payload["approval_id"] = approval_id

    body_bytes = json.dumps(payload).encode("utf-8")
    signed = sign_request(
        secret=secret,
        method="POST",
        path="/self/update",
        actor=actor,
        body=body_bytes,
    )
    headers = {"Content-Type": "application/json"}
    headers.update(headers_from_signed(signed))

    req = urllib.request.Request(
        f"{url}/self/update",
        data=body_bytes,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            data = json.loads(body)
        except Exception:
            data = {"error": body}
        print(f"❌ Update request failed: HTTP {exc.code}: {data.get('error')}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"❌ Update request failed: {exc}", file=sys.stderr)
        return 1

    if data.get("ok"):
        print(f"✅ {data.get('message') or 'update requested'}")
        return 0
    print(f"⚠️  Update request rejected: {data.get('error')}", file=sys.stderr)
    return 1


def _cmd_run(args: argparse.Namespace) -> int:
    """Run the gateway service in the foreground."""

    config = GatewayConfig.from_env()
    web.run_app(create_app(config), host=config.host, port=config.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="system-gateway",
        description="Native System Gateway service CLI.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("SYSTEM_GATEWAY_URL", DEFAULT_URL),
        help="Base URL of the running gateway (default: %(default)s).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("status", help="Show gateway health.")
    subparsers.add_parser("doctor", help="Health + capabilities + config checks.")
    subparsers.add_parser("capabilities", help="Show gateway capabilities.")
    subparsers.add_parser("logs", help="Tail native service logs.")

    pair_parser = subparsers.add_parser("pair", help="Generate and store shared secret.")
    pair_parser.add_argument(
        "--secret-file",
        default=None,
        help="Path to write the secret (default: platform config dir).",
    )
    pair_parser.add_argument(
        "--approval-secret-file",
        default=None,
        help="Path for the owner approval key (default: host-private config).",
    )
    pair_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if a secret already exists.",
    )

    install_parser = subparsers.add_parser("install", help="Install native service (Linux/macOS).")
    install_parser.add_argument(
        "--approval-secret-file",
        default=None,
        help="Path for the owner approval key (default: host-private config).",
    )
    subparsers.add_parser("uninstall", help="Remove native service (Linux/macOS).")

    update_parser = subparsers.add_parser("update", help="Request an in-place update.")
    update_parser.add_argument(
        "--target-version",
        default=None,
        help="Target version to update to.",
    )
    update_parser.add_argument(
        "--from-version",
        default=None,
        help="Current version (default: CLI version).",
    )
    update_parser.add_argument(
        "--approval-secret-file",
        default=None,
        help="Path to the owner approval key file.",
    )

    run_parser = subparsers.add_parser("run", help="Run the service in the foreground.")
    run_parser.add_argument(
        "--host",
        default=None,
        help="Bind host (default: config/env).",
    )
    run_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: config/env).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = args.command or "run"
    if command == "run":
        return _cmd_run(args)

    handler = {
        "status": _cmd_status,
        "doctor": _cmd_doctor,
        "capabilities": _cmd_capabilities,
        "logs": _cmd_logs,
        "pair": _cmd_pair,
        "install": _cmd_install,
        "uninstall": _cmd_uninstall,
        "update": _cmd_update,
    }.get(command)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)
