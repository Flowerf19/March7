#!/usr/bin/env python3
"""Cross-platform bootstrap for the System Gateway native host service.

One-command setup for Linux, macOS and Windows. Run from the march7 repo
root:

    Linux:  python3 scripts/bootstrap_system_gateway.py
    macOS:  python3 scripts/bootstrap_system_gateway.py
    Windows (admin shell):  python scripts\\bootstrap_system_gateway.py

What it does (in order):
  1. Ensure a shared request secret exists — reuse SYSTEM_GATEWAY_SHARED_SECRET
     from ``.env`` if present, otherwise generate one and write it to both the
     host secret file and ``.env`` so the agent (in Docker) and the gateway
     (on host) share the same request-signing secret.
  2. Ensure a SEPARATE owner approval key exists in a host-private file
     outside the repo (0600, never printed, never written to the shared
     ``.env``). Only the gateway service, Evernight (via a dedicated mount),
     and the owner CLI may hold it; March7 must never receive it.
  3. Create the gateway virtualenv.
  4. ``pip install`` the system_gateway package into that venv.
  5. ``system-gateway install`` registers + starts the native service on
     Linux/macOS. Windows is foreground-only for now, so the script prints the
     run command instead of registering a service.
  6. Restart the service and run a health check.

Requires Python 3.11+. Idempotent — safe to re-run.
"""
from __future__ import annotations

import importlib.util
import os
import secrets
import subprocess
import sys
import venv
from pathlib import Path

HOST = os.getenv("SYSTEM_GATEWAY_HOST", "0.0.0.0")
PORT = os.getenv("SYSTEM_GATEWAY_PORT", "8380")
VENV = os.getenv(
    "SYSTEM_GATEWAY_BOOTSTRAP_VENV"
)  # if unset, default per-platform below
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
PKG_DIR = REPO_ROOT / "services" / "system_gateway"

IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


def info(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[bootstrap] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# --------------------------------------------------------------------------- #
# Privilege handling
# --------------------------------------------------------------------------- #
def is_admin() -> bool:
    if IS_WIN:
        try:
            import ctypes

            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return hasattr(os, "geteuid") and os.geteuid() == 0


def ensure_privilege() -> None:
    """Re-exec with sudo on Linux only; macOS installs a user LaunchAgent."""
    if is_admin():
        return
    if IS_WIN:
        die(
            "Windows requires an Administrator shell. Re-run from an elevated "
            "PowerShell/cmd:  python scripts\\bootstrap_system_gateway.py",
            code=2,
        )
    if IS_MAC:
        return
    # Linux: re-exec ourselves via sudo, preserving env vars we care about.
    env_keep = ",".join(
        v
        for v in (
            "SYSTEM_GATEWAY_HOST",
            "SYSTEM_GATEWAY_PORT",
            "SYSTEM_GATEWAY_BOOTSTRAP_VENV",
        )
        if os.getenv(v)
    )
    sudo_args = ["sudo", "-E"]
    if env_keep:
        sudo_args += ["--preserve-env=" + env_keep]
    sudo_args += [sys.executable, str(Path(__file__).resolve())]
    info("Not running as root; re-execing via sudo ...")
    os.execvp("sudo", sudo_args)  # no return


# --------------------------------------------------------------------------- #
# Platform defaults (unified stdlib-only resolvers, no duplication)
# --------------------------------------------------------------------------- #
_PATHS_CACHE = None


def _unified_paths():
    """Load services/system_gateway/paths.py without an installed package.

    Bootstrap runs BEFORE `pip install`, so it cannot `import system_gateway`
    (that would also pull aiohttp via __init__). Load the stdlib-only
    paths module repo-relative by file path. Reuse the already-imported
    module in tests so platform/UID mocks stay coherent across resolvers.
    """

    global _PATHS_CACHE
    if _PATHS_CACHE is not None:
        return _PATHS_CACHE
    already = sys.modules.get("system_gateway.paths")
    if already is not None:
        _PATHS_CACHE = already
        return already
    paths_file = PKG_DIR / "paths.py"
    spec = importlib.util.spec_from_file_location(
        "system_gateway_bootstrap_paths", paths_file
    )
    if spec is None or spec.loader is None:
        die(f"path helpers not found at {paths_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PATHS_CACHE = module
    return module


def default_venv() -> Path:
    if VENV:
        return Path(VENV)
    if IS_WIN:
        return Path(os.getenv("LOCALAPPDATA", "")) / "system-gateway" / "venv"
    if IS_MAC:
        return Path.home() / "Library" / "Application Support" / "system-gateway" / "venv"
    return Path("/opt/system-gateway/venv")


def config_dir() -> Path:
    return _unified_paths().default_config_dir()


def secret_file_path() -> Path:
    return _unified_paths().default_shared_secret_file()


def approval_secret_file_path() -> Path:
    """Host-private path for the owner approval key (outside repo, never .env)."""

    return _unified_paths().default_approval_secret_file()


# --------------------------------------------------------------------------- #
# .env handling
# --------------------------------------------------------------------------- #
def read_env_value(key: str) -> str | None:
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(key + "="):
            return line[len(key) + 1 :].strip().strip('"').strip("'")
    return None


def ensure_env_key(key: str, value: str) -> bool:
    """Append ``key=value`` to .env if missing. Returns True if written."""
    existing = read_env_value(key)
    if existing == value:
        return False
    if existing is not None:
        # Replace the existing line.
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
        out: list[str] = []
        for ln in lines:
            if ln.strip().startswith(key + "="):
                out.append(f"{key}={value}\n")
            else:
                out.append(ln)
        ENV_FILE.write_text("".join(out), encoding="utf-8")
        return True
    sep = "" if ENV_FILE.exists() and ENV_FILE.read_text(encoding="utf-8").endswith("\n") else "\n"
    with ENV_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{sep}{key}={value}\n")
    return True


# --------------------------------------------------------------------------- #
# Secret
# --------------------------------------------------------------------------- #
def _write_new_secret_0600(path: Path, secret: str) -> bool:
    """Exclusively create a 0600 secret file; False when it already exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(secret)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return True


def _overwrite_secret_atomic_0600(path: Path, secret: str) -> None:
    """Atomically publish a 0600 secret file (explicit overwrite only)."""

    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".tmp-secret-"
    )
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secret)
        os.replace(tmp_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


def ensure_secret() -> str:
    sf = secret_file_path()
    env_secret = read_env_value("SYSTEM_GATEWAY_SHARED_SECRET")
    file_secret = None
    if sf.exists():
        file_secret = sf.read_text(encoding="utf-8").strip() or None

    # Prefer the .env secret (owner-controlled) so the gateway matches the agent.
    secret = env_secret or file_secret
    if not secret:
        secret = secrets.token_urlsafe(32)
        info("Generated new shared secret.")

    _overwrite_secret_atomic_0600(sf, secret)

    written_env = ensure_env_key("SYSTEM_GATEWAY_SHARED_SECRET", secret)
    if written_env:
        info(f"Wrote secret to {sf} and appended SYSTEM_GATEWAY_SHARED_SECRET to {ENV_FILE}")
        info("NOTE: Restart the Docker stack (docker compose restart) so the")
        info("    agent picks up the new secret from .env.")
    else:
        info("Secret already matches between host file and .env; no change needed.")
    return secret


def ensure_approval_secret() -> Path:
    """Ensure the host-private owner approval key file exists. Returns its path.

    The key value is never printed and never written to the shared repo .env
    (which is bind-mounted into both agents). Existing files are kept unless
    empty; pass no force flag here — rotation is an explicit owner action.
    """

    sf = approval_secret_file_path()
    if sf.exists():
        existing = sf.read_text(encoding="utf-8").strip() or None
        if existing:
            info(f"Owner approval key already exists at {sf} (kept, not shown).")
            return sf
    secret = secrets.token_urlsafe(32)
    if not _write_new_secret_0600(sf, secret):
        # Lost a concurrent creation race; keep the winner, never overwrite.
        info(f"Owner approval key already exists at {sf} (kept, not shown).")
        return sf
    info(f"Generated owner approval key at {sf} (value not shown, 0600).")
    info("Mount this file read-only into Evernight only; never into March7.")
    info("Never copy this value into the shared repo .env.")
    return sf


# --------------------------------------------------------------------------- #
# venv + install
# --------------------------------------------------------------------------- #
def venv_python(venv_path: Path) -> Path:
    return venv_path / ("Scripts" if IS_WIN else "bin") / "python"


def ensure_venv() -> Path:
    vp = default_venv()
    py = venv_python(vp)
    if not (py.exists()):
        info(f"Creating virtualenv at {vp}")
        vp.parent.mkdir(parents=True, exist_ok=True)
        venv.create(vp, with_pip=True, clear=False, system_site_packages=True)
    return vp


def pip_install(venv_path: Path) -> None:
    py = venv_python(venv_path)
    info(f"Installing services/system_gateway into venv ...")
    subprocess.run(
        [str(py), "-m", "pip", "install", "--no-build-isolation", str(PKG_DIR)],
        check=True,
    )


def gateway_install(venv_path: Path) -> None:
    py = venv_python(venv_path)
    if IS_WIN:
        info("Windows background service registration is not implemented yet.")
        info(f"To run in foreground: {py} -m system_gateway run")
        info("Set up a scheduled task / NSSM if you need auto-start.")
        return
    # Pass host/port through so the rendered unit matches the .env the agent uses.
    env = os.environ.copy()
    env["SYSTEM_GATEWAY_HOST"] = HOST
    env["SYSTEM_GATEWAY_PORT"] = PORT
    info("Running `system-gateway install` ...")
    rc = subprocess.run([str(py), "-m", "system_gateway", "install"], env=env).returncode
    if rc != 0:
        die(f"`system-gateway install` exited with code {rc}")

def restart_and_health() -> None:
    if IS_LINUX:
        info("Restarting system-gateway.service ...")
        subprocess.run(["systemctl", "restart", "system-gateway"], check=False)
    elif IS_MAC:
        plist = Path.home() / "Library" / "LaunchAgents" / "com.twin.system-gateway.plist"
        if plist.exists():
            info("Reloading launchd plist ...")
            subprocess.run(["launchctl", "unload", str(plist)], check=False)
            subprocess.run(["launchctl", "load", "-w", str(plist)], check=False)
    # Windows: foreground; nothing to restart.

    # Health check.
    import urllib.request

    url = f"http://127.0.0.1:{PORT}/health"
    info(f"Health check: GET {url}")
    for attempt in range(1, 11):
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                body = resp.read().decode("utf-8", "replace")
            info(f"Gateway healthy (attempt {attempt}): {body.strip()[:120]}")
            return
        except Exception as exc:
            if attempt == 10:
                info(f"ERROR: Health check failed after 10 attempts: {exc}")
                if IS_LINUX:
                    info("Inspect logs:  sudo journalctl -u system-gateway -n 50 --no-pager")
                return
            import time

            time.sleep(1.0)


def main() -> int:
    if not PKG_DIR.exists():
        die(f"system_gateway package not found at {PKG_DIR} (run from repo root)")
    info(f"Repo root: {REPO_ROOT}")
    _plat = "linux" if IS_LINUX else "mac" if IS_MAC else "win" if IS_WIN else "unknown"
    info(f"Platform: {_plat}")
    ensure_privilege()
    info(f"Host={HOST} Port={PORT}")
    info(f"Venv: {default_venv()}")
    info(f"Secret file: {secret_file_path()}")
    info(f"Approval key file: {approval_secret_file_path()}")
    ensure_secret()
    ensure_approval_secret()
    vp = ensure_venv()
    pip_install(vp)
    gateway_install(vp)
    restart_and_health()
    info("Done. If the Docker stack was running, restart it to pick up the secret.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
