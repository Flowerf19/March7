# System Gateway Windows Packaging

System Gateway is a Python-native service. Windows service support is not yet
implemented; the current recommended approach is to run it in the foreground
inside a scheduled task or a third-party service wrapper.

## Manual install

1. Install the package:

   ```powershell
   python -m pip install -e services\system_gateway
   ```

2. Generate secrets once (request secret + separate owner approval key):

   ```powershell
   python -m system_gateway pair --secret-file %LOCALAPPDATA%\system-gateway\secret
   ```

   `pair` also creates the owner approval key at `%LOCALAPPDATA%\system-gateway\approval_secret`
   for user runs (`%ProgramData%\system-gateway\approval_secret` for the admin
   service install; override with `--approval-secret-file`). Its value is never
   printed and must never be copied into the shared repo `.env`.

3. Set the environment variables before starting the service:

   ```powershell
   $env:SYSTEM_GATEWAY_SHARED_SECRET_FILE = "$env:LOCALAPPDATA\system-gateway\secret"
   $env:SYSTEM_GATEWAY_APPROVAL_SECRET_FILE = "$env:LOCALAPPDATA\system-gateway\approval_secret"
   $env:SYSTEM_GATEWAY_HOST = "127.0.0.1"
   $env:SYSTEM_GATEWAY_PORT = "8380"
   $env:SYSTEM_GATEWAY_RAW_SHELL = "true"
   ```

   Raw shell is denied by default and requires explicit opt-in.

4. Run the gateway:

   ```powershell
   python -m system_gateway run
   ```

## Running as a Windows service

Use a service wrapper such as `NSSM` or `WinSW`, or implement a
`pywin32` service harness in `services/system_gateway/packaging/windows/`
as a follow-up. The harness should:

* set `SYSTEM_GATEWAY_SHARED_SECRET_FILE` to a protected path,
* bind to `127.0.0.1:8380`,
* forward stdout/stderr to Event Log or a log file,
* refuse to start if the secret file is missing or world-readable.

## Security notes

* Do not store `SYSTEM_GATEWAY_SHARED_SECRET` directly in the Windows Registry
  or in a world-readable scheduled-task XML file.
* Restrict the secret file to the account that runs the service.
* The gateway binds to localhost by default; do not expose it to the network.
