"""Self-heal monitor for Evernight → March7."""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import aiohttp

from twin.evernight.host_gateway.monitor import RESTART_ALLOWED_CONTAINERS

logger = logging.getLogger(__name__)


class RecoveryExecutor(ABC):
    @abstractmethod
    async def restart_container(self, container_name: str) -> tuple[bool, str]:
        """Restart a container and return (success, detail)."""


class DenyRecoveryExecutor(RecoveryExecutor):
    """Fail-closed default when no authorized recovery path is configured."""

    def __init__(self, reason: str = "recovery unavailable"):
        self._reason = reason

    async def restart_container(self, container_name: str) -> tuple[bool, str]:
        return False, self._reason


def _resolve_dm_backend(discord_adapter: object):
    """Build a Discord DM backend from the adapter's bot, if it is ready."""
    bot = getattr(discord_adapter, "bot", None)
    if bot is None:
        return None
    try:
        if not bot.is_ready():
            return None
    except Exception:
        return None
    from gateway.adapters.discord.dm_delivery import DiscordDMDelivery

    return DiscordDMDelivery(bot)


class OwnerApprovalRecoveryExecutor(RecoveryExecutor):
    """Restart containers only after a real owner approval.

    The recovery shell (`docker restart <name>`) goes through the same
    owner DM approval + issuer grant as any other host command, then through
    the native gateway. No owner, no approval, or no DM path means no
    restart. There is no direct Docker/subprocess side channel.
    """

    def __init__(
        self,
        *,
        gateway_monitor,
        owner_user_id: int | str | None = None,
        dm_backend=None,
        discord_adapter=None,
        dm_timeout: float = 60.0,
    ):
        self.gateway_monitor = gateway_monitor
        self._owner_user_id = owner_user_id
        self._dm_backend = dm_backend
        self._discord_adapter = discord_adapter
        self._dm_timeout = dm_timeout

    async def restart_container(self, container_name: str) -> tuple[bool, str]:
        from twin.evernight.server.approval_issuer import approve_local_shell

        if self.gateway_monitor is None:
            return False, "GatewayMonitor not available"
        if container_name not in RESTART_ALLOWED_CONTAINERS:
            return False, f"container {container_name!r} not in allowed list"

        backend = self._dm_backend or _resolve_dm_backend(self._discord_adapter)
        command = f"docker restart {container_name}"
        # Bind the exact timeout the monitor will send to the gateway.
        timeout = getattr(self.gateway_monitor, "timeout", 10)
        decision = await approve_local_shell(
            backend=backend,
            owner_user_id=self._owner_user_id,
            command=command,
            timeout=timeout,
            channel_name="self-heal",
            dm_timeout=self._dm_timeout,
        )
        if not decision.approved or not decision.grant:
            return False, f"restart denied: {decision.reason}"
        return await self.gateway_monitor.request_container_restart(
            container_name, approval_id=decision.grant
        )


class SelfHealMonitor:
    """Polls March7 health endpoint and restarts it on repeated failures."""

    def __init__(
        self,
        march7_url: str = "http://march7:8000",
        interval: int = 30,
        timeout: int = 10,
        failure_threshold: int = 3,
        container_name: str = "march7",
        discord_adapter=None,
        notify_user_id: int | str | None = None,
        recovery_executor: RecoveryExecutor | None = None,
        gateway_monitor=None,
    ):
        self.march7_url = march7_url.rstrip("/")
        self.interval = interval
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.container_name = container_name
        self._discord_adapter = discord_adapter
        self._notify_user_id = notify_user_id
        self._gateway_monitor = gateway_monitor

        if recovery_executor is not None:
            self._recovery_executor = recovery_executor
        elif gateway_monitor is not None:
            self._recovery_executor = OwnerApprovalRecoveryExecutor(
                gateway_monitor=gateway_monitor,
                owner_user_id=notify_user_id,
                discord_adapter=discord_adapter,
            )
        else:
            self._recovery_executor = DenyRecoveryExecutor(
                "GatewayMonitor not available"
            )

        self._failure_count = 0
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start the monitoring loop."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Self-heal monitor started, polling %s every %ds", self.march7_url, self.interval)

    async def stop(self):
        """Stop the monitoring loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Self-heal monitor stopped")

    async def _poll_loop(self):
        while self._running:
            try:
                healthy = await self._check_health()
                if healthy:
                    self._failure_count = 0
                else:
                    self._failure_count += 1
                    logger.warning(
                        "March7 health check failed (%d/%d consecutive failures)",
                        self._failure_count,
                        self.failure_threshold,
                    )
                    if self._failure_count >= self.failure_threshold:
                        await self._restart_march7()
                        self._failure_count = 0
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Self-heal monitor error in poll loop")

            await asyncio.sleep(self.interval)

    async def _check_health(self) -> bool:
        """Check March7 health via its /health endpoint.

        The agent card answers 200 whenever the A2A server is up, even with
        Discord disconnected, so it must never be used as a health signal.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.march7_url}/health") as resp:
                    return resp.status == 200
        except Exception:
            logger.debug("March7 health check unreachable at %s", self.march7_url)
            return False

    async def _restart_march7(self):
        """Restart the March7 container through the authorized recovery path."""
        logger.warning("Restarting March7 container: %s", self.container_name)
        success, detail = await self._recovery_executor.restart_container(self.container_name)
        if success:
            logger.info("March7 container restarted successfully: %s", detail)
            await self._notify_restart()
        else:
            logger.error("March7 restart failed: %s", detail)

    async def _notify_restart(self):
        """Notify users about the restart via Discord DM."""
        if self._discord_adapter is None or self._notify_user_id is None:
            logger.info("March7 container restarted (no Discord adapter for notification)")
            return

        try:
            await self._discord_adapter.send_dm(
                user_id=self._notify_user_id,
                content=f"⚠️ **March7 tự khởi động lại:** Container `{self.container_name}` đã được self-heal monitor tự động restart sau {self.failure_threshold} lần health check thất bại.",
            )
            logger.info("Restart notification sent to user %s", self._notify_user_id)
        except Exception:
            logger.exception("Failed to send restart notification")
