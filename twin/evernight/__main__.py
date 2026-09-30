"""Entry point for Evernight Agent."""
import asyncio
import logging
import signal
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

from twin.shared.config.logging_config import setup_logging

setup_logging()

logger = logging.getLogger("evernight.main")


def _notify_user_id(owner_user_id: str | int | None) -> int | None:
    """Parse owner for self-heal DM; unknown/malformed fails closed (None)."""
    if owner_user_id is None or isinstance(owner_user_id, bool):
        return None
    if isinstance(owner_user_id, int):
        return owner_user_id if owner_user_id >= 0 else None
    text = str(owner_user_id).strip()
    return int(text) if text.isdigit() else None


async def main():
    from twin.evernight.config import EvernightConfig
    from twin.evernight.container import EvernightContainer
    from twin.evernight.triggers.inactivity_trigger import InactivityTrigger
    from twin.evernight.self_heal.monitor import SelfHealMonitor

    config = EvernightConfig.from_env()
    container = EvernightContainer(config)
    await container.initialize()

    # Start Evernight Discord bot first if token is set (needed for approval DM endpoint)
    evernight_adapter = None
    discord_token = config.discord_evernight_token
    if discord_token:
        from gateway.adapters.discord.evernight_adapter import EvernightDiscordAdapter
        evernight_adapter = EvernightDiscordAdapter(
            token=discord_token,
            agent=container.agent,
            owner_user_id=config.owner_user_id,
        )
        await evernight_adapter.connect()
        logger.info("Evernight Discord bot started")

    # Start A2A server with Discord bot reference for approval DM support
    from twin.evernight.server.a2a_server import start_server
    discord_bot = evernight_adapter.bot if evernight_adapter else None
    server = start_server(
        container.agent,
        port=config.port,
        discord_bot=discord_bot,
        owner_user_id=config.owner_user_id,
    )
    await server.start()
    logger.info(f"Evernight Agent listening on port {config.port}")

    # Start self-healing monitor
    self_heal = None
    if config.self_heal_enabled:
        self_heal = SelfHealMonitor(
            march7_url=config.march7_url,
            interval=config.self_heal_interval,
            timeout=config.self_heal_timeout,
            discord_adapter=evernight_adapter,
            notify_user_id=_notify_user_id(config.owner_user_id),
            gateway_monitor=container.gateway_monitor,
        )
        await self_heal.start()
        logger.info(f"Self-heal monitor started (interval={config.self_heal_interval}s)")

    # InactivityTrigger over Evernight's own SummaryStateRepository
    # (March7 channel scopes are polled by March7's own InactivityTrigger).
    trigger = None
    if container.state_repo is not None and container.summary_policy is not None:
        trigger = InactivityTrigger(
            state_repo=container.state_repo,
            summary_policy=container.summary_policy,
            scopes=("user",),
            poll_interval=config.poll_interval,
        )
        await trigger.start()
    else:
        logger.warning("State repo / SummaryPolicy not available - inactivity trigger disabled")

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler():
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await shutdown_event.wait()

    # Cleanup in reverse order
    if evernight_adapter:
        await evernight_adapter.disconnect()
    if self_heal:
        await self_heal.stop()
    if trigger:
        await trigger.stop()
    await server.stop()
    await container.shutdown()
    logger.info("Evernight Agent shut down")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Evernight crashed")
        sys.exit(1)
