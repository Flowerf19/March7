"""Core gateway orchestrator.

Manages the lifecycle of platform adapters, routes messages between adapters
and handlers, and provides error isolation so that one adapter crashing does
not affect others.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.shared.adapter_base import PlatformAdapter
    from gateway.shared.handler_base import GatewayHandler
    from gateway.shared.model import UnifiedMessage, UnifiedEvent

logger = logging.getLogger(__name__)

# Reconnection settings
MAX_RECONNECT_RETRIES = 5
RECONNECT_BASE_DELAY = 2.0  # seconds


class ChatGateway:
    """Central orchestrator that wires platform adapters to a handler."""

    def __init__(self, handler: GatewayHandler) -> None:
        self._handler = handler
        self._adapters: dict[str, PlatformAdapter] = {}
        self._reconnect_tasks: dict[str, asyncio.Task] = {}
        self._running = False

    # ------------------------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------------------------

    def register_adapter(self, name: str, adapter: PlatformAdapter) -> None:
        """Register *adapter* under the given *name* (e.g. ``"discord"``)."""
        self._adapters[name] = adapter
        logger.info("Registered platform adapter: %s", name)

    def unregister_adapter(self, name: str) -> None:
        """Remove and disconnect the adapter identified by *name*."""
        adapter = self._adapters.pop(name, None)
        if adapter:
            self._cancel_reconnect(name)
            asyncio.create_task(self._safe_disconnect(name, adapter))
            logger.info("Unregistered platform adapter: %s", name)

    def _cancel_reconnect(self, name: str) -> None:
        """Cancel a tracked reconnect loop so it cannot resurrect *name*."""
        task = self._reconnect_tasks.pop(name, None)
        if task is not None and not task.done():
            task.cancel()

    async def _safe_disconnect(
        self, name: str, adapter: PlatformAdapter
    ) -> None:
        try:
            await adapter.disconnect()
        except Exception:
            logger.exception(
                "Error disconnecting adapter %s — ignored", name
            )

    async def start_all(self) -> None:
        """Connect every registered adapter.

        Each adapter connection is isolated — one failure does not stop the
        others from starting.
        """
        self._running = True
        for name, adapter in self._adapters.items():
            try:
                logger.info("Connecting adapter: %s …", name)
                await adapter.connect()
                logger.info("Adapter connected: %s", name)
            except Exception:
                logger.exception(
                    "Failed to connect adapter %s — will attempt reconnect",
                    name,
                )
                # Start reconnection loop in background.
                self._cancel_reconnect(name)
                self._reconnect_tasks[name] = asyncio.create_task(
                    self._reconnect_loop(name, adapter)
                )

    async def stop_all(self) -> None:
        """Disconnect all adapters gracefully."""
        self._running = False
        # Cancel reconnect loops first so none can reconnect after disconnect.
        tasks = list(self._reconnect_tasks.values())
        self._reconnect_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for name, adapter in self._adapters.items():
            try:
                await adapter.disconnect()
                logger.info("Disconnected adapter: %s", name)
            except Exception:
                logger.exception("Error disconnecting adapter %s", name)

    async def _reconnect_loop(
        self, name: str, adapter: PlatformAdapter
    ) -> None:
        """Exponential-backoff reconnection for a single adapter."""
        current = asyncio.current_task()
        try:
            await self._reconnect_loop_inner(name, adapter)
        finally:
            if self._reconnect_tasks.get(name) is current:
                self._reconnect_tasks.pop(name, None)

    async def _reconnect_loop_inner(
        self, name: str, adapter: PlatformAdapter
    ) -> None:
        attempt = 0
        while self._running and attempt < MAX_RECONNECT_RETRIES:
            delay = RECONNECT_BASE_DELAY * (2 ** attempt)
            logger.info(
                "Reconnecting adapter %s in %.1fs (attempt %d/%d)",
                name, delay, attempt + 1, MAX_RECONNECT_RETRIES,
            )
            await asyncio.sleep(delay)
            # stop_all/unregister may have run during the backoff sleep.
            if not self._running:
                return
            if self._adapters.get(name) is not adapter:
                return
            try:
                await adapter.connect()
            except RuntimeError as e:
                # Config errors (missing token, etc.) — won't be fixed by retrying.
                logger.error(
                    "Reconnect failed for adapter %s (config error, not retrying): %s",
                    name, e,
                )
                return
            except Exception as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in (
                    "name resolution", "gaierror", "connectordnserror",
                    "connection refused", "network is unreachable",
                    "temporary failure",
                )):
                    logger.warning(
                        "Reconnect attempt %d failed for %s (network issue): %s",
                        attempt + 1, name, e,
                    )
                else:
                    logger.exception(
                        "Reconnect attempt %d failed for adapter %s",
                        attempt + 1, name,
                    )
                attempt += 1
                continue
            # connect() awaited through a stop/unregister boundary — recheck
            # before treating this as a live link.
            if not self._running or self._adapters.get(name) is not adapter:
                try:
                    await adapter.disconnect()
                except Exception:
                    logger.exception(
                        "Error disconnecting stale reconnect for adapter %s", name
                    )
                return
            logger.info("Adapter %s reconnected on attempt %d", name, attempt + 1)
            return

        if self._running:
            logger.error(
                "Adapter %s failed to reconnect after %d attempts — giving up",
                name, MAX_RECONNECT_RETRIES,
            )

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    async def route_message(self, platform_name: str, msg: UnifiedMessage) -> None:
        """Route a unified message from *platform_name* through the handler.

        The handler produces a response string, which is sent back through
        the same adapter.  Errors are caught and logged, except for explicit
        platform-capability control-flow exceptions that adapters can render
        with native UI.
        """
        adapter = self._adapters.get(platform_name)
        if adapter is None:
            logger.warning(
                "No adapter registered for platform '%s' — dropping message",
                platform_name,
            )
            return

        try:
            response_text = await self._handler.handle_message(msg)
            if response_text and response_text.strip():
                # Build a reply message pointing back to the original channel.
                from gateway.shared.model import (
                    UnifiedMessage,
                    UnifiedUser,
                    UnifiedChannel,
                )
                from datetime import datetime, timezone

                reply_msg = UnifiedMessage(
                    message_id=f"reply-{msg.message_id}",
                    user=UnifiedUser(
                        platform_id="gateway",
                        platform_name=platform_name,
                        display_name="Gateway",
                        is_bot=True,
                    ),
                    channel=msg.channel,
                    content=response_text,
                    timestamp=datetime.now(timezone.utc),
                )
                await adapter.send_message(reply_msg)
        except Exception:
            logger.exception(
                "Error routing message from platform %s", platform_name
            )

    async def route_event(
        self, platform_name: str, event: UnifiedEvent, msg: UnifiedMessage
    ) -> None:
        """Route a non-message event through the handler."""
        try:
            await self._handler.handle_event(event, msg)
        except Exception:
            logger.exception(
                "Error routing event %s from platform %s", event, platform_name
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def adapter_names(self) -> list[str]:
        """Return the list of registered adapter names."""
        return list(self._adapters.keys())

    def get_adapter(self, name: str) -> PlatformAdapter | None:
        """Return the adapter for *name*, or ``None``."""
        return self._adapters.get(name)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_healthy(self) -> bool:
        """Whether every registered adapter still has a live platform link.

        The A2A server answers even when a bot is disconnected, so liveness of
        the process alone is not a usable health signal.
        """
        if not self._adapters:
            return False
        return all(adapter.is_connected for adapter in self._adapters.values())
