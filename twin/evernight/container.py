"""EvernightContainer - DI container for shared memory runtime."""
from __future__ import annotations

import logging

from dotenv import load_dotenv

from twin.evernight.agent import EvernightAgent
from twin.evernight.config import EvernightConfig
from twin.shared.agent.runtime import SharedAgentRuntime, build_shared_agent_runtime
from twin.shared.tools.registry.bootstrap import build_tool_registry
from twin.shared.memory.diary import TimelineSummaryStore
from twin.shared.config.settings import Config
from twin.evernight.host_gateway import GatewayMonitor

logger = logging.getLogger(__name__)


class EvernightContainer:
    _instance = None

    def __init__(self, config: EvernightConfig):
        self.config = config
        self.runtime: SharedAgentRuntime | None = None
        self.llm_service = None
        self.embedding_service = None
        self.memory_manager = None
        self.agent = None
        self.tool_registry = None
        self.redis_client = None
        self.timeline_redis_client = None
        self.timeline_summary_store = None
        self.profile_store = None
        self.state_repo = None
        self.summary_policy = None
        self.gateway_monitor = None

    @classmethod
    def get_instance(cls, config: EvernightConfig = None):
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    async def initialize(self):
        load_dotenv(override=True)
        logger.info("EvernightContainer initializing...")

        self.runtime = await build_shared_agent_runtime(
            redis_db=self.config.redis_db,
            persona_path=self.config.persona_path,
        )
        self.llm_service = self.runtime.llm_service
        self.embedding_service = self.runtime.embedding_service
        self.memory_manager = self.runtime.memory_manager
        self.redis_client = self.runtime.redis_client
        self.timeline_redis_client = self.runtime.timeline_redis_client
        self.timeline_summary_store = self.runtime.timeline_summary_store
        self.profile_store = self.runtime.profile_store
        self.state_repo = self.runtime.state_repo
        self.summary_policy = self.runtime.summary_policy

        # Create GatewayMonitor if SYSTEM_GATEWAY_URL is configured
        gateway_url = getattr(Config, "SYSTEM_GATEWAY_URL", "")
        if gateway_url:
            self.gateway_monitor = GatewayMonitor(
                base_url=gateway_url,
                timeout=Config.SYSTEM_GATEWAY_TIMEOUT,
                shared_secret=Config.SYSTEM_GATEWAY_SHARED_SECRET,
            )
            await self.gateway_monitor.start()
            logger.info("GatewayMonitor started: %s", gateway_url)

        tools = build_tool_registry(
            agent_name="evernight",
            core_manager=None,
            memory_manager=self.memory_manager,
            profile_store=self.profile_store,
            llm_service=self.llm_service,
            base_memory_path=self.config.persona_path,
            embedding_service=self.embedding_service,
            timeline_summary_store=self.timeline_summary_store,
            owner_user_id=self.config.owner_user_id,
            gateway_monitor=self.gateway_monitor,
            # Production self-DM: gateway_admin update requests its grant
            # through this live gate (actor evernight, self endpoint).
            use_evernight_dm_approval=True,
        )
        self.tool_registry = tools.registry
        self.llm_service.set_tool_registry(self.tool_registry)
        self.llm_service.set_tool_prompt_catalog(tools.tool_prompt_catalog)

        self.agent = EvernightAgent(
            memory_manager=self.memory_manager,
            episodic_memory=self.memory_manager,
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            march7_url=self.config.march7_url,
            consolidator=self.memory_manager,
        )

        # Evernight is the consolidation worker — its auto-trigger must consolidate
        # LOCALLY (it has no remote ConsolidationClient to call). Wire the manager's
        # consolidate_scope to fall back to the local consolidate_memory tool.
        self.memory_manager.local_consolidator = self.agent.consolidate_via_tool

        logger.info("EvernightContainer initialized")

    async def shutdown(self):
        if self.gateway_monitor is not None:
            await self.gateway_monitor.stop()
            logger.info("GatewayMonitor stopped")
        if self.runtime:
            await self.runtime.close()
