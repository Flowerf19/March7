"""EvernightAgent - consolidation + chat agent."""
import logging
from typing import Any, List, Optional

from twin.shared.agent import ChatTurnRunner
from twin.shared.llm.base_llm_service import BaseLLMService
from twin.shared.observability import call_with_langsmith_extra, langsmith_extra
from twin.shared.observability.langsmith import summarize_trace_output, traceable
from twin.shared.tools.registry import ToolRegistry
from twin.shared.a2a.types import AgentCard
from twin.shared.memory import SharedMemoryManager

logger = logging.getLogger(__name__)

TOOL_EXECUTION_TIMEOUT = 60

# Shown to the user when the LLM endpoint fails — keeps the raw
# "Error generating response." sentinel from leaking into the chat.
LLM_FAILURE_REPLY = "Xin lỗi, tôi đang gặp chút trục trặc với hệ thống ký ức."

class EvernightAgent:
    def __init__(
        self,
        memory_manager: SharedMemoryManager | None = None,
        episodic_memory: Any = None,
        llm_service: BaseLLMService = None,
        tool_registry: Optional[ToolRegistry] = None,
        use_native_tools: bool = True,
        march7_url: str = "http://march7:8000",
        consolidator: Any = None,
        **kwargs,
    ):
        self.memory = memory_manager
        self.episodic = episodic_memory or self.memory
        self.llm = llm_service or kwargs.pop("llm_client", None)
        if self.llm is None:
            raise ValueError("llm_service or llm_client must be provided")
        self._embedding_service = kwargs.pop("embedding_service", None)
        self.tool_registry = tool_registry
        self.use_native_tools = use_native_tools
        self.march7_url = march7_url
        self.consolidator = consolidator

        self._chat_turn = ChatTurnRunner(
            llm=self.llm,
            tool_registry=self.tool_registry,
            use_native_tools=self.use_native_tools,
            logger=logger,
        )
        self._llm_type = self._chat_turn.llm_type
        self._model_name = getattr(self.llm, "model", "unknown") if self.llm else "unknown"

        logger.debug(f"EvernightAgent initialized: model={self._model_name}")

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Evernight",
            description="Memory consolidation and analysis agent",
            url="http://evernight:8001",
            version="1.0.0",
            capabilities=["chat", "consolidation", "streaming"],
            skills=[
                {"id": "chat", "name": "Chat", "description": "Conversational chat with memory and tools"},
                {"id": "consolidate", "name": "Consolidate", "description": "Consolidate T1 snapshot into T2 timeline"},
                {"id": "consolidate_discussion", "name": "Consolidate Discussion", "description": "Process shared-memory payload into user-centric T2 timeline entries"},
                {"id": "get_snapshot", "name": "Get Snapshot", "description": "Get T1 memory snapshot"},
            ],
        )

    async def get_status(self) -> dict:
        return {"status": "online", "model": self._model_name}

    # ------------------------------------------------------------------
    # Consolidate
    # ------------------------------------------------------------------

    async def consolidate(self, user_id: str, snapshot: List[dict], reason: str = "manual") -> bool:
        logger.info(f"Evernight: Consolidating snapshot for user {user_id}")
        try:
            if not self.consolidator or not hasattr(self.consolidator, "consolidate_snapshot"):
                logger.error("Evernight: Missing shared consolidator")
                return False
            return await self.consolidator.consolidate_snapshot(user_id, snapshot, reason)
        except Exception as e:
            logger.error(f"Evernight: Consolidation failed: {e}", exc_info=True)
            return False

    async def consolidate_via_tool(self, scope: str = "user", scope_id: str = "", reason: str = "manual", max_messages: int = 200, entries: list[dict] | None = None) -> dict:
        """
        New consolidation flow: use consolidate_memory tool directly.

        This replaces the old pipeline (Extractor → PromotionGuard → Cleanup → Curator)
        with a single LLM call via the Summarizer prompt.

        When ``entries`` are shipped over A2A, the tool consolidates them directly
        instead of reading Evernight's own (empty for the requester's scopes) T1.

        Returns: dict with status, timeline_summary, profile_updates, etc.
        """
        logger.info(f"Evernight: Consolidating via tool for scope={scope}/{scope_id}, reason={reason}")

        if not self.tool_registry:
            logger.error("Evernight: Tool registry not available")
            return {"status": "failed", "error": "tool_registry_not_available"}

        try:
            # Get the consolidate_memory tool from registry
            tool = self.tool_registry.get_tool("consolidate_memory")
            if not tool:
                logger.error("Evernight: consolidate_memory tool not found in registry")
                return {"status": "failed", "error": "tool_not_found"}

            # Execute the tool
            result_str = await call_with_langsmith_extra(
                tool.execute,
                scope=scope,
                scope_id=scope_id,
                reason=reason,
                max_messages=max_messages,
                entries=entries,
                langsmith_extra=langsmith_extra(
                    tags=["evernight", "memory", "consolidation"],
                    metadata={
                        "workflow": "evernight.memory_consolidation",
                        "agent_name": "evernight",
                        "provider": self._llm_type,
                        "model": self._model_name,
                        "scope": scope,
                        "scope_id": scope_id,
                        "reason": reason,
                        "max_messages": max_messages,
                    },
                ),
            )

            # Parse JSON result
            import json
            result = json.loads(result_str)

            logger.info(
                f"Evernight: Consolidation completed for scope={scope}/{scope_id}, "
                f"status={result.get('status')}, "
                f"messages={result.get('messages_summarized', 0)}"
            )

            return result

        except Exception as e:
            logger.error(f"Evernight: Consolidation via tool failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def _format_snapshot(self, snapshot: List[dict]) -> str:
        lines = []
        for msg in snapshot:
            role = msg.get("role", "unknown") if isinstance(msg, dict) else getattr(msg, "role", "unknown")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            lines.append(f"[{role.upper()}]: {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Chat (new capability for Evernight)
    # ------------------------------------------------------------------

    @traceable(
        name="evernight.chat",
        run_type="chain",
        tags=["evernight"],
        process_outputs=summarize_trace_output,
    )
    async def handle_chat(
        self, user_id: str, content: str, *, observe_input: bool = True
    ) -> str:
        try:
            if observe_input:
                await self.memory.add_message(user_id=user_id, role="user", content=content)

            sys_prompt, context_msgs = await self.memory.get_context(
                user_id=user_id, current_query=content
            )

            turn = await self._chat_turn.run(
                messages=context_msgs,
                system_prompt=sys_prompt,
                max_iterations=10,
                tool_timeout=TOOL_EXECUTION_TIMEOUT,
                trace_metadata={
                    "workflow": "evernight.chat",
                    "agent_name": "evernight",
                    "provider": self._llm_type,
                    "model": self._model_name,
                    "user_id": user_id,
                },
            )

            bot_response = turn.content

            # LLM hard-failure sentinel: never relay it to the user or persist it
            # to memory — surface a friendly retry message instead.
            if turn.is_failure:
                logger.error("LLM returned error sentinel for user=%s: %s", user_id, bot_response)
                return LLM_FAILURE_REPLY

            if bot_response and not bot_response.startswith("Error:") and not turn.reasoning_only:
                await self.memory.add_message(
                    user_id=user_id, role="assistant", content=bot_response
                )

            return bot_response

        except Exception as e:
            logger.error(f"EvernightAgent chat error: {e}")
            return LLM_FAILURE_REPLY

    async def clear_chat_history(self, user_id: str):
        await self.memory.clear_session(user_id)
