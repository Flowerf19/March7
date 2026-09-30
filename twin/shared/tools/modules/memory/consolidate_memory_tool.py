"""ConsolidateMemoryTool - consolidate T1 active memory into T2 timeline + T3 profile."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from twin.shared.agent.contract import extract_json
from twin.shared.config.settings import Config
from twin.shared.memory.consolidation_journal import (
    LocalBatchError,
    read_local_tool_batch,
)
from twin.shared.memory.profile.codec import profile_hash
from twin.shared.memory.vn_time import vn_now
from twin.shared.observability import call_with_langsmith_extra, langsmith_extra
from twin.shared.observability.langsmith import traceable
from twin.shared.tools.modules.memory.consolidation_prompt import (
    ConsolidationPromptBuilder,
    SUMMARIZER_PROMPT,
    entry_epoch,
)
from twin.shared.tools.modules.memory.consolidation_plan_cache import (
    CanonicalPlanResolver,
    build_profile_recompute_prompt,
)
from twin.shared.tools.modules.memory.consolidation_schema import (
    ConsolidationPlanValidator,
    PlanError,
)
from twin.shared.tools.modules.memory.consolidation_store import (
    ConsolidationProfileWriter,
    ConsolidationWriter,
)
from twin.shared.tools.registry.base import BaseTool, ToolExecutionError

logger = logging.getLogger(__name__)

_SUMMARIZER_PROMPT = SUMMARIZER_PROMPT


class ConsolidateMemoryTool(BaseTool):
    """Consolidate T1 messages into T2 timeline + T3 profile."""

    def __init__(
        self,
        memory_manager: Any,
        llm_service: Any,
        embedding_service: Any,
        timeline_summary_store: Any,
    ):
        self.memory_manager = memory_manager
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.timeline_summary_store = timeline_summary_store

    @property
    def name(self) -> str:
        return "consolidate_memory"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "Scope của consolidation (user hoặc channel)",
                    "default": "user",
                },
                "scope_id": {
                    "type": "string",
                    "description": "ID của scope cần consolidate",
                },
                "reason": {
                    "type": "string",
                    "description": "Lý do consolidate",
                },
                "max_messages": {
                    "type": "integer",
                    "default": 200,
                    "description": "Số messages tối đa cần tóm tắt",
                },
                "entries": {
                    "type": "array",
                    "description": "Danh sách entries được ship qua A2A để consolidate trực tiếp (không đọc T1 local)",
                    "items": {"type": "object"},
                },
            },
            "required": ["scope_id", "reason"],
        }

    @property
    def visible_to_agents(self) -> Optional[set[str]]:
        return {"evernight"}

    @traceable(
        name="memory.consolidate",
        run_type="chain",
        tags=["memory", "consolidation", "evernight"],
    )
    async def execute(
        self,
        scope: str,
        scope_id: str,
        reason: str,
        max_messages: int = 200,
        entries: list[dict] | None = None,
    ) -> str:
        scope_id = str(scope_id or "").strip()
        scope = str(scope or "user").strip()
        if not scope_id:
            return "Lỗi: Thiếu scope_id."
        if entries is not None and not isinstance(entries, list):
            return json.dumps({"status": "failed", "reason": "invalid_entries"})
        logger.info(
            "ConsolidateMemoryTool: scope=%s scope_id=%s reason=%s max_messages=%d",
            scope, scope_id, reason, max_messages,
        )
        trace_base = {
            "workflow": "evernight.memory_consolidation",
            "workflow_step": "memory.consolidate",
            "agent_name": "evernight",
            "provider": self._provider_name(),
            "model": self._model_name(),
            "scope": scope,
            "scope_id": scope_id,
            "reason": reason,
            "max_messages": max_messages,
        }
        try:
            if entries is None:
                t1_entries = await self._read_t1_context(
                    scope, scope_id, max_messages, trace_base,
                    langsmith_extra=langsmith_extra(
                        tags=["memory", "t1", "read"],
                        metadata={**trace_base, "workflow_step": "memory.t1_read"},
                    ),
                )
                entry_ids = [e.entry_id for e in t1_entries]
            else:
                t1_entries = await self._shipped_entries_as_t1(
                    entries, trace_base,
                    langsmith_extra=langsmith_extra(
                        tags=["memory", "t1", "read", "shipped"],
                        metadata={
                            **trace_base, "workflow_step": "memory.t1_read",
                            "source": "shipped_entries", "count": len(entries),
                        },
                    ),
                )
                entry_ids = [str(d.get("entry_id")) for d in t1_entries if d.get("entry_id")]
            if not t1_entries:
                return json.dumps({
                    "status": "skipped", "reason": "no_messages",
                    "messages_summarized": 0, "entry_ids": [],
                })
            messages_text = ConsolidationPromptBuilder.format_messages(t1_entries)
            profile_text = ""
            expected_hash: str | None = None
            profile_read_failed = False
            if scope != "channel":
                try:
                    profile_text = await self._read_profile(
                        scope_id, trace_base,
                        langsmith_extra=langsmith_extra(
                            tags=["memory", "t3", "read"],
                            metadata={**trace_base, "workflow_step": "memory.t3_profile_read"},
                        ),
                    )
                    expected_hash = profile_hash(profile_text)
                except Exception as exc:
                    profile_read_failed = True
                    logger.debug("ConsolidateMemoryTool: no profile for scope_id=%s: %s", scope_id, exc)
            resolver = CanonicalPlanResolver.from_store(self.timeline_summary_store)

            async def fetch_full_plan():
                prompt = ConsolidationPromptBuilder.build(
                    vn_now().strftime("%H:%M %d/%m/%Y"), profile_text, messages_text,
                )
                response = await call_with_langsmith_extra(
                    self.llm_service.generate_response,
                    messages=[{"role": "user", "content": prompt}],
                    include_tool_catalog=False,
                    include_persona=False,
                    max_tokens=Config.LLM_CONSOLIDATION_MAX_TOKENS,
                    reasoning_effort=Config.LLM_CONSOLIDATION_REASONING_EFFORT,
                    langsmith_extra=langsmith_extra(
                        tags=["memory", "consolidation", "summarizer", "llm"],
                        metadata={**trace_base, "workflow_step": "memory.summarizer"},
                    ),
                )
                content = getattr(response, "content", None) or str(response)
                try:
                    data = json.loads(extract_json(content))
                except json.JSONDecodeError as exc:
                    return None, PlanError("parse_failed", str(exc))
                return ConsolidationPlanValidator.validate(data)

            async def fetch_profile_patch(topics_json):
                prompt = build_profile_recompute_prompt(profile_text, messages_text)
                response = await call_with_langsmith_extra(
                    self.llm_service.generate_response,
                    messages=[{"role": "user", "content": prompt}],
                    include_tool_catalog=False,
                    include_persona=False,
                    max_tokens=Config.LLM_CONSOLIDATION_MAX_TOKENS,
                    reasoning_effort=Config.LLM_CONSOLIDATION_REASONING_EFFORT,
                    langsmith_extra=langsmith_extra(
                        tags=["memory", "consolidation", "profiler", "llm"],
                        metadata={**trace_base, "workflow_step": "memory.profile_recompute"},
                    ),
                )
                content = getattr(response, "content", None) or str(response)
                try:
                    data = json.loads(extract_json(content))
                except json.JSONDecodeError as exc:
                    return {}, {}, PlanError("parse_failed", str(exc))
                if not isinstance(data, dict):
                    return {}, {}, PlanError("invalid_schema", "profile patch must be an object")
                wrapped = {
                    "has_meaningful_content": True,
                    "topics": topics_json,
                    "profile_updates": data.get("profile_updates", {}),
                    "profile_rewrites": data.get("profile_rewrites", {}),
                }
                patch_plan, patch_err = ConsolidationPlanValidator.validate(wrapped)
                if patch_err is not None or patch_plan is None:
                    return {}, {}, patch_err or PlanError("invalid_schema", "invalid profile patch")
                return (
                    dict(patch_plan.profile_updates),
                    dict(patch_plan.profile_rewrites),
                    None,
                )

            resolution = await resolver.resolve(
                scope=scope, scope_id=scope_id, entries=t1_entries,
                profile_hash=expected_hash,
                fetch_full_plan=fetch_full_plan,
                fetch_profile_patch=fetch_profile_patch,
            )
            if resolution.error is not None or resolution.plan is None:
                plan_err = resolution.error or PlanError("invalid_schema", "empty plan")
                logger.error("ConsolidateMemoryTool: invalid plan: %s", plan_err)
                return json.dumps({
                    "status": "failed", "reason": plan_err.reason, "error": plan_err.detail,
                })
            plan = resolution.plan
            if not plan.has_meaningful:
                return json.dumps({
                    "status": "ok", "has_meaningful_content": False,
                    "topics_stored": 0, "topics_failed": 0, "summary_ids": [],
                    "profile_updates": {}, "updated_sections": [],
                    "rewritten_sections": [], "messages_summarized": len(t1_entries),
                    "entry_ids": entry_ids,
                }, ensure_ascii=False)
            if profile_read_failed and (plan.profile_updates or plan.profile_rewrites):
                return json.dumps({"status": "failed", "reason": "profile_read_failed"})
            stamps = [ts for ts in (entry_epoch(e) for e in t1_entries) if ts is not None]
            now_ts = datetime.now(timezone.utc).timestamp()
            period_start = min(stamps) if stamps else now_ts
            period_end = max(stamps) if stamps else now_ts
            writer = ConsolidationWriter(self.embedding_service, self.timeline_summary_store)
            summary_ids, attempted, failed = await writer.store_topics(
                scope=scope, scope_id=scope_id, entry_ids=entry_ids,
                topics=plan.topics, period_start=period_start, period_end=period_end,
            )
            if failed > 0 or len(summary_ids) != attempted:
                logger.error("ConsolidateMemoryTool: %d/%d T2 stores failed; refusing trim", failed, attempted)
                return json.dumps({
                    "status": "failed", "reason": "t2_store_failed",
                    "topics_attempted": attempted, "topics_failed": failed,
                    "topics_stored": len(summary_ids),
                }, ensure_ascii=False)
            profile_writer = ConsolidationProfileWriter(self.memory_manager.profile)
            try:
                prof = await profile_writer.apply(
                    scope=scope, scope_id=scope_id,
                    updates=dict(plan.profile_updates), rewrites=dict(plan.profile_rewrites),
                    expected_profile_hash=expected_hash,
                )
            except ValueError as exc:
                logger.error("ConsolidateMemoryTool: invalid profile bullets: %s", exc)
                return json.dumps({"status": "failed", "reason": "invalid_schema", "error": str(exc)})
            except OSError as exc:
                logger.error("ConsolidateMemoryTool: profile write failed: %s", exc)
                return json.dumps({"status": "failed", "reason": "profile_write_failed", "error": str(exc)})
            except Exception as exc:
                logger.error("ConsolidateMemoryTool: profile apply failed: %s", exc)
                return json.dumps({"status": "failed", "reason": "profile_write_failed", "error": str(exc)})
            if isinstance(prof, dict) and prof.get("conflict"):
                logger.error("ConsolidateMemoryTool: profile conflict; keeping new facts, no trim")
                return json.dumps({
                    "status": "failed", "reason": "profile_conflict",
                    "profile_hash": prof.get("profile_hash"),
                    "topics_stored": len(summary_ids), "summary_ids": summary_ids,
                }, ensure_ascii=False)
            if isinstance(prof, dict) and prof.get("ok") is False:
                return json.dumps({"status": "failed", "reason": "profile_write_failed"})
            updated = list(prof.get("updated_sections", [])) if isinstance(prof, dict) else []
            rewritten = list(prof.get("rewritten_sections", [])) if isinstance(prof, dict) else []
            if resolution.refresh is not None:
                rkey, rpayload = resolution.refresh
                await resolver.refresh(rkey, rpayload)
            return json.dumps({
                "status": "ok", "has_meaningful_content": True,
                "topics_stored": len(summary_ids), "topics_failed": 0,
                "summary_ids": summary_ids, "profile_updates": dict(plan.profile_updates),
                "updated_sections": updated, "rewritten_sections": rewritten,
                "messages_summarized": len(t1_entries), "entry_ids": entry_ids,
                "receiver_generation": resolution.generation,
            }, ensure_ascii=False)
        except LocalBatchError as exc:
            return json.dumps({"status": "failed", "reason": exc.reason, "error": exc.detail})
        except Exception as exc:
            logger.error("ConsolidateMemoryTool: execution failed: %s", exc, exc_info=True)
            raise ToolExecutionError(self.name, f"Lỗi khi consolidate: {exc}", original_error=exc)

    @traceable(name="memory.t1_read", run_type="retriever", tags=["memory", "t1"])
    async def _read_t1_context(self, scope: str, scope_id: str, max_messages: int, trace_base: dict[str, Any]) -> list:
        del trace_base
        return await read_local_tool_batch(self.memory_manager, scope, scope_id, max_messages)

    @traceable(name="memory.t1_read", run_type="retriever", tags=["memory", "t1"])
    async def _shipped_entries_as_t1(self, entries: list[dict], trace_base: dict[str, Any]) -> list[dict]:
        del trace_base
        return list(entries)

    @traceable(name="memory.t3_profile_read", run_type="retriever", tags=["memory", "t3"])
    async def _read_profile(self, scope_id: str, trace_base: dict[str, Any]) -> str:
        del trace_base
        return await self.memory_manager.profile.read_raw(scope_id)

    def _provider_name(self) -> str:
        class_name = self.llm_service.__class__.__name__ if self.llm_service else ""
        if "Gemini" in class_name:
            return "gemini"
        return "openai_compatible"

    def _model_name(self) -> str:
        return str(getattr(self.llm_service, "model", "unknown") or "unknown")
