"""Declared logical tool catalog.

This file is the audit point for declared tools: what exists, which backend
kind owns execution, and which agents may see or execute each tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, get_args


ToolBackend = Literal["local", "remote_mcp"]
_TOOL_BACKENDS = frozenset(get_args(ToolBackend))


@dataclass(frozen=True)
class ToolSpec:
    module: str
    class_name: str
    visible_to: Optional[frozenset[str]] = None
    allowed_to: Optional[frozenset[str]] = None
    guide_path: str | None = None
    description_tag: str = "tool_description"
    backend: ToolBackend = "local"

    def __post_init__(self) -> None:
        if self.backend not in _TOOL_BACKENDS:
            raise ValueError(f"Invalid tool backend for {self.class_name}: {self.backend!r}")
        if self.backend == "remote_mcp" and not self.guide_path:
            raise ValueError(
                f"remote_mcp tool declarations require a local guide_path: {self.class_name}"
            )


SYSTEM_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        module="twin.shared.tools.modules.memory.search_memory_tool",
        class_name="SearchMemoryTool",
        guide_path="guides/search_memory.md",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.memory.consolidate_memory_tool",
        class_name="ConsolidateMemoryTool",
        visible_to=frozenset({"evernight"}),
        allowed_to=frozenset({"evernight"}),
        guide_path="guides/consolidate_memory.md",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.profile.get_profile_tool",
        class_name="GetProfileTool",
        guide_path="guides/get_profile.md",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.profile.update_profile_tool",
        class_name="UpdateUserProfileTool",
        guide_path="guides/update_user_profile.md",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.profile.manage_profile_tool",
        class_name="ManageUserProfileTool",
        visible_to=frozenset({"evernight"}),
        allowed_to=frozenset({"evernight"}),
        guide_path="guides/manage_user_profile.md",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.profile.update_personality_tool",
        class_name="UpdatePersonalityTool",
        guide_path="guides/update_personality.md",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.web.tavily_search_tool",
        class_name="TavilySearchTool",
        guide_path="guides/web_search.md",
        backend="remote_mcp",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.execution.code_interpreter_tool",
        class_name="CodeInterpreterTool",
        guide_path="guides/run_python_code.md",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.system.host_system_tool",
        class_name="HostSystemTool",
        guide_path="guides/host_system.md",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.system.gateway_admin_tool",
        class_name="GatewayAdminTool",
        visible_to=frozenset({"evernight"}),
        allowed_to=frozenset({"evernight"}),
        guide_path="guides/gateway_admin.md",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.a2a.march7_snapshot_tool",
        class_name="March7SnapshotTool",
        visible_to=frozenset({"evernight"}),
        allowed_to=frozenset({"evernight"}),
        guide_path="guides/march7_snapshot.md",
    ),
    ToolSpec(
        module="twin.shared.tools.modules.a2a.request_consolidation_tool",
        class_name="RequestConsolidationTool",
        visible_to=frozenset({"march7"}),
        allowed_to=frozenset({"march7"}),
        guide_path="guides/request_consolidation.md",
    ),
)
