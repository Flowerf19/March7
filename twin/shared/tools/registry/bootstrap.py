"""Build agent-scoped tool registries from explicit declarations."""
from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Optional

from twin.shared.config.settings import Config
from twin.shared.external.codebox_client import CodeBoxClient
from twin.shared.system_gateway import HostGatewayClient
from twin.shared.tools.approval_gate import ApprovalGate
from twin.shared.tools.registry.base import BaseTool
from twin.shared.tools.declarations.system_tools import SYSTEM_TOOL_SPECS, ToolSpec
from twin.shared.tools.dm_client import DMClient
from twin.shared.tools.mcp_client import MCPClient
from twin.shared.tools.mcp_transport import HTTPTransport
from twin.shared.tools.prompts.catalog import ToolPromptCatalog, read_tool_description
from twin.shared.tools.registry.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolBootstrapResult:
    registry: ToolRegistry
    tool_prompt_catalog: ToolPromptCatalog
    system_tools: list[BaseTool]
    approval_gate: ApprovalGate
    tavily_mcp_client: Optional[MCPClient]
    codebox_client: Optional[CodeBoxClient]
    host_gateway_client: Optional[HostGatewayClient]


class DeclaredToolProxy(BaseTool):
    """Apply catalog permissions without mutating implementation classes."""

    def __init__(self, tool: BaseTool, spec: ToolSpec):
        self._tool = tool
        self._spec = spec
        self._description = (
            read_tool_description(tool.name, spec.guide_path, spec.description_tag)
            if spec.guide_path
            else tool.description
        )

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._tool.parameters_schema

    @property
    def allowed_agents(self) -> Optional[set[str]]:
        if self._spec.allowed_to is not None:
            return set(self._spec.allowed_to)
        return self._tool.allowed_agents

    @property
    def visible_to_agents(self) -> Optional[set[str]]:
        if self._spec.visible_to is not None:
            return set(self._spec.visible_to)
        return self._tool.visible_to_agents

    async def execute(self, **kwargs) -> str:
        return await self._tool.execute(**kwargs)

    def __repr__(self) -> str:
        return f"<DeclaredToolProxy: {self.name} ({self._tool.__class__.__name__})>"


def build_tool_registry(
    *,
    agent_name: str,
    core_manager: Any,
    memory_manager: Any,
    llm_service: Any,
    base_memory_path: str,
    profile_store: Any = None,
    approval_gate: ApprovalGate | None = None,
    use_evernight_dm_approval: bool = False,
    embedding_service: Any = None,
    timeline_summary_store: Any = None,
    owner_user_id: str | None = None,
    gateway_monitor: Any = None,
) -> ToolBootstrapResult:
    """Build one registry for an agent from the shared declared tool catalog."""
    tavily_mcp_client = _init_tavily_mcp_client()
    codebox_client = _init_codebox_client()
    host_gateway_client = _init_host_gateway_client(agent_name=agent_name)

    if approval_gate is None:
        approval_gate = _build_approval_gate(
            agent_name=agent_name,
            use_evernight_dm_approval=use_evernight_dm_approval,
        )

    dependencies = {
        "core_manager": core_manager,
        "memory_manager": memory_manager,
        "profile_store": profile_store,
        "llm_service": llm_service,
        "base_memory_path": base_memory_path,
        "tavily_mcp_client": tavily_mcp_client,
        "codebox_client": codebox_client,
        "host_gateway_client": host_gateway_client,
        "host_gateway_timeout": Config.SYSTEM_GATEWAY_TIMEOUT,
        "approval_gate": approval_gate,
        "embedding_service": embedding_service,
        "timeline_summary_store": timeline_summary_store,
        "owner_user_id": owner_user_id,
        "gateway_monitor": gateway_monitor,
    }

    registry = ToolRegistry(agent_name=agent_name)
    system_tools = [
        _instantiate_declared_tool(spec, dependencies)
        for spec in SYSTEM_TOOL_SPECS
    ]
    registry.register_tools(system_tools)
    tool_prompt_catalog = ToolPromptCatalog.from_tools_and_specs(
        system_tools,
        SYSTEM_TOOL_SPECS,
        agent_name=agent_name,
    )
    tool_prompt_catalog.render_catalog()
    logger.info(
        "Declared tools loaded for %s: %s - %s",
        agent_name,
        len(system_tools),
        [tool.name for tool in system_tools],
    )

    return ToolBootstrapResult(
        registry=registry,
        tool_prompt_catalog=tool_prompt_catalog,
        system_tools=system_tools,
        approval_gate=approval_gate,
        tavily_mcp_client=tavily_mcp_client,
        codebox_client=codebox_client,
        host_gateway_client=host_gateway_client,
    )


def _instantiate_declared_tool(
    spec: ToolSpec,
    dependencies: dict[str, Any],
) -> BaseTool:
    module = importlib.import_module(spec.module)
    tool_class = getattr(module, spec.class_name)
    sig = inspect.signature(tool_class.__init__)

    kwargs = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param_name in dependencies:
            kwargs[param_name] = dependencies[param_name]
        elif param.default != inspect.Parameter.empty:
            continue
        else:
            logger.warning(
                "Missing required dependency '%s' for %s",
                param_name,
                spec.class_name,
            )

    tool = tool_class(**kwargs)
    return DeclaredToolProxy(tool, spec)


def _build_approval_gate(
    *,
    agent_name: str,
    use_evernight_dm_approval: bool,
) -> ApprovalGate:
    if not use_evernight_dm_approval:
        return ApprovalGate()

    evernight_url = getattr(Config, "EVERNIGHT_A2A_URL", None)
    if not evernight_url:
        return ApprovalGate()

    dm_client = DMClient(evernight_url=evernight_url, actor=agent_name)
    logger.info("DM approval client configured for %s: %s", agent_name, evernight_url)
    return ApprovalGate(dm_client=dm_client)


def _init_tavily_mcp_client() -> Optional[MCPClient]:
    if not Config.TAVILY_API_KEY:
        return None
    try:
        transport = HTTPTransport(
            Config.TAVILY_MCP_URL,
            timeout=Config.TAVILY_TIMEOUT,
            headers={"Authorization": f"Bearer {Config.TAVILY_API_KEY}"},
        )
        return MCPClient(transport)
    except Exception as exc:
        logger.warning("Tavily MCP init failed: %s", exc)
        return None


def _init_codebox_client() -> Optional[CodeBoxClient]:
    try:
        return CodeBoxClient()
    except Exception as exc:
        logger.warning("CodeBox init failed: %s", exc)
        return None


def _init_host_gateway_client(*, agent_name: str) -> Optional[HostGatewayClient]:
    gateway_url = getattr(Config, "SYSTEM_GATEWAY_URL", "")
    if not gateway_url:
        return None
    try:
        return HostGatewayClient(
            base_url=gateway_url,
            timeout=Config.SYSTEM_GATEWAY_TIMEOUT,
            shared_secret=Config.SYSTEM_GATEWAY_SHARED_SECRET,
            actor=agent_name,
        )
    except Exception as exc:
        logger.warning("System Gateway client init failed: %s", exc)
        return None
