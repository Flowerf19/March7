from pathlib import Path

import pytest

from twin.shared.llm.base_llm_service import BaseLLMService
from twin.shared.tools.declarations.system_tools import SYSTEM_TOOL_SPECS, ToolSpec
from twin.shared.tools.prompts.catalog import (
    ToolPromptCatalog,
    ToolPromptSpec,
    read_tool_description,
    _read_guide,
)
from twin.shared.tools.registry import ToolExecutionError, ToolRegistry, build_tool_registry
from twin.shared.tools.registry.base import BaseTool
from twin.shared.tools.registry.bootstrap import DeclaredToolProxy


class DummyTool(BaseTool):
    @property
    def name(self):
        return "dummy_tool"

    @property
    def description(self):
        return "Dummy."

    @property
    def parameters_schema(self):
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        return "ok"


class RestrictedDummyTool(DummyTool):
    @property
    def allowed_agents(self):
        return {"evernight"}


class ExplodingDummyTool(DummyTool):
    async def execute(self, **kwargs):
        raise RuntimeError("boom")


class DummyLLMService(BaseLLMService):
    async def generate_response(
        self,
        messages,
        system_prompt=None,
        use_native_tools=False,
        include_tool_catalog=True,
        max_tokens=None,
        tool_choice=None,
    ):
        return "ok"


def test_catalog_renders_only_micro_lines_from_guides():
    result = build_tool_registry(
        agent_name="march7",
        core_manager=None,
        memory_manager=None,
        llm_service=object(),
        base_memory_path="memories",
    )

    catalog = result.tool_prompt_catalog.render_catalog()
    search_description = result.tool_prompt_catalog.render_tool_description("search_memory")
    web_description = result.tool_prompt_catalog.render_tool_description("web_search")

    assert f"- search_memory: {search_description}" in catalog
    assert f"- web_search: {web_description}" in catalog
    assert "## search_memory" not in catalog
    assert "### Khi nên dùng" not in catalog


def test_render_tool_guide_wraps_full_selected_guide():
    result = build_tool_registry(
        agent_name="march7",
        core_manager=None,
        memory_manager=None,
        llm_service=object(),
        base_memory_path="memories",
    )

    guide = result.tool_prompt_catalog.render_tool_guide("web_search")
    web_description = result.tool_prompt_catalog.render_tool_description("web_search")

    assert guide.startswith('<tool_guide name="web_search">')
    assert web_description in guide
    assert "## web_search" in guide
    assert "<tool_description>" not in guide


def test_include_directive_inlines_guide_body(tmp_path: Path):
    # @include in a guide body pulls in another guide file (single level).
    included = tmp_path / "included.md"
    included.write_text(
        "## included body\nDòng bootstrap được nhúng từ file khác.", encoding="utf-8"
    )
    host = tmp_path / "host.md"
    host.write_text(
        "<tool_description>\nMô tả ngắn.\n</tool_description>\n\n"
        "## host\nTrước include.\n\n@include included.md\n\nSau include.",
        encoding="utf-8",
    )
    spec = ToolPromptSpec(name="host_tool", guide_path=host)

    # Short description must NOT contain the included body.
    short = read_tool_description("host_tool", host)
    assert "bootstrap được nhúng" not in short
    assert "Mô tả ngắn" in short

    # Full guide must contain the inlined body, with the @include line gone.
    full = _read_guide(spec)
    assert "@include" not in full
    assert "Dòng bootstrap được nhúng từ file khác." in full
    assert "Trước include." in full and "Sau include." in full


def test_include_missing_target_fails_loudly(tmp_path: Path):
    host = tmp_path / "host.md"
    host.write_text(
        "<tool_description>\nMô tả.\n</tool_description>\n\n"
        "@include does_not_exist.md",
        encoding="utf-8",
    )
    spec = ToolPromptSpec(name="host_tool", guide_path=host)
    with pytest.raises(FileNotFoundError, match="@include target not found"):
        _read_guide(spec)


def test_missing_description_block_fails_clearly(tmp_path: Path):
    guide = tmp_path / "bad.md"
    guide.write_text("## bad\nNo description block", encoding="utf-8")
    catalog = ToolPromptCatalog(
        [ToolPromptSpec(name="bad_tool", guide_path=guide)]
    )

    with pytest.raises(ValueError, match="missing <tool_description>"):
        catalog.render_catalog()


def test_native_schema_description_comes_from_tool_description_tag():
    result = build_tool_registry(
        agent_name="march7",
        core_manager=None,
        memory_manager=None,
        llm_service=object(),
        base_memory_path="memories",
    )

    schemas = result.registry.get_all_openai_schemas()
    expected_guides = {
        "search_memory": "guides/search_memory.md",
        "get_profile": "guides/get_profile.md",
        "update_user_profile": "guides/update_user_profile.md",
        "update_personality": "guides/update_personality.md",
        "web_search": "guides/web_search.md",
        "run_python_code": "guides/run_python_code.md",
        "host_system": "guides/host_system.md",
        "request_consolidation": "guides/request_consolidation.md",
    }
    descriptions = {
        schema["function"]["name"]: schema["function"]["description"]
        for schema in schemas
    }

    assert [schema["function"]["name"] for schema in schemas] == list(expected_guides)
    assert descriptions == {
        name: read_tool_description(name, guide_path)
        for name, guide_path in expected_guides.items()
    }


def test_visible_tool_without_guide_fails_clearly():
    with pytest.raises(ValueError, match="Missing guide_path for visible tool: dummy_tool"):
        ToolPromptCatalog.from_tools_and_specs(
            [DummyTool()],
            [
                ToolSpec(
                    module="tests.unit.tool_prompt_catalog_test",
                    class_name="DummyTool",
                )
            ],
            agent_name="march7",
        )


def test_final_system_prompt_uses_micro_catalog_not_persona_tool_md():
    llm = DummyLLMService(persona_path="twin/march7/personas")
    result = build_tool_registry(
        agent_name="march7",
        core_manager=None,
        memory_manager=None,
        llm_service=object(),
        base_memory_path="memories",
    )
    llm.set_tool_prompt_catalog(result.tool_prompt_catalog)

    prompt = llm._build_final_system_prompt("dynamic memory")
    search_description = result.tool_prompt_catalog.render_tool_description("search_memory")

    assert "=== CÔNG CỤ ===" in prompt
    assert f"- search_memory: {search_description}" in prompt
    assert "## search_memory" not in prompt
    assert "Quick routing" not in prompt
    assert "dynamic memory" in prompt


def test_final_system_prompt_includes_identity_and_soul():
    # Both Think(decide) calls go through _build_final_system_prompt, so the
    # bot's IDENTITY.md and SOUL.md must always be present in the system prompt.
    llm = DummyLLMService(persona_path="twin/march7/personas")

    prompt = llm._build_final_system_prompt("")

    assert "=== NHÂN CÁCH CỦA BẠN ===" in prompt
    assert "## Danh tính" in prompt  # from IDENTITY.md
    assert "=== HƯỚNG DẪN HỘI THOẠI ===" in prompt
    assert "## Phong cách giao tiếp" in prompt  # from SOUL.md


def test_current_tool_backend_classification():
    backends_by_class = {
        spec.class_name: spec.backend
        for spec in SYSTEM_TOOL_SPECS
    }

    assert backends_by_class["TavilySearchTool"] == "remote_mcp"
    assert {
        class_name
        for class_name, backend in backends_by_class.items()
        if backend == "remote_mcp"
    } == {"TavilySearchTool"}
    assert all(
        backend == "local"
        for class_name, backend in backends_by_class.items()
        if class_name != "TavilySearchTool"
    )


def test_catalog_output_is_unchanged_by_backend_metadata():
    result = build_tool_registry(
        agent_name="march7",
        core_manager=None,
        memory_manager=None,
        llm_service=object(),
        base_memory_path="memories",
    )

    catalog = result.tool_prompt_catalog.render_catalog()

    assert catalog == "\n\n".join(
        [
            f"- get_profile: {read_tool_description('get_profile', 'guides/get_profile.md')}",
            f"- host_system: {read_tool_description('host_system', 'guides/host_system.md')}",
            f"- request_consolidation: {read_tool_description('request_consolidation', 'guides/request_consolidation.md')}",
            f"- run_python_code: {read_tool_description('run_python_code', 'guides/run_python_code.md')}",
            f"- search_memory: {read_tool_description('search_memory', 'guides/search_memory.md')}",
            f"- update_personality: {read_tool_description('update_personality', 'guides/update_personality.md')}",
            f"- update_user_profile: {read_tool_description('update_user_profile', 'guides/update_user_profile.md')}",
            f"- web_search: {read_tool_description('web_search', 'guides/web_search.md')}",
        ]
    )


def test_backend_metadata_does_not_enter_prompt_catalog(tmp_path: Path):
    guide = tmp_path / "dummy.md"
    guide.write_text(
        "<tool_description>\nLocal guide description.\n</tool_description>\n\n"
        "## dummy_tool\nFull local guide.",
        encoding="utf-8",
    )
    tool = DummyTool()
    spec = ToolSpec(
        module="tests.unit.tool_prompt_catalog_test",
        class_name="DummyTool",
        guide_path=str(guide),
        backend="remote_mcp",
    )

    catalog = ToolPromptCatalog.from_tools_and_specs([tool], [spec])

    assert catalog.render_catalog() == "- dummy_tool: Local guide description."
    assert "remote_mcp" not in catalog.render_catalog()


def test_declared_proxy_description_uses_local_guide_for_remote_backend(tmp_path: Path):
    guide = tmp_path / "dummy.md"
    guide.write_text(
        "<tool_description>\nLocal guide description.\nRemote metadata is not here.\n</tool_description>",
        encoding="utf-8",
    )
    local_spec = ToolSpec(
        module="tests.unit.tool_prompt_catalog_test",
        class_name="DummyTool",
        guide_path=str(guide),
    )
    remote_spec = ToolSpec(
        module="tests.unit.tool_prompt_catalog_test",
        class_name="DummyTool",
        guide_path=str(guide),
        backend="remote_mcp",
    )

    local_proxy = DeclaredToolProxy(DummyTool(), local_spec)
    remote_proxy = DeclaredToolProxy(DummyTool(), remote_spec)

    assert remote_proxy.description == "Local guide description.\nRemote metadata is not here."
    assert remote_proxy.get_openai_schema() == local_proxy.get_openai_schema()
    assert "remote_mcp" not in remote_proxy.get_openai_schema()["function"]["description"]


def test_remote_mcp_visible_declarations_require_local_guide_path():
    with pytest.raises(ValueError, match="remote_mcp tool declarations require a local guide_path"):
        ToolSpec(
            module="tests.unit.tool_prompt_catalog_test",
            class_name="DummyTool",
            backend="remote_mcp",
        )


@pytest.mark.asyncio
async def test_registry_local_execution_checks_permission_before_execution():
    registry = ToolRegistry(agent_name="march7")
    registry.register_tool(RestrictedDummyTool())

    with pytest.raises(ToolExecutionError) as exc:
        await registry.execute_tool("dummy_tool", {})

    assert "không có quyền" in str(exc.value)


@pytest.mark.asyncio
async def test_registry_local_execution_validates_parameters_before_execution():
    class RequiredDummyTool(DummyTool):
        @property
        def parameters_schema(self):
            return {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }

    registry = ToolRegistry(agent_name="march7")
    registry.register_tool(RequiredDummyTool())

    with pytest.raises(ToolExecutionError) as exc:
        await registry.execute_tool("dummy_tool", {})

    assert "Missing required parameter: 'query'" in str(exc.value)


@pytest.mark.asyncio
async def test_registry_local_execution_wraps_tool_exceptions():
    registry = ToolRegistry(agent_name="march7")
    registry.register_tool(ExplodingDummyTool())

    with pytest.raises(ToolExecutionError) as exc:
        await registry.execute_tool("dummy_tool", {})

    assert str(exc.value) == "Tool 'dummy_tool' failed: boom"
    assert isinstance(exc.value.original_error, RuntimeError)
