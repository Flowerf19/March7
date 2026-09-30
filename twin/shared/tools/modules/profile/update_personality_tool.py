"""
UpdatePersonalityTool - Rewrite the current bot's persona Markdown files.

Tool for updating bot's personality/behavior files.
OVERWRITES entire file - bot must provide full merged content.

Bot workflow:
1. Read current content from system prompt (=== NHÂN CÁCH === / === HƯỚNG DẪN ===)
2. Merge with new info
3. Call this tool with target_file and full Markdown content
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from twin.shared.memory.profile.file_io import ProfileFileIO
from twin.shared.tools.registry.base import BaseTool, ToolExecutionError

logger = logging.getLogger(__name__)


class UpdatePersonalityTool(BaseTool):
    """
    Tool for updating bot persona files.

    target_file controls the target (always in the current bot persona dir):
    - IDENTITY.md: Character identity (who the bot is)
    - SOUL.md: Bot-specific communication style + conversation rules
    - Any other .md filename in the current bot persona directory, when provided

    Attributes:
        base_memory_path: Path to the current bot persona directory

    Example:
        tool = UpdatePersonalityTool()
        # target_file="IDENTITY.md" → current bot identity
        # target_file="SOUL.md" → current bot speech style
    """

    def __init__(
        self,
        base_memory_path: str = "memories",
        llm_service: Any = None,
    ):
        self.base_memory_path = base_memory_path
        self.llm_service = llm_service
        logger.debug(f"UpdatePersonalityTool initialized with base_path={base_memory_path}")

    # ==========================================
    # BASE TOOL PROPERTIES
    # ==========================================

    @property
    def name(self) -> str:
        return "update_personality"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Markdown content replacing the target file.",
                },
                "target_file": {
                    "type": "string",
                    "description": "Markdown filename only, no path.",
                },
            },
            "required": ["instruction", "target_file"],
        }

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[5]

    def _resolve_base_path(self, path: str | None, default: Path) -> Path:
        if not path:
            return default
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._repo_root() / candidate
        return candidate

    def _persona_dir(self) -> Path:
        return self._resolve_base_path(self.base_memory_path, self._repo_root() / "memories")

    def _normalize_target_file(self, target_file: str | None) -> str:
        target = (target_file or "").strip()
        if not target:
            raise ValueError("Thiếu target_file.")
        if not target.lower().endswith(".md"):
            target = f"{target}.md"
        if Path(target).name != target or "/" in target or "\\" in target:
            raise ValueError("target_file chỉ được là tên file, không được chứa path.")
        if target.startswith(".") or target in {".md", "..md"}:
            raise ValueError("target_file không hợp lệ.")
        if not target.endswith(".md"):
            raise ValueError("target_file phải là file Markdown .md.")
        if target.lower() == "identity.md":
            return "IDENTITY.md"
        if target.lower() == "soul.md":
            return "SOUL.md"
        return target

    def _target_path(self, filename: str) -> Path:
        return self._persona_dir() / filename

    @staticmethod
    def _atomic_write_sync(path: Path, content: str) -> None:
        """Hardened persona write; old file survives pre-replace faults."""
        ProfileFileIO.atomic_write_sync(path, content)

    # ==========================================
    # EXECUTION
    # ==========================================

    async def execute(self, instruction: str, target_file: str | None = None) -> str:
        """
        Rewrite personality file with new content.

        Args:
            instruction: Full content for the file (Markdown)

        Returns:
            str: Success message with target file info
        """
        if not instruction:
            return "Lỗi: Thiếu instruction."

        target = "(unknown)"
        try:
            target = self._normalize_target_file(target_file)
            file_path = self._target_path(target)
            os.makedirs(file_path.parent, exist_ok=True)
            content = instruction if instruction.endswith("\n") else f"{instruction}\n"
            self._atomic_write_sync(file_path, content)
            if self.llm_service and hasattr(self.llm_service, "reload_persona_prompts"):
                self.llm_service.reload_persona_prompts()
            logger.info("✅ UpdatePersonalityTool: Đã viết lại %s", file_path)
            return f"Đã cập nhật {target} (persona) thành công. Áp dụng ngay cho bot hiện tại từ tin nhắn tiếp theo."

        except ValueError as e:
            return f"Lỗi: {e}"
        except Exception as e:
            logger.error(f"Lỗi khi ghi file {target}: {e}")
            raise ToolExecutionError(self.name, f"Lỗi khi ghi file: {e}", original_error=e)

    def __repr__(self) -> str:
        return f"<UpdatePersonalityTool: base_path={self.base_memory_path}>"
