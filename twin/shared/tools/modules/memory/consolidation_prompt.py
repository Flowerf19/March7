"""Consolidation prompt builder with per-entry VN timestamps."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from twin.shared.memory.consolidation_journal import entry_epoch

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

SUMMARIZER_PROMPT = """Bạn là Memory Summarizer. Đọc cuộc trò chuyện và trích xuất thông tin đáng nhớ theo từng topic.

Bây giờ là {now} (giờ Việt Nam).

=== HỒ SƠ HIỆN TẠI ===
{profile}

=== TIN NHẮN GẦN ĐÂY ===
{messages}

Nhiệm vụ:
1. Lọc noise (chào hỏi đơn thuần, emoji phiếm, thông tin tạm thời vô nghĩa).
2. Nếu session TOÀN noise hoặc không có gì đáng nhớ → set has_meaningful_content=false, topics=[].
3. Với nội dung có ý nghĩa: nhóm theo topic, mỗi topic viết summary 3-5 câu như một trang nhật ký — GIỮ chi tiết cụ thể (tên riêng, con số, tên hàm/lỗi, địa danh, món đồ), không tóm tắt khô kiểu 1 câu.
4. Đổi mọi mốc thời gian tương đối thành tuyệt đối dựa trên thời điểm của từng tin nhắn (ghi trong ngoặc vuông trước mỗi tin, giờ Việt Nam), KHÔNG dựa trên thời điểm hiện tại ở trên. 'Bây giờ là ...' chỉ để biết hiện tại. Ví dụ tin ngày 01/07 nói 'ngày mai' → 02/07 ("deadline tuần tới" → ghi rõ ngày tuyệt đối theo tuần của tin đó, "hôm qua" → ghi rõ ngày của hôm trước tin đó).
5. Tối đa 5 topics. Topic slug: lowercase, underscore, tự đặt (ví dụ: work, interest, health, travel, relationship...).
6. Trích xuất facts mới đáng lưu vào hồ sơ → profile_updates (bỏ qua nếu đã có trong hồ sơ).
7. Nếu fact mới MÂU THUẪN với hồ sơ hiện tại (ví dụ user nói "hết thích game" mà hồ sơ có "Thích chơi game"): đưa section đó vào profile_rewrites với TOÀN BỘ danh sách bullet của section viết lại sạch — bỏ bullet lỗi thời, giữ bullet còn đúng, thêm bullet mới. Section đã nằm trong profile_rewrites thì KHÔNG đưa vào profile_updates nữa. Không có mâu thuẫn → profile_rewrites để {{}}.

Return JSON:
{{
  "has_meaningful_content": true,
  "topics": [
    {{
      "topic": "work",
      "topic_display": "Công việc",
      "summary": "User đang làm dự án X cho khách hàng Y, deadline ~10/07/2026. Gặp lỗi ImportError ở module auth khi deploy staging, đã thử hạ Python 3.12 xuống 3.11 nhưng chưa ăn thua. Dự định hỏi anh Nam team infra vào 04/07/2026.",
      "importance": 4
    }}
  ],
  "profile_updates": {{
    "basic": [],
    "work": ["Đang làm dự án X"],
    "interest": [],
    "relationship": [],
    "habit": [],
    "psychological": [],
    "rules": [],
    "contact": []
  }},
  "profile_rewrites": {{}}
}}

Rules importance:
- 5: identity/contact critical
- 4: work/relationship/habit quan trọng
- 3: interest/event thông thường
- 2-1: casual, temporary

Chỉ return JSON, không giải thích."""


def entry_vn_str(entry: Any) -> str:
    """VN-local 'HH:MM dd/mm/yyyy' for one entry, or unknown marker."""
    epoch = entry_epoch(entry)
    if epoch is None:
        return "(không rõ thời gian)"
    try:
        return datetime.fromtimestamp(epoch, tz=VN_TZ).strftime("%H:%M %d/%m/%Y")
    except (OverflowError, OSError, ValueError):
        return "(không rõ thời gian)"


class ConsolidationPromptBuilder:
    """Format shipped/local entries and build the summarizer prompt."""

    @staticmethod
    def format_messages(entries: list[Any]) -> str:
        lines: list[str] = []
        for entry in entries:
            if isinstance(entry, dict):
                role = entry.get("role") or "user"
                content = entry.get("content", "")
                author = entry.get("author_name") or role
            else:
                role = getattr(entry, "role", None) or "user"
                content = getattr(entry, "content", None)
                if content is None:
                    content = str(entry)
                author = getattr(entry, "author_name", None) or role
            vn_time = entry_vn_str(entry)
            lines.append(f"[{vn_time} (VN) {author}]: {content}")
        return "\n".join(lines)

    @staticmethod
    def build(now_vn: str, profile_text: str, messages_text: str) -> str:
        return SUMMARIZER_PROMPT.format(
            now=now_vn, profile=profile_text, messages=messages_text,
        )
