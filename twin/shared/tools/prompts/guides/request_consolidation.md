<tool_description>
request_consolidation — Ép nén T1 (ngữ cảnh ngắn hạn) thành T2/T3 NGAY qua Evernight, thay vì chờ tự động. Chỉ dùng khi user yêu cầu rõ ràng ("lưu lại đi", "nén memory đi", "ghi nhớ đoạn này").
KHÔNG gọi theo thói quen mỗi turn — đường tự động (đủ 2000 từ / idle 15 phút) đã lo việc này. Gọi thừa chỉ tốn 1 lượt LLM của Evernight mà không thêm được gì.
</tool_description>

## request_consolidation

Kích hoạt đúng flow consolidate tự động (ship T1 → Evernight summarize → ghi T2/T3 → trim T1 giữ 5 tin gần nhất), nhưng chạy ngay theo lệnh.

### Input

- `user_id`: bắt buộc, Discord ID dạng số.
- `channel_id`: chỉ truyền khi muốn nén T1 của kênh chung. Nén kênh chỉ ghi T2, không đụng T3 profile.

### Đọc kết quả

- "Đã nén T1... X tin → T2/T3" = xong. Trả lời user ngắn gọn là đã lưu.
- "Bỏ qua: không có gì mới" = T1 trống/sạch — đừng gọi lại.
- "thất bại (Evernight không phản hồi?)" = Evernight container down hoặc sai `EVERNIGHT_A2A_URL` — báo user thử lại sau.
