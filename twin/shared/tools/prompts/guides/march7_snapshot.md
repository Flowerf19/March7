<tool_description>
march7_snapshot — Xem T1 (ngữ cảnh ngắn hạn) của Bé Bảy/March7 cho 1 user qua A2A. Đây là đường DUY NHẤT để biết March7 đang giữ gì — KHÔNG đọc trực tiếp Redis của March7, KHÔNG đoán.
Khi nên dùng:
- Owner hỏi March7 đang nhớ/giữ gì về ai đó ("Bé Bảy đang nhớ gì về X?", "check T1 của Y").
- Cần đối chiếu trước khi dọn dẹp, migrate, hoặc debug memory.
- Không bao giờ dùng để theo dõi lén — chỉ khi owner yêu cầu hoặc phục vụ bảo trì memory.
</tool_description>

## march7_snapshot

Đọc T1 của March7 qua A2A skill `get_snapshot`. Read-only, không sửa gì bên March7.

### Input

- `user_id`: bắt buộc, Discord ID dạng số.
- `limit`: số tin gần nhất, mặc định 20, tối đa 50. Snapshot đầy đủ có thể tới 200 tin — đừng xin nhiều nếu chỉ cần xem lướt.

### Đọc kết quả

- Mỗi dòng là 1 tin: `[role/author] nội dung`. Tin dài bị cắt ở 300 ký tự.
- "trống hoặc đã consolidate" = T1 sạch, muốn biết chuyện cũ thì dùng `search_memory` (T2) như thường.
- "không kết nối được March7 qua A2A" = March7 container down hoặc sai `MARCH7_URL` — báo owner, đừng bịa nội dung.
