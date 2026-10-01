import asyncio
import os
import sys

# Thêm đường dẫn để import twin
sys.path.insert(0, os.path.abspath('.'))

# Nạp .env TRƯỚC khi import twin để Config đọc được A2A_SHARED_SECRET.
# Chạy từ host nên mặc định dùng loopback; trong container mới dùng hostname.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.abspath('.'), '.env'))
except ImportError:
    pass

from twin.shared.memory.consolidation_client import ConsolidationClient

DEFAULT_URL = "http://127.0.0.1:8001"

async def main():
    print("Testing A2A Consolidation Client to Evernight...")
    # Ưu tiên: arg CLI > EVERNIGHT_A2A_URL_TEST > mặc định loopback (host).
    # Không dùng thẳng EVERNIGHT_A2A_URL vì trong .env nó là hostname
    # container (http://evernight:8001), host không phân giải được.
    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.getenv("EVERNIGHT_A2A_URL_TEST", DEFAULT_URL)
    )
    print(f"Target: {url}")
    if not os.getenv("A2A_SHARED_SECRET"):
        print("WARNING: A2A_SHARED_SECRET chưa có trong env — request sẽ fail auth.")
    # Khởi tạo client gọi tới Evernight (secret tự lấy từ Config.A2A_SHARED_SECRET)
    client = ConsolidationClient(evernight_url=url)

    result = await client.consolidate_scope(
        scope="user",
        scope_id="1234567890", # Test user ID
        reason="manual_test"
    )
    print("Result:", result)
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
