"""Evernight Agent configuration."""
import os
from dataclasses import dataclass


@dataclass
class EvernightConfig:
    port: int = 8001
    redis_db: int = 1
    persona_path: str = "twin/evernight/personas"
    agent_name: str = "evernight"
    march7_url: str = "http://march7:8000"
    poll_interval: int = 60
    discord_evernight_token: str | None = None
    # Explicit owner authority only; None/empty means unknown -> deny. No default.
    owner_user_id: str | None = None
    self_heal_enabled: bool = True
    self_heal_interval: int = 30
    self_heal_timeout: int = 10

    @classmethod
    def from_env(cls) -> "EvernightConfig":
        return cls(
            port=int(os.getenv("EVERNIGHT_A2A_PORT", "8001")),
            redis_db=int(os.getenv("EVERNIGHT_REDIS_DB", "1")),
            persona_path=os.getenv("EVERNIGHT_PERSONA_PATH", "twin/evernight/personas"),
            agent_name=os.getenv("AGENT_NAME", "evernight"),
            march7_url=os.getenv("MARCH7_URL", "http://march7:8000"),
            poll_interval=int(os.getenv("POLL_INTERVAL", "60")),
            discord_evernight_token=(os.getenv("DISCORD_EVERNIGHT_TOKEN") or "").strip() or None,
            owner_user_id=(os.getenv("EVERNIGHT_OWNER_USER_ID") or "").strip() or None,
            self_heal_enabled=os.getenv("SELF_HEAL_ENABLED", "true").lower() == "true",
            self_heal_interval=int(os.getenv("SELF_HEAL_INTERVAL", "30")),
            self_heal_timeout=int(os.getenv("SELF_HEAL_TIMEOUT", "10")),
        )
