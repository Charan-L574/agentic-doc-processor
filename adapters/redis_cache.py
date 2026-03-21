from __future__ import annotations

from typing import Optional

import redis

from services.cache_service import CacheService
from utils.config import settings


class RedisCacheAdapter(CacheService):
    """Redis-backed cache adapter for cloud profile."""

    def __init__(self):
        host = settings.AWS_REDIS_ENDPOINT
        if not host:
            raise ValueError("Redis endpoint not configured. Set [aws].redis_endpoint in config.ini")
        self.client = redis.Redis(host=host, port=6379, decode_responses=True)

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

    def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        self.client.set(name=key, value=value, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self.client.delete(key)
