"""
Redis cache layer.
- Query result caching (LRU, TTL=1hr)
- Rate limiting via sliding window counter
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, List, Optional, Tuple

import redis

from core.config import settings


def get_redis() -> redis.Redis:
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=0,
        decode_responses=True,
    )


class QueryCache:
    TTL = 3600  # 1 hour

    def __init__(self):
        self.r = get_redis()

    def _key(self, query: str, k: int) -> str:
        h = hashlib.sha256(f"{query}:{k}".encode()).hexdigest()[:20]
        return f"cortex:cache:{h}"

    def get(self, query: str, k: int) -> Optional[List[dict]]:
        raw = self.r.get(self._key(query, k))
        if raw:
            return json.loads(raw)
        return None

    def set(self, query: str, k: int, results: List[dict]):
        self.r.setex(self._key(query, k), self.TTL, json.dumps(results))

    def invalidate_repo(self, repo_id: str):
        """Called when a repo is re-indexed — flush related cache keys."""
        pattern = "cortex:cache:*"
        keys = self.r.keys(pattern)
        if keys:
            self.r.delete(*keys)


class RateLimiter:
    """
    Sliding window rate limiter.
    Allows `limit` requests per `window_seconds` per identifier (IP or user_id).
    """

    def __init__(self, limit: int = 100, window_seconds: int = 60):
        self.r = get_redis()
        self.limit = limit
        self.window = window_seconds

    def is_allowed(self, identifier: str) -> Tuple[bool, int]:
        """
        Returns (allowed, remaining_requests).
        """
        key = f"cortex:ratelimit:{identifier}"
        now = time.time()
        window_start = now - self.window

        pipe = self.r.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, self.window)
        results = pipe.execute()

        count = results[2]
        remaining = max(0, self.limit - count)
        return count <= self.limit, remaining
