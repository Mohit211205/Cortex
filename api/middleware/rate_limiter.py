from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from core.cache import RateLimiter
from core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.limiter = RateLimiter(
            limit=settings.RATE_LIMIT,
            window_seconds=settings.RATE_LIMIT_WINDOW,
        )

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host
        allowed, remaining = self.limiter.is_allowed(ip)

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
