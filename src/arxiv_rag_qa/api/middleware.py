import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "Request completed",
            extra={
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
                "user_agent": request.headers.get("user-agent", ""),
            },
        )
        return response
