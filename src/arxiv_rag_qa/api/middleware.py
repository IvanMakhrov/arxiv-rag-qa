import time
from threading import Lock

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


class ThroughputTracker:
    """Simple sliding-window throughput tracker.

    Tracks request counts in 1-minute buckets to compute
    requests per second (RPS) over a configurable window.
    """

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._buckets: list[tuple[float, int]] = []
        self._latencies: list[float] = []
        self._lock = Lock()

    def record_request(self, latency_ms: float) -> None:
        """Record a completed request with its latency."""
        now = time.time()
        with self._lock:
            self._buckets.append((now, 1))
            self._latencies.append(latency_ms)
            cutoff = now - self.window_seconds
            self._buckets = [(t, c) for t, c in self._buckets if t >= cutoff]
            keep_cutoff = now - 300
            self._latencies = [lat for lat in self._latencies if lat >= keep_cutoff]

    def get_rps(self) -> float:
        """Requests per second over the sliding window."""
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            count = sum(c for t, c in self._buckets if t >= cutoff)
        return count / self.window_seconds if self.window_seconds > 0 else 0.0

    def get_stats(self) -> dict:
        """Return snapshot of current throughput and latency stats."""
        with self._lock:
            total_requests = sum(c for _, c in self._buckets)
            rps = total_requests / self.window_seconds if self.window_seconds > 0 else 0.0
            latencies = list(self._latencies)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
        p99 = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else 0.0
        return {
            "total_requests_in_window": total_requests,
            "window_seconds": self.window_seconds,
            "rps": round(rps, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "p50_latency_ms": round(p50, 2),
            "p99_latency_ms": round(p99, 2),
        }


throughput_tracker = ThroughputTracker(window_seconds=60)


class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000

        if request.url.path != "/health":
            throughput_tracker.record_request(latency_ms)

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
