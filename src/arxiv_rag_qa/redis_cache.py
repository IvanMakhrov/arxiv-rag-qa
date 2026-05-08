import hashlib
import json
import os
from typing import Any

from redis import Redis

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


class RedisCacheManager:
    """Redis-based cache manager for query embeddings with frequency-based TTL."""

    def __init__(
        self,
        host: str | None = None,
        port: int = 6379,
        db: int = 0,
        cache_key_prefix: str = "rag:embedding:",
        frequency_key_prefix: str = "rag:frequency:",
        base_ttl: int = 3600,
        max_ttl: int = 86400,
        min_ttl: int = 600,
        frequency_decay: float = 0.95,
    ):
        """
        Initialize Redis cache manager.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            cache_key_prefix: Prefix for cache keys
            frequency_key_prefix: Prefix for frequency tracking keys
            base_ttl: Base TTL in seconds
            max_ttl: Maximum TTL in seconds
            min_ttl: Minimum TTL in seconds
            frequency_decay: Decay factor for frequency (0-1)
        """
        redis_host = os.getenv("REDIS_HOST", host) or "redis"
        self.redis_client = Redis(host=redis_host, port=port, db=db, decode_responses=True)
        self.cache_key_prefix = cache_key_prefix
        self.frequency_key_prefix = frequency_key_prefix
        self.base_ttl = base_ttl
        self.max_ttl = max_ttl
        self.min_ttl = min_ttl
        self.frequency_decay = frequency_decay

        try:
            self.redis_client.ping()
            logger.info("Redis cache manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def _generate_cache_key(self, query: str) -> str:
        """Generate a cache key for the query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{self.cache_key_prefix}{query_hash}"

    def _generate_frequency_key(self, query: str) -> str:
        """Generate a frequency tracking key for the query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{self.frequency_key_prefix}{query_hash}"

    def _get_ttl_for_frequency(self, frequency: float) -> int:
        """
        Calculate TTL based on query frequency.

        Args:
            frequency: Query frequency (higher = more frequent)

        Returns:
            TTL in seconds
        """
        if frequency <= 1:
            return self.min_ttl

        frequency_factor = min(frequency / 10, 10)
        ttl = self.base_ttl * (1 + frequency_factor)

        return int(min(max(ttl, self.min_ttl), self.max_ttl))

    def get_embedding(self, query: str) -> list[float] | None:
        """
        Get cached embedding for a query.

        Args:
            query: The query string

        Returns:
            Cached embedding as list of floats, or None if not found
        """
        if not self.redis_client:
            return None

        try:
            cache_key = self._generate_cache_key(query)
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                self._update_frequency(query, increment=True)
                embedding = json.loads(cached_data)
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return embedding
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Error getting cached embedding: {e}")
            return None

    def set_embedding(self, query: str, embedding: list[float]) -> bool:
        """
        Cache an embedding for a query.

        Args:
            query: The query string
            embedding: The embedding to cache

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            cache_key = self._generate_cache_key(query)
            frequency_key = self._generate_frequency_key(query)

            current_frequency = float(self.redis_client.get(frequency_key) or 1.0)
            ttl = self._get_ttl_for_frequency(current_frequency)

            self.redis_client.setex(cache_key, ttl, json.dumps(embedding))

            if current_frequency == 1.0:
                self.redis_client.setex(frequency_key, ttl, current_frequency)

            logger.debug(f"Cached embedding for query: {query[:50]}... with TTL: {ttl}s")
            return True

        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
            return False

    def _update_frequency(self, query: str, increment: bool = True) -> bool:
        """
        Update query frequency.

        Args:
            query: The query string
            increment: Whether to increment or decay frequency

        Returns:
            True if successfully updated, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            frequency_key = self._generate_frequency_key(query)
            current_frequency = float(self.redis_client.get(frequency_key) or 1.0)

            if increment:
                new_frequency = current_frequency * 1.1 + 0.1
            else:
                new_frequency = current_frequency * self.frequency_decay

            ttl = self._get_ttl_for_frequency(new_frequency)

            self.redis_client.setex(frequency_key, ttl, new_frequency)

            return True

        except Exception as e:
            logger.error(f"Error updating frequency: {e}")
            return False

    def clear_cache(self, pattern: str | None = None) -> bool:
        """
        Clear cached embeddings.

        Args:
            pattern: Pattern to match for cache keys (optional)

        Returns:
            True if successfully cleared, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache keys matching pattern: {pattern}")
            else:
                pattern = f"{self.cache_key_prefix}*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache keys")

            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.redis_client:
            return {"error": "Redis not connected"}

        try:
            cache_pattern = f"{self.cache_key_prefix}*"
            cache_keys = self.redis_client.keys(cache_pattern)

            frequency_pattern = f"{self.frequency_key_prefix}*"
            frequency_keys = self.redis_client.keys(frequency_pattern)

            if cache_keys:
                sample_size = min(10, len(cache_keys))
                sample_keys = cache_keys[:sample_size]
                ttl_values = []

                for key in sample_keys:
                    ttl = self.redis_client.ttl(key)
                    if ttl > 0:
                        ttl_values.append(ttl)

                avg_ttl = sum(ttl_values) / len(ttl_values) if ttl_values else 0
            else:
                avg_ttl = 0

            return {
                "cache_keys_count": len(cache_keys),
                "frequency_keys_count": len(frequency_keys),
                "average_ttl_seconds": avg_ttl,
                "cache_key_prefix": self.cache_key_prefix,
                "base_ttl": self.base_ttl,
                "max_ttl": self.max_ttl,
                "min_ttl": self.min_ttl,
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
