from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest

from arxiv_rag_qa.rag.base_retriever import BaseRetriever
from arxiv_rag_qa.redis_cache import RedisCacheManager
from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        collection_name: str = "rag_data",
        embedding_model=None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        top_k: int = 5,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        enable_redis_cache: bool = True,
    ):
        """
        Initialize DenseRetriever with optional Redis caching.

        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Function to generate embeddings
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
            top_k: Number of results to return
            redis_host: Redis host for caching
            redis_port: Redis port for caching
            redis_db: Redis database for caching
            enable_redis_cache: Whether to enable Redis caching
        """
        super().__init__(top_k=top_k)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.enable_redis_cache = enable_redis_cache

        self.redis_cache = None
        if enable_redis_cache:
            try:
                self.redis_cache = RedisCacheManager(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                )
                logger.info("Redis caching enabled for DenseRetriever")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}. Caching disabled.")
                self.enable_redis_cache = False

        if self.embedding_model is None:
            raise ValueError("embedding_model must be provided (callable that returns list[float])")

    def embed(self, query: str) -> list[float]:
        """
        Generate embedding for a query with Redis caching.

        Args:
            query: The query string

        Returns:
            Embedding as list of floats
        """
        if self.enable_redis_cache and self.redis_cache:
            cached_embedding = self.redis_cache.get_embedding(query)
            if cached_embedding is not None:
                logger.debug(f"Using cached embedding for query: {query[:50]}...")
                return cached_embedding

        logger.debug(f"Generating new embedding for query: {query[:50]}...")
        embedding = self.embedding_model(query)

        if self.enable_redis_cache and self.redis_cache:
            try:
                self.redis_cache.set_embedding(query, embedding)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")

        return embedding

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_: dict[str, Any] | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a query.

        Returns:
            List of hits, each with keys: 'id', 'score', 'payload'
        """
        try:
            k = top_k if top_k is not None else self.top_k
            query_vector = self.embed(query)

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=filter_,
                limit=k,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
        except Exception as e:
            logger.error(f"Error getting data from vector db: {e}")
            if "doesn't exist" in str(e) or "Not found" in str(e):
                logger.warning(
                    f"Collection '{self.collection_name}' not found. Please run Qdrant setup first."
                )
            return []

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        """Retrieve for multiple queries efficiently."""
        k = top_k if top_k is not None else self.top_k
        query_vectors = [self.embed(q) for q in queries]

        try:
            results = self.client.search_batch(
                collection_name=self.collection_name,
                requests=[
                    SearchRequest(
                        vector=vec,
                        filter=filter_,
                        limit=k,
                        with_payload=True,
                    )
                    for vec in query_vectors
                ],
            )
        except Exception as e:
            logger.error(f"Error getting data from vector db: {e}")
            if "doesn't exist" in str(e) or "Not found" in str(e):
                logger.warning(
                    f"Collection '{self.collection_name}' not found. Please run Qdrant setup first."
                )
            return []

        batch_hits = []
        for result in results:
            hits = [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in result]
            batch_hits.append(hits)
        return batch_hits

    def get_cache_stats(self) -> dict:
        """Get Redis cache statistics if enabled."""
        if self.enable_redis_cache and self.redis_cache:
            return self.redis_cache.get_cache_stats()
        return {"error": "Redis caching not enabled"}

    def clear_cache(self, pattern: str | None = None) -> bool:
        """Clear Redis cache if enabled."""
        if self.enable_redis_cache and self.redis_cache:
            return self.redis_cache.clear_cache(pattern)
        return False
