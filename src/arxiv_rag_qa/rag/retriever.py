from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchRequest


class DenseRetriever:
    def __init__(
        self,
        collection_name: str = "arxiv_rag",
        embedding_model=None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        top_k: int = 5,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

        if self.embedding_model is None:
            raise ValueError("embedding_model must be provided (callable that returns list[float])")

    def embed(self, query: str) -> list[float]:
        """Generate embedding for a query."""
        return self.embedding_model(query)

    def retrieve(
        self,
        query: str,
        top_k: int = 0,
        filter_: Filter = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a query.

        Returns:
            List of hits, each with keys: 'id', 'score', 'payload'
        """
        k = top_k or self.top_k
        query_vector = self.embed(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter_,
            limit=k,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]

    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int = 0,
        filter_: Filter = None,
    ) -> list[list[dict[str, Any]]]:
        """Retrieve for multiple queries efficiently."""
        k = top_k or self.top_k
        query_vectors = [self.embed(q) for q in queries]

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

        batch_hits = []
        for result in results:
            hits = [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in result]
            batch_hits.append(hits)
        return batch_hits
