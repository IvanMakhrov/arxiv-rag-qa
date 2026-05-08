import logging
from typing import Any

from .base_retriever import BaseRetriever
from .retriever import DenseRetriever
from .sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)

WEIGHT_SUM_TOLERANCE = 1e-6


class HybridRetriever(BaseRetriever):
    """Hybrid retriever that combines dense and sparse retrieval with score fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion_method: str = "weighted_sum",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rank_fusion_k: int = 100,
        normalize_scores: bool = True,
        deduplicate: bool = True,
        **kwargs,
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_retriever: Dense retriever instance
            sparse_retriever: Sparse retriever instance
            fusion_method: Method for fusing scores ('weighted_sum', 'rank_fusion', 'rrf')
            dense_weight: Weight for dense retriever scores (0-1)
            sparse_weight: Weight for sparse retriever scores (0-1)
            rank_fusion_k: Parameter for rank fusion methods
            normalize_scores: Whether to normalize scores before fusion
            deduplicate: Whether to remove duplicate documents
            **kwargs: Additional parameters
        """
        super().__init__(top_k=dense_retriever.top_k)

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rank_fusion_k = rank_fusion_k
        self.normalize_scores = normalize_scores
        self.deduplicate = deduplicate

        if not (0 <= dense_weight <= 1 and 0 <= sparse_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if abs(dense_weight + sparse_weight - 1.0) > WEIGHT_SUM_TOLERANCE:
            logger.warning("Weights don't sum to 1, results may not be optimal")

    def _normalize_scores(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize scores to [0, 1] range."""
        if not results:
            return results

        scores = [doc["score"] for doc in results]
        min_score, max_score = min(scores), max(scores)

        if max_score == min_score:
            normalized_score = 0.5
            return [{**doc, "score": normalized_score} for doc in results]

        normalized_results = []
        for doc in results:
            normalized_score = (doc["score"] - min_score) / (max_score - min_score)
            normalized_results.append({**doc, "score": normalized_score})

        return normalized_results

    def _weighted_sum_fusion(
        self, dense_results: list[dict[str, Any]], sparse_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fuse scores using weighted sum."""
        doc_scores = {}

        for doc in dense_results:
            doc_id = doc["id"]
            doc_scores[doc_id] = {
                "score": doc["score"] * self.dense_weight,
                "payload": doc["payload"],
                "sources": ["dense"],
            }

        for doc in sparse_results:
            doc_id = doc["id"]
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += doc["score"] * self.sparse_weight
                doc_scores[doc_id]["sources"].append("sparse")
            else:
                doc_scores[doc_id] = {
                    "score": doc["score"] * self.sparse_weight,
                    "payload": doc["payload"],
                    "sources": ["sparse"],
                }

        fused_results = []
        for doc_id, doc_data in doc_scores.items():
            fused_results.append(
                {
                    "id": doc_id,
                    "score": doc_data["score"],
                    "payload": doc_data["payload"],
                    "sources": doc_data["sources"],
                }
            )

        fused_results.sort(key=lambda x: x["score"], reverse=True)

        return fused_results

    def _rank_fusion(
        self, dense_results: list[dict[str, Any]], sparse_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fuse scores using rank fusion (weighted rank averaging)."""
        dense_ranks = {doc["id"]: rank for rank, doc in enumerate(dense_results, 1)}
        sparse_ranks = {doc["id"]: rank for rank, doc in enumerate(sparse_results, 1)}

        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        fused_scores = {}
        for doc_id in all_doc_ids:
            dense_rank = dense_ranks.get(doc_id, self.rank_fusion_k)
            sparse_rank = sparse_ranks.get(doc_id, self.rank_fusion_k)

            fused_score = (self.dense_weight / dense_rank) + (self.sparse_weight / sparse_rank)
            fused_scores[doc_id] = {
                "score": fused_score,
                "payload": self._get_payload_by_id(doc_id, dense_results, sparse_results),
                "sources": self._get_sources_by_id(doc_id, dense_results, sparse_results),
            }

        fused_results = []
        for doc_id, doc_data in fused_scores.items():
            fused_results.append(
                {
                    "id": doc_id,
                    "score": doc_data["score"],
                    "payload": doc_data["payload"],
                    "sources": doc_data["sources"],
                }
            )

        fused_results.sort(key=lambda x: x["score"], reverse=True)

        return fused_results

    def _rrf_fusion(
        self, dense_results: list[dict[str, Any]], sparse_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fuse scores using Reciprocal Rank Fusion (RRF)."""
        k = self.rank_fusion_k

        dense_ranks = {doc["id"]: rank for rank, doc in enumerate(dense_results, 1)}
        sparse_ranks = {doc["id"]: rank for rank, doc in enumerate(sparse_results, 1)}

        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        rrf_scores = {}
        for doc_id in all_doc_ids:
            dense_rank = dense_ranks.get(doc_id, k)
            sparse_rank = sparse_ranks.get(doc_id, k)

            rrf_score = (self.dense_weight / (k + dense_rank)) + (
                self.sparse_weight / (k + sparse_rank)
            )
            rrf_scores[doc_id] = {
                "score": rrf_score,
                "payload": self._get_payload_by_id(doc_id, dense_results, sparse_results),
                "sources": self._get_sources_by_id(doc_id, dense_results, sparse_results),
            }

        fused_results = []
        for doc_id, doc_data in rrf_scores.items():
            fused_results.append(
                {
                    "id": doc_id,
                    "score": doc_data["score"],
                    "payload": doc_data["payload"],
                    "sources": doc_data["sources"],
                }
            )

        fused_results.sort(key=lambda x: x["score"], reverse=True)

        return fused_results

    def _get_payload_by_id(
        self, doc_id: str, dense_results: list[dict], sparse_results: list[dict]
    ) -> dict[str, Any]:
        """Get payload for a document ID from either dense or sparse results."""
        for doc in dense_results:
            if doc["id"] == doc_id:
                return doc["payload"]

        for doc in sparse_results:
            if doc["id"] == doc_id:
                return doc["payload"]

        return {}

    def _get_sources_by_id(
        self, doc_id: str, dense_results: list[dict], sparse_results: list[dict]
    ) -> list[str]:
        """Get source list for a document ID."""
        sources = []

        for doc in dense_results:
            if doc["id"] == doc_id:
                sources.append("dense")

        for doc in sparse_results:
            if doc["id"] == doc_id:
                sources.append("sparse")

        return sources

    def _deduplicate_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate documents while keeping the highest score."""
        seen_ids = set()
        deduplicated = []

        for doc in results:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                deduplicated.append(doc)

        return deduplicated

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_: dict[str, Any] | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search.

        Args:
            query: The search query
            top_k: Number of documents to return after fusion
            filter_: Optional filter (passed to retrievers if supported)
            dense_top_k: Number of documents to retrieve from dense retriever
            sparse_top_k: Number of documents to retrieve from sparse retriever
            **kwargs: Additional parameters

        Returns:
            List of fused retrieved documents
        """
        k = top_k or self.top_k
        dense_k = dense_top_k or max(k * 2, 10)
        sparse_k = sparse_top_k or max(k * 2, 10)

        dense_results = self.dense_retriever.retrieve(
            query, top_k=dense_k, filter_=filter_, **kwargs
        )
        sparse_results = self.sparse_retriever.retrieve(
            query, top_k=sparse_k, filter_=filter_, **kwargs
        )

        logger.info(f"Dense retriever returned {len(dense_results)} documents")
        logger.info(f"Sparse retriever returned {len(sparse_results)} documents")

        if self.normalize_scores:
            dense_results = self._normalize_scores(dense_results)
            sparse_results = self._normalize_scores(sparse_results)

        if self.fusion_method == "weighted_sum":
            fused_results = self._weighted_sum_fusion(dense_results, sparse_results)
        elif self.fusion_method == "rank_fusion":
            fused_results = self._rank_fusion(dense_results, sparse_results)
        elif self.fusion_method == "rrf":
            fused_results = self._rrf_fusion(dense_results, sparse_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        if self.deduplicate:
            fused_results = self._deduplicate_results(fused_results)

        return fused_results[:k]

    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_: dict[str, Any] | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        """
        Retrieve documents for multiple queries using hybrid search.

        Args:
            queries: List of search queries
            top_k: Number of documents to return per query after fusion
            filter_: Optional filter
            dense_top_k: Number of documents to retrieve from dense retriever per query
            sparse_top_k: Number of documents to retrieve from sparse retriever per query
            **kwargs: Additional parameters

        Returns:
            List of lists of fused retrieved documents
        """
        return [
            self.retrieve(query, top_k, filter_, dense_top_k, sparse_top_k, **kwargs)
            for query in queries
        ]

    def get_retriever_info(self) -> dict[str, Any]:
        """Get information about the hybrid retriever."""
        info = super().get_retriever_info()
        info.update(
            {
                "fusion_method": self.fusion_method,
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight,
                "normalize_scores": self.normalize_scores,
                "deduplicate": self.deduplicate,
                "rank_fusion_k": self.rank_fusion_k,
                "dense_retriever_info": self.dense_retriever.get_retriever_info(),
                "sparse_retriever_info": self.sparse_retriever.get_retriever_info(),
            }
        )
        return info
