import logging
import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


MIN_TOKEN_LENGTH = 2


class SparseRetriever(BaseRetriever):
    """Sparse retriever using BM25 and TF-IDF for keyword-based search."""

    def __init__(
        self,
        collection_name: str = "arxiv_rag",
        corpus: list[str] | None = None,
        corpus_payloads: list[dict[str, Any]] | None = None,
        top_k: int = 5,
        use_tfidf: bool = False,
        tfidf_params: dict[str, Any] | None = None,
        bm25_params: dict[str, Any] | None = None,
    ):
        """
        Initialize sparse retriever.

        Args:
            collection_name: Name of the collection (for metadata)
            corpus: List of document texts to index. If None, will load from Qdrant payload
            corpus_payloads: Full payload dicts corresponding to each corpus entry
            (preserves metadata)
            top_k: Number of documents to retrieve
            use_tfidf: Whether to use TF-IDF instead of BM25
            tfidf_params: Parameters for TF-IDF vectorizer
            bm25_params: Parameters for BM25
        """
        super().__init__(top_k=top_k)
        self.collection_name = collection_name
        self.use_tfidf = use_tfidf
        self.corpus = corpus or []
        self.corpus_payloads = corpus_payloads or []
        self.is_built = len(self.corpus) > 0

        self.tfidf_params = tfidf_params or {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.9,
        }

        self.bm25_params = bm25_params or {
            "k1": 1.2,
            "b": 0.75,
            "epsilon": 0.25,
        }

        self.vectorizer = None
        self.bm25 = None
        self.tfidf_matrix = None
        self.doc_ids = []

        if self.is_built:
            self._build_index()

    def _build_index(self):
        """Build the sparse index from corpus."""
        if not self.corpus:
            logger.warning("No corpus provided to build index")
            return

        logger.info(f"Building sparse index for {len(self.corpus)} documents...")

        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(**self.tfidf_params)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
            logger.info("TF-IDF index built")
        else:
            tokenized_corpus = []
            for doc in self.corpus:
                tokens = self._tokenize_text(doc)
                tokenized_corpus.append(tokens)

            self.bm25 = BM25Okapi(tokenized_corpus, **self.bm25_params)
            logger.info("BM25 index built")

    def _tokenize_text(self, text: str) -> list[str]:
        """Tokenize text for BM25."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return [token for token in tokens if len(token) > MIN_TOKEN_LENGTH]

    def _extract_corpus_from_qdrant(self, qdrant_client, collection_name: str) -> list[str]:
        """Extract document texts and full payloads from Qdrant collection."""
        try:
            response = qdrant_client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            corpus = []
            self.doc_ids = []
            self.corpus_payloads = []

            for point in response[0]:
                text = point.payload.get("text", "")
                if text:
                    corpus.append(text)
                    self.doc_ids.append(point.id)
                    self.corpus_payloads.append(point.payload)

            logger.info(f"Extracted {len(corpus)} documents from Qdrant")
            return corpus

        except Exception as e:
            logger.error(f"Failed to extract corpus from Qdrant: {e}")
            if "doesn't exist" in str(e) or "Not found" in str(e):
                logger.warning(
                    f"Collection '{collection_name}' not found. Please run Qdrant setup first."
                )
            return []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_: dict[str, Any] | None = None,
        use_qdrant_corpus: bool = False,
        qdrant_client=None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents using sparse search.

        Args:
            query: The search query
            top_k: Number of documents to retrieve
            filter_: Optional filter (currently not implemented for sparse search)
            use_qdrant_corpus: Whether to use corpus from Qdrant
            qdrant_client: Qdrant client to extract corpus if needed
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents with scores and metadata
        """
        k = top_k or self.top_k

        if use_qdrant_corpus and not self.is_built and qdrant_client:
            self.corpus = self._extract_corpus_from_qdrant(qdrant_client, self.collection_name)
            if self.corpus:
                self.is_built = True
                self._build_index()
            else:
                logger.warning("Could not extract corpus from Qdrant")
                return []

        if not self.is_built:
            logger.error("Sparse index not built. Provide corpus or use_qdrant_corpus=True")
            return []

        if self.use_tfidf:
            query_vec = self.vectorizer.transform([query])
            similarities = (query_vec * self.tfidf_matrix.T).toarray()[0]

            top_indices = np.argsort(similarities)[::-1][:k]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    payload = self.corpus_payloads[idx] if idx < len(self.corpus_payloads) else {}
                    results.append(
                        {
                            "id": self.doc_ids[idx] if self.doc_ids else idx,
                            "score": float(similarities[idx]),
                            "payload": {
                                "text": self.corpus[idx],
                                **payload,
                                "retriever_type": "tfidf",
                                "collection": self.collection_name,
                            },
                        }
                    )
        else:
            query_tokens = self._tokenize_text(query)
            bm25_scores = self.bm25.get_scores(query_tokens)

            top_indices = np.argsort(bm25_scores)[::-1][:k]

            results = []
            for idx in top_indices:
                if bm25_scores[idx] > 0:
                    payload = self.corpus_payloads[idx] if idx < len(self.corpus_payloads) else {}
                    results.append(
                        {
                            "id": self.doc_ids[idx] if self.doc_ids else idx,
                            "score": float(bm25_scores[idx]),
                            "payload": {
                                "text": self.corpus[idx],
                                **payload,
                                "retriever_type": "bm25",
                                "collection": self.collection_name,
                            },
                        }
                    )

        return results

    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_: dict[str, Any] | None = None,
        use_qdrant_corpus: bool = False,
        qdrant_client=None,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of search queries
            top_k: Number of documents to retrieve per query
            filter_: Optional filter
            use_qdrant_corpus: Whether to use corpus from Qdrant
            qdrant_client: Qdrant client
            **kwargs: Additional parameters

        Returns:
            List of lists of retrieved documents
        """
        return [
            self.retrieve(query, top_k, filter_, use_qdrant_corpus, qdrant_client, **kwargs)
            for query in queries
        ]

    def get_retriever_info(self) -> dict[str, Any]:
        """Get information about the sparse retriever."""
        info = super().get_retriever_info()
        info.update(
            {
                "sparse_method": "tfidf" if self.use_tfidf else "bm25",
                "corpus_size": len(self.corpus),
                "is_built": self.is_built,
                "tfidf_params": self.tfidf_params if self.use_tfidf else None,
                "bm25_params": self.bm25_params if not self.use_tfidf else None,
            }
        )
        return info
