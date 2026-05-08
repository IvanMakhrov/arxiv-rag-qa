from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """Base interface for all retriever implementations."""

    def __init__(self, top_k: int = 5):
        """
        Initialize the retriever.

        Args:
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query
            top_k: Number of documents to retrieve (overrides instance default if provided)
            filter_: Optional filter to apply to search
            **kwargs: Additional parameters specific to the retriever

        Returns:
            List of retrieved documents, each with keys: 'id', 'score', 'payload'
        """
        pass

    @abstractmethod
    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        """
        Retrieve documents for multiple queries efficiently.

        Args:
            queries: List of search queries
            top_k: Number of documents to retrieve per query
            filter_: Optional filter to apply to search
            **kwargs: Additional parameters specific to the retriever

        Returns:
            List of lists of retrieved documents
        """
        pass

    def get_retriever_info(self) -> dict[str, Any]:
        """Get information about the retriever implementation."""
        return {
            "type": self.__class__.__name__,
            "top_k": self.top_k,
        }
