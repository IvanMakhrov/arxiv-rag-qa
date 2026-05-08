import logging
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from arxiv_rag_qa.rag.generator import QwenGenerator
from arxiv_rag_qa.rag.hybrid_retriever import HybridRetriever
from arxiv_rag_qa.rag.pipeline import RAGPipeline
from arxiv_rag_qa.rag.retriever import DenseRetriever
from arxiv_rag_qa.rag.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)


def create_dense_retriever(config: dict[str, Any], qdrant_client=None) -> DenseRetriever:
    """Create a dense retriever instance."""
    embedder = SentenceTransformer(config.get("embedding_model", "all-MiniLM-L6-v2"))

    def embedding_model(x):
        return embedder.encode(
            x, normalize_embeddings=config.get("normalize_embeddings", True)
        ).tolist()

    return DenseRetriever(
        collection_name=config.get("collection_name", "rag_data"),
        embedding_model=embedding_model,
        top_k=config.get("top_k", 5),
        qdrant_host=config.get("qdrant_host", "localhost"),
        qdrant_port=config.get("qdrant_port", 6333),
        redis_host=config.get("redis_host", "localhost"),
        redis_port=config.get("redis_port", 6379),
        redis_db=config.get("redis_db", 0),
        enable_redis_cache=config.get("enable_redis_cache", True),
    )


def create_sparse_retriever(config: dict[str, Any], qdrant_client=None) -> SparseRetriever:
    """Create a sparse retriever instance."""
    return SparseRetriever(
        collection_name=config.get("collection_name", "arxiv_rag"),
        corpus=config.get("corpus"),
        top_k=config.get("top_k", 5),
        use_tfidf=config.get("method") == "tfidf",
        tfidf_params=config.get("tfidf_params"),
        bm25_params=config.get("bm25_params"),
    )


def create_hybrid_retriever(
    config: dict[str, Any],
    dense_config: dict[str, Any],
    sparse_config: dict[str, Any],
    qdrant_client=None,
) -> HybridRetriever:
    """Create a hybrid retriever instance."""
    dense_retriever = create_dense_retriever(dense_config, qdrant_client)
    sparse_retriever = create_sparse_retriever(sparse_config, qdrant_client)

    return HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion_method=config.get("fusion_method", "weighted_sum"),
        dense_weight=config.get("dense_weight", 0.7),
        sparse_weight=config.get("sparse_weight", 0.3),
        rank_fusion_k=config.get("rank_fusion_k", 100),
        normalize_scores=config.get("normalize_scores", True),
        deduplicate=config.get("deduplicate", True),
    )


def get_response(
    emb_model_name: str,
    collection_name: str,
    top_k: int,
    gen_model_name: str,
    query: str,
    qdrant_host: str,
    qdrant_port: int,
    retriever_type: str = "dense",
    **kwargs,
):
    """
    Get response from RAG system with support for different retriever types.

    Args:
        emb_model_name: Embedding model name
        collection_name: Qdrant collection name
        top_k: Number of documents to retrieve
        gen_model_name: Generator model name
        query: Search query
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        retriever_type: Type of retriever ("dense", "sparse", "hybrid")
        **kwargs: Additional parameters for retriever configuration

    Returns:
        Dictionary with RAG results
    """
    try:
        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    except Exception as e:
        logger.warning(f"Could not create Qdrant client: {e}")
        qdrant_client = None

    if retriever_type == "dense":
        dense_config = {
            "embedding_model": emb_model_name,
            "collection_name": collection_name,
            "top_k": top_k,
            "normalize_embeddings": True,
            "qdrant_host": qdrant_host,
            "qdrant_port": qdrant_port,
            "in_memory": kwargs.get("in_memory", False),
            "redis_host": kwargs.get("redis_host", "localhost"),
            "redis_port": kwargs.get("redis_port", 6379),
            "redis_db": kwargs.get("redis_db", 0),
            "enable_redis_cache": kwargs.get("enable_redis_cache", True),
        }
        retriever = create_dense_retriever(dense_config, qdrant_client)

    elif retriever_type == "sparse":
        sparse_section = kwargs.get("sparse", {})
        sparse_config = {
            "collection_name": collection_name,
            "top_k": top_k,
            "method": sparse_section.get("method", kwargs.get("sparse_method", "bm25")),
            "use_qdrant_corpus": sparse_section.get(
                "use_qdrant_corpus", kwargs.get("use_qdrant_corpus", True)
            ),
            "tfidf_params": sparse_section.get(
                "tfidf_params", kwargs.get("sparse_params", {}).get("tfidf_params")
            ),
            "bm25_params": sparse_section.get(
                "bm25_params", kwargs.get("sparse_params", {}).get("bm25_params")
            ),
        }
        retriever = create_sparse_retriever(sparse_config, qdrant_client)

    elif retriever_type == "hybrid":
        hybrid_section = kwargs.get("hybrid", {})
        sparse_section = kwargs.get("sparse", {})
        sparse_params = kwargs.get("sparse_params", {})

        hybrid_config = {
            "fusion_method": hybrid_section.get(
                "fusion_method", kwargs.get("fusion_method", "weighted_sum")
            ),
            "dense_weight": hybrid_section.get("dense_weight", kwargs.get("dense_weight", 0.7)),
            "sparse_weight": hybrid_section.get("sparse_weight", kwargs.get("sparse_weight", 0.3)),
            "rank_fusion_k": hybrid_section.get("rank_fusion_k", kwargs.get("rank_fusion_k", 100)),
            "normalize_scores": hybrid_section.get(
                "normalize_scores", kwargs.get("normalize_scores", True)
            ),
            "deduplicate": hybrid_section.get("deduplicate", kwargs.get("deduplicate", True)),
        }
        dense_config = {
            "embedding_model": emb_model_name,
            "collection_name": collection_name,
            "top_k": top_k * 2,
            "normalize_embeddings": True,
            "qdrant_host": qdrant_host,
            "qdrant_port": qdrant_port,
            "in_memory": kwargs.get("in_memory", False),
            "redis_host": kwargs.get("redis_host", "localhost"),
            "redis_port": kwargs.get("redis_port", 6379),
            "redis_db": kwargs.get("redis_db", 0),
            "enable_redis_cache": kwargs.get("enable_redis_cache", True),
        }
        sparse_config = {
            "collection_name": collection_name,
            "top_k": top_k * 2,
            "method": sparse_section.get("method", kwargs.get("sparse_method", "bm25")),
            "use_qdrant_corpus": sparse_section.get(
                "use_qdrant_corpus", kwargs.get("use_qdrant_corpus", True)
            ),
            "tfidf_params": sparse_section.get("tfidf_params", sparse_params.get("tfidf_params")),
            "bm25_params": sparse_section.get("bm25_params", sparse_params.get("bm25_params")),
        }
        retriever = create_hybrid_retriever(
            hybrid_config, dense_config, sparse_config, qdrant_client
        )

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    generator = QwenGenerator(
        model_name=gen_model_name,
        load_in_4bit=kwargs.get("load_in_4bit", True),
    )

    rag = RAGPipeline(retriever=retriever, generator=generator)
    return rag.run(query)


def get_response_advanced(config: dict[str, Any], query: str, qdrant_client=None, **kwargs):
    """
    Advanced RAG response function with full configuration support.

    Args:
        config: Full configuration dictionary
        query: Search query
        qdrant_client: Optional Qdrant client
        **kwargs: Additional override parameters

    Returns:
        Dictionary with RAG results
    """
    retriever_config = config.get("retriever", {})
    generator_config = config.get("generator", {})

    if kwargs:
        retriever_config = {**retriever_config, **kwargs}

    retriever_type = retriever_config.get("type", "dense")
    top_k = retriever_config.get("top_k", 5)

    collection_name = config.get("qdrant", {}).get("collection_name", "arxiv_rag")

    if retriever_type in ("dense", "hybrid"):
        emb_model_name = retriever_config.get("dense", {}).get(
            "embedding_model", config.get("embeddings", {}).get("default_model", "all-MiniLM-L6-v2")
        )
    else:
        emb_model_name = config.get("embeddings", {}).get("default_model", "all-MiniLM-L6-v2")

    gen_model_name = generator_config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")

    qdrant_config = config.get("qdrant", {})
    qdrant_host = qdrant_config.get("host", "localhost")
    qdrant_port = qdrant_config.get("port", 6333)

    sparse_section = retriever_config.get("sparse", {})
    hybrid_section = retriever_config.get("hybrid", {})
    dense_section = retriever_config.get("dense", {})

    extra_kwargs = {
        "sparse": sparse_section,
        "hybrid": hybrid_section,
        "dense": dense_section,
        "in_memory": kwargs.get("in_memory", False),
        "redis_host": kwargs.get("redis_host", "localhost"),
        "redis_port": kwargs.get("redis_port", 6379),
        "redis_db": kwargs.get("redis_db", 0),
        "enable_redis_cache": kwargs.get("enable_redis_cache", True),
    }

    return get_response(
        emb_model_name=emb_model_name,
        collection_name=collection_name,
        top_k=top_k,
        gen_model_name=gen_model_name,
        query=query,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        retriever_type=retriever_type,
        **extra_kwargs,
    )
