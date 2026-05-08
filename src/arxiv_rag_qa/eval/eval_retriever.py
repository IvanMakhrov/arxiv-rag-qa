import json
import math
from typing import Any

import boto3
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from arxiv_rag_qa.rag.hybrid_retriever import HybridRetriever
from arxiv_rag_qa.rag.retriever import DenseRetriever
from arxiv_rag_qa.rag.sparse_retriever import SparseRetriever
from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


def get_minio_client():
    return boto3.client("s3")


def load_test_set_from_minio(bucket_name: str, test_data_key: str, s3_client: Any) -> list:
    """Load test samples from MinIO JSONL file."""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=test_data_key)
        samples = []
        for line in response["Body"].iter_lines():
            if line:
                samples.append(json.loads(line.decode("utf-8")))
        return samples
    except Exception as e:
        logger.error(f"Test file not found in s3://{bucket_name}/{test_data_key}: {e}")
        raise FileNotFoundError(
            f"Test file not found in s3://{bucket_name}/{test_data_key}: {e}"
        ) from e


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Recall@k: For a single query, proportion of relevant chunks found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Precision@k: For a single query, proportion of relevant chunks among top-k results."""
    top_k = set(retrieved_ids[:k])
    if not top_k:
        return 0.0
    return len(top_k & relevant_ids) / k


def accuracy_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Accuracy@k: For a single query, whether at least one relevant chunk is in top-k."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if (top_k & relevant_ids) else 0.0


def f1_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    precision = precision_at_k(retrieved_ids, relevant_ids, k)
    recall = recall_at_k(retrieved_ids, relevant_ids, k)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def mrr_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    relevant_ids = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain with logarithmic discount."""
    if not relevant_ids:
        return 0.0

    relevance = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]

    if not relevance:
        return 0.0

    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / math.log2(i + 2)

    ideal_relevance = sorted(relevance, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def create_eval_retriever(
    retriever_type: str,
    collection_name: str,
    top_k: int,
    model_name: str,
    qdrant_host: str,
    qdrant_port: int,
    **kwargs,
):
    """
    Create a retriever for evaluation based on the specified type.

    Args:
        retriever_type: Type of retriever ("dense", "sparse", "hybrid")
        collection_name: Qdrant collection name
        top_k: Number of documents to retrieve
        model_name: Embedding model name (for dense/hybrid) or sparse method name
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        **kwargs: Additional retriever-specific configuration

    Returns:
        A retriever instance (DenseRetriever, SparseRetriever, or HybridRetriever)
    """
    if retriever_type == "dense":
        embedder = SentenceTransformer(model_name)

        def embedding_func(x):
            return embedder.encode(x, normalize_embeddings=True).tolist()

        return DenseRetriever(
            collection_name=collection_name,
            embedding_model=embedding_func,
            top_k=top_k,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
        )

    if retriever_type == "sparse":
        sparse_method = kwargs.get("sparse_method", "bm25")
        use_qdrant_corpus = kwargs.get("use_qdrant_corpus", True)
        sparse_params = kwargs.get("sparse_params", {})

        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        retriever = SparseRetriever(
            collection_name=collection_name,
            top_k=top_k,
            use_tfidf=sparse_method == "tfidf",
            tfidf_params=sparse_params.get("tfidf_params"),
            bm25_params=sparse_params.get("bm25_params"),
        )

        if use_qdrant_corpus:
            corpus = retriever._extract_corpus_from_qdrant(qdrant_client, collection_name)
            if corpus:
                retriever.corpus = corpus
                retriever.is_built = True
                retriever._build_index()

        return retriever

    if retriever_type == "hybrid":
        embedder = SentenceTransformer(model_name)

        def embedding_func(x):
            return embedder.encode(x, normalize_embeddings=True).tolist()

        dense_retriever = DenseRetriever(
            collection_name=collection_name,
            embedding_model=embedding_func,
            top_k=top_k,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
        )

        sparse_method = kwargs.get("sparse_method", "bm25")
        use_qdrant_corpus = kwargs.get("use_qdrant_corpus", True)
        sparse_params = kwargs.get("sparse_params", {})

        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        sparse_retriever = SparseRetriever(
            collection_name=collection_name,
            top_k=top_k,
            use_tfidf=sparse_method == "tfidf",
            tfidf_params=sparse_params.get("tfidf_params"),
            bm25_params=sparse_params.get("bm25_params"),
        )

        if use_qdrant_corpus:
            corpus = sparse_retriever._extract_corpus_from_qdrant(qdrant_client, collection_name)
            if corpus:
                sparse_retriever.corpus = corpus
                sparse_retriever.is_built = True
                sparse_retriever._build_index()

        hybrid_config = kwargs.get("hybrid_config", {})
        return HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion_method=hybrid_config.get("fusion_method", "weighted_sum"),
            dense_weight=hybrid_config.get("dense_weight", 0.7),
            sparse_weight=hybrid_config.get("sparse_weight", 0.3),
            rank_fusion_k=hybrid_config.get("rank_fusion_k", 100),
            normalize_scores=hybrid_config.get("normalize_scores", True),
            deduplicate=hybrid_config.get("deduplicate", True),
        )

    raise ValueError(f"Unknown retriever type: {retriever_type}")


def retriever_eval(  # noqa: PLR0913
    bucket_name: str,
    test_data_dir: str,
    collection_name: str,
    top_k: int,
    model_name: str,
    qdrant_host: str,
    qdrant_port: int,
    retriever_type: str = "dense",
    sparse_method: str = "bm25",
    use_qdrant_corpus: bool = True,
    hybrid_config: dict | None = None,
    sparse_params: dict | None = None,
):
    s3_client = get_minio_client()
    test_samples = load_test_set_from_minio(bucket_name, test_data_dir, s3_client)

    retriever = create_eval_retriever(
        retriever_type=retriever_type,
        collection_name=collection_name,
        top_k=top_k,
        model_name=model_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        sparse_method=sparse_method,
        use_qdrant_corpus=use_qdrant_corpus,
        hybrid_config=hybrid_config or {},
        sparse_params=sparse_params or {},
    )

    recall_scores = []
    precision_scores = []
    accuracy_scores = []
    f1_scores = []
    mrr_scores = []
    ndcg_scores = []

    for i, sample in enumerate(test_samples):
        logger.info(f"\rEvaluating {i + 1}/{len(test_samples)}")

        docs = retriever.retrieve(sample["question"], top_k=top_k)
        retrieved_ids = [doc["id"] for doc in docs]

        relevant_ids = set(sample["relevant_chunk_ids"])

        if not relevant_ids:
            continue

        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, top_k))
        precision_scores.append(precision_at_k(retrieved_ids, relevant_ids, top_k))
        accuracy_scores.append(accuracy_at_k(retrieved_ids, relevant_ids, top_k))
        f1_scores.append(f1_at_k(retrieved_ids, relevant_ids, top_k))
        mrr_scores.append(mrr_at_k(retrieved_ids, relevant_ids, top_k))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, top_k))

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    logger.info(
        f"Retriever type: {retriever_type}, Top-k: {top_k}, "
        f"Precision@{top_k}: {avg_precision:.4f}, "
        f"Recall@{top_k}: {avg_recall:.4f}, "
        f"Accuracy@{top_k}: {avg_accuracy:.4f}, "
        f"F1@{top_k}: {avg_f1:.4f}, "
        f"MRR@{top_k}: {avg_mrr:.4f}, "
        f"NDCG@{top_k}: {avg_ndcg:.4f} "
    )

    return {
        "config": {
            "test_file": test_data_dir,
            "collection_name": collection_name,
            "top_k": top_k,
            "embedder": model_name,
            "retriever_type": retriever_type,
            "sparse_method": sparse_method,
        },
        "metrics": {
            "precision_at_k": avg_precision,
            "recall_at_k": avg_recall,
            "accuracy_at_k": avg_accuracy,
            "f1_at_k": avg_f1,
            "mrr_at_k": avg_mrr,
            "ndcg_at_k": avg_ndcg,
        },
    }
