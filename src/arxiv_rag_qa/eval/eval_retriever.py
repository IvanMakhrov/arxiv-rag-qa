import json
import math
from typing import Any

import boto3
from sentence_transformers import SentenceTransformer

from arxiv_rag_qa.rag.retriever import DenseRetriever
from utils.setup_logger import setup_logger

# Logging setup
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

    # Create a relevance mapping (binary relevance: 1 if relevant, 0 otherwise)
    relevance = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]

    if not relevance:
        return 0.0

    # Calculate DCG with logarithmic discount
    dcg = 0.0
    for i, rel in enumerate(relevance):
        # Using log2(i+2) for 0-based index (i+1 would be rank, i+2 = rank+1)
        dcg += rel / math.log2(i + 2)

    # Calculate IDCG (ideal DCG)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def retriever_eval(
    bucket_name: str,
    test_data_dir: str,
    collection_name: str,
    top_k: int,
    model_name: str,
    qdrant_host: str,
    qdrant_port: int,
):
    s3_client = get_minio_client()
    test_samples = load_test_set_from_minio(bucket_name, test_data_dir, s3_client)

    embedder = SentenceTransformer(model_name)

    def embedding_func(x):
        return embedder.encode(x, normalize_embeddings=True).tolist()

    retriever = DenseRetriever(
        collection_name=collection_name,
        embedding_model=embedding_func,
        top_k=top_k,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
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
        f"Metrics: Top-k: {top_k}, "
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
