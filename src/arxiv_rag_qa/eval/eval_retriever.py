import json
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
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    relevant_ids = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    return 1.0 if (top_k & relevant_ids) else 0.0


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
    mrr_scores = []
    hit_rates = []

    for i, sample in enumerate(test_samples):
        logger.info(f"\rEvaluating {i + 1}/{len(test_samples)}")

        docs = retriever.retrieve(sample["question"], top_k=top_k)
        retrieved_ids = [doc["id"] for doc in docs]

        relevant_ids = set(sample["relevant_chunk_ids"])

        if not relevant_ids:
            continue

        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, top_k))
        mrr_scores.append(mrr_at_k(retrieved_ids, relevant_ids, top_k))
        hit_rates.append(hit_rate_at_k(retrieved_ids, relevant_ids, top_k))

    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    avg_hit = sum(hit_rates) / len(hit_rates)

    logger.info(
        f"Metrics: Top-k: {top_k}, Recall@{top_k}: {avg_recall:.4f}, "
        f"MRR@{top_k}: {avg_mrr:.4f}, Hit Rate@{top_k}: {avg_hit:.4f}"
    )

    return {
        "config": {
            "test_file": test_data_dir,
            "collection_name": collection_name,
            "top_k": top_k,
            "embedder": model_name,
        },
        "metrics": {"recall_at_k": avg_recall, "mrr_at_k": avg_mrr, "hit_rate_at_k": avg_hit},
    }
