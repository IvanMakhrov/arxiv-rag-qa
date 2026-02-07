import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from arxiv_rag_qa.rag.retriever import DenseRetriever


def load_test_set(path: str) -> list:
    samples = []
    with Path.open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


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
    test_data_dir: str,
    collection_name: str,
    top_k: int,
    model_name: str,
    qdrant_host: str,
    qdrant_port: int,
):
    test_path = Path(test_data_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    test_samples = load_test_set(str(test_path))

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
        print(f"\rEvaluating {i + 1}/{len(test_samples)}", end="", flush=True)

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

    print(f"Top-k: {top_k}")
    print(f"Recall@{top_k}:   {avg_recall:.4f}")
    print(f"MRR@{top_k}:      {avg_mrr:.4f}")
    print(f"Hit Rate@{top_k}: {avg_hit:.4f}")

    return {
        "config": {
            "test_file": test_data_dir,
            "collection_name": collection_name,
            "top_k": top_k,
            "embedder": model_name,
        },
        "metrics": {"recall_at_k": avg_recall, "mrr_at_k": avg_mrr, "hit_rate_at_k": avg_hit},
    }
