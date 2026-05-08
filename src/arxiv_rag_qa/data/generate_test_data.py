import json
import random
import re
from collections import defaultdict
from typing import Any

import boto3

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


def get_minio_client():
    return boto3.client("s3")


def load_chunks_from_minio(
    bucket_name: str, chunk_key: str, s3_client: Any
) -> list[dict[str, Any]]:
    """Load chunked documents with metadata from MinIO."""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=chunk_key)
        chunks = []
        for line in response["Body"].iter_lines():
            if line:
                chunks.append(json.loads(line.decode("utf-8")))
        return chunks
    except Exception as e:
        logger.error(f"Chunk file not found in s3://{bucket_name}/{chunk_key}: {e}")
        raise FileNotFoundError(
            f"Chunk file not found in s3://{bucket_name}/{chunk_key}: {e}"
        ) from e


def load_metadata_from_minio(
    bucket_name: str, metadata_key: str, s3_client: Any
) -> dict[str, dict[str, Any]]:
    """Load metadata.json from MinIO and return {arxiv_id: paper_data} mapping."""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=metadata_key)
        papers = json.loads(response["Body"].read().decode("utf-8"))
        return {paper["arxiv_id"]: paper for paper in papers}
    except Exception as e:
        logger.error(f"Metadata file not found in s3://{bucket_name}/{metadata_key}: {e}")
        raise FileNotFoundError(
            f"Metadata file not found in s3://{bucket_name}/{metadata_key}: {e}"
        ) from e


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Remove version suffix from arXiv ID (e.g., '2310.12345v1' -> '2310.12345')."""
    if arxiv_id:
        return re.sub(r"v\d+$", "", arxiv_id)
    return arxiv_id


def build_paper_to_chunks(chunks: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Map arXiv ID (normalized) to list of chunk IDs belonging to that paper."""
    paper_to_chunks = {}
    for idx, chunk in enumerate(chunks):
        raw_arxiv_id = chunk["metadata"].get("arxiv_id")
        if raw_arxiv_id:
            arxiv_id = normalize_arxiv_id(raw_arxiv_id)
            if arxiv_id not in paper_to_chunks:
                paper_to_chunks[arxiv_id] = []
            paper_to_chunks[arxiv_id].append(idx)
    return paper_to_chunks


def generate_test_samples(  # noqa: C901
    chunks: list[dict[str, Any]],
    paper_to_chunks: dict[str, list[int]],
    metadata_map: dict[str, dict[str, Any]],
    min_abstract_len: int = 50,
    test_data_size: int = 0,
    max_questions_per_paper: int = 3,
) -> list[dict[str, Any]]:
    """Generate diverse Q&A pairs from paper chunks and metadata."""
    random.seed(42)  # For reproducibility

    templates = {
        "contribution": [
            "What is the main contribution of the paper titled '{title}'?",
            "What does the paper '{title}' contribute?",
            "What is the key insight presented in '{title}'?",
        ],
        "method": [
            "What method does the paper titled '{title}' propose?",
            "What approach is presented in '{title}'?",
            "What technique is introduced in the paper '{title}'?",
        ],
        "dataset": [
            "What dataset was used in the paper '{title}'?",
            "What data source is used in '{title}'?",
            "Which dataset is employed in the experiments of '{title}'?",
        ],
        "result": [
            "What were the main findings of '{title}'?",
            "What performance did the paper '{title}' achieve?",
            "What are the key results reported in '{title}'?",
        ],
        "limitation": [
            "What are the limitations mentioned in '{title}'?",
            "What future work is suggested in '{title}'?",
            "What drawbacks does the paper '{title}' acknowledge?",
        ],
        "other": [
            "What is discussed in the section from '{title}'?",
            "What does '{title}' say about this topic?",
        ],
    }

    keywords = {
        "method": [
            "propose",
            "approach",
            "method",
            "technique",
            "algorithm",
            "framework",
            "model",
            "architecture",
        ],
        "dataset": [
            "dataset",
            "data",
            "corpus",
            "collection",
            "evaluation set",
            "training set",
            "benchmark",
        ],
        "result": [
            "result",
            "performance",
            "accuracy",
            "f1",
            "score",
            "outperform",
            "experiment",
            "evaluation",
        ],
        "limitation": [
            "limitation",
            "drawback",
            "weakness",
            "future work",
            "failure case",
            "error",
        ],
    }

    def categorize_chunk(text: str) -> str:
        text_lower = text.lower()
        for category, words in keywords.items():
            if any(word in text_lower for word in words):
                return category
        return "other"

    total_papers = len(paper_to_chunks)
    if test_data_size > 0:
        target_papers = int(total_papers * 0.01 * test_data_size)
    else:
        target_papers = total_papers

    paper_ids = list(paper_to_chunks.keys())
    random.shuffle(paper_ids)
    selected_paper_ids = paper_ids[:target_papers]

    samples = []

    for arxiv_id in selected_paper_ids:
        if arxiv_id not in metadata_map:
            continue

        paper = metadata_map[arxiv_id]
        title = paper.get("title", "").strip()
        abstract = paper.get("abstract", "").strip()
        chunk_indices = paper_to_chunks[arxiv_id]

        if not title:
            continue

        paper_questions = []

        if abstract and len(abstract) >= min_abstract_len:
            question = random.choice(templates["contribution"]).format(title=title)
            paper_questions.append(
                {
                    "question": question,
                    "answer": abstract,
                    "relevant_chunk_ids": chunk_indices,
                    "category": "contribution",
                }
            )

        cat_to_indices = defaultdict(list)
        for idx in chunk_indices:
            chunk = chunks[idx]
            text = chunk.get("text", "")
            cat = categorize_chunk(text)
            cat_to_indices[cat].append(idx)

        categories = ["method", "dataset", "result", "limitation", "other"]
        for cat in categories:
            if len(paper_questions) >= max_questions_per_paper:
                break
            indices = cat_to_indices[cat]
            if not indices:
                continue
            chosen_idx = max(indices, key=lambda i: len(chunks[i].get("text", "")))
            chunk = chunks[chosen_idx]
            answer_text = chunk.get("text", "").strip()
            if len(answer_text) < 20:  # noqa: PLR2004
                continue
            question = random.choice(templates[cat]).format(title=title)
            paper_questions.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "relevant_chunk_ids": [chosen_idx],
                    "category": cat,
                }
            )

        samples.extend(paper_questions[:max_questions_per_paper])

    return samples


def generate_test_data(
    bucket_name: str,
    chunk_dir: str,
    test_data_dir: str,
    metadata_dir: str,
    test_data_size: int,
    max_questions_per_paper: int = 3,
):
    """Generate test data directly from chunks without using metadata."""
    s3_client = get_minio_client()
    chunks = load_chunks_from_minio(bucket_name, chunk_dir, s3_client)

    if not chunks:
        logger.error("No chunks found. Run ingestion first")
        raise ValueError("No chunks found. Run ingestion first")

    metadata_map_raw = load_metadata_from_minio(bucket_name, metadata_dir, s3_client)
    metadata_map = {
        normalize_arxiv_id(paper.get("arxiv_id", "")): paper
        for paper in metadata_map_raw.values()
        if paper.get("arxiv_id")
    }

    paper_to_chunks = build_paper_to_chunks(chunks)
    test_samples = generate_test_samples(
        chunks=chunks,
        paper_to_chunks=paper_to_chunks,
        metadata_map=metadata_map,
        test_data_size=test_data_size,
        max_questions_per_paper=max_questions_per_paper,
    )

    jsonl_content = "\n".join(json.dumps(sample, ensure_ascii=False) for sample in test_samples)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=test_data_dir,
        Body=jsonl_content.encode("utf-8"),
        ContentType="application/jsonl",
    )

    logger.info(f"Saved {len(test_samples)} test samples to s3://{bucket_name}/{test_data_dir}")

    return len(test_samples)
