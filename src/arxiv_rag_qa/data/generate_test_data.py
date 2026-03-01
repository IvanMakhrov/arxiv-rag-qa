import json
from typing import Any

import boto3

from utils.setup_logger import setup_logger

# Logging setup
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


def build_paper_to_chunks(chunks: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Map arXiv ID to list of chunk IDs belonging to that paper."""
    paper_to_chunks = {}
    for idx, chunk in enumerate(chunks):
        arxiv_id = chunk["metadata"].get("arxiv_id")
        if arxiv_id:
            if arxiv_id not in paper_to_chunks:
                paper_to_chunks[arxiv_id] = []
            paper_to_chunks[arxiv_id].append(idx)
    return paper_to_chunks


def generate_test_samples(
    chunks: list[dict[str, Any]],
    paper_to_chunks: dict[str, list[int]],
    metadata_map: dict[str, dict[str, Any]],
    min_abstract_len: int = 50,
    test_data_size: int = 0,
) -> list[dict[str, Any]]:
    """
    Generate static Q&A pairs.
    Question: "What is the main contribution of the paper titled '<TITLE>'?"
    Answer: Abstract (if long enough)
    """
    seen_papers = set()
    samples = []
    test_size = int(len(paper_to_chunks) * 0.01 * test_data_size)

    for chunk in chunks:
        arxiv_id = chunk["metadata"].get("arxiv_id")

        if len(seen_papers) >= test_size:
            break

        if not arxiv_id or arxiv_id in seen_papers:
            continue

        paper = metadata_map.get(arxiv_id)
        if not paper:
            continue

        title = paper.get("title", "").strip()
        abstract = paper.get("abstract", "").strip()

        if not title or not abstract or len(abstract) < min_abstract_len:
            continue

        question = f"What is the main contribution of the paper titled '{title}'?"
        relevant_chunk_ids = paper_to_chunks.get(arxiv_id, [])

        if not relevant_chunk_ids:
            continue

        samples.append(
            {
                "question": question,
                "answer": abstract,
                "arxiv_id": arxiv_id,
                "relevant_chunk_ids": relevant_chunk_ids,
                "title": title,
            }
        )

        seen_papers.add(arxiv_id)

    return samples


def generate_test_data(
    bucket_name: str, chunk_dir: str, test_data_dir: str, metadata_dir: str, test_data_size: int
):
    s3_client = get_minio_client()
    chunks = load_chunks_from_minio(bucket_name, chunk_dir, s3_client)

    if not chunks:
        logger.error("No chunks found. Run ingestion first")
        raise ValueError("No chunks found. Run ingestion first")

    metadata_map = load_metadata_from_minio(bucket_name, metadata_dir, s3_client)
    paper_to_chunks = build_paper_to_chunks(chunks)
    test_samples = generate_test_samples(
        chunks=chunks,
        paper_to_chunks=paper_to_chunks,
        metadata_map=metadata_map,
        test_data_size=test_data_size,
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
