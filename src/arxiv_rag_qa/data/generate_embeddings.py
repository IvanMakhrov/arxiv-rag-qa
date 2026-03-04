import json

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from sentence_transformers import SentenceTransformer

from utils.setup_logger import setup_logger

# Logging setup
logger = setup_logger(__name__)


def get_minio_client():
    """Initialize S3/MinIO client with retry config for stability."""
    return boto3.client(
        "s3",
        config=Config(
            retries={"max_attempts": 3, "mode": "standard"},
            read_timeout=300,
            connect_timeout=60,
        ),
    )


def _s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    """Check if an object exists in S3 without downloading it."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        logger.info(e.response["Error"]["Code"])
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def _load_checkpoint(s3_client, bucket: str, checkpoint_key: str) -> tuple[list[str], int]:
    """Load checkpoint from S3 and return (jsonl_lines_list, processed_count)."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=checkpoint_key)
        checkpoint_data = response["Body"].read().decode("utf-8")
        lines = [line for line in checkpoint_data.strip().split("\n") if line.strip()]
        logger.info(f"Resumed from checkpoint: {len(lines)} records already processed")
        return lines, len(lines)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.info("No checkpoint found, starting from scratch")
            return [], 0
        raise


def _save_checkpoint(s3_client, bucket: str, checkpoint_key: str, jsonl_lines: list[str]):
    """Save current progress to checkpoint file in S3."""
    content = "\n".join(jsonl_lines)
    s3_client.put_object(
        Bucket=bucket,
        Key=checkpoint_key,
        Body=content.encode("utf-8"),
        ContentType="application/jsonl",
    )
    logger.info(f"Checkpoint saved: {len(jsonl_lines)} records")


def generate_embeddings(  # noqa: C901
    bucket_name: str,
    chunk_dir: str,
    embedding_dir: str,
    model_name: str,
    batch_size: int,
    checkpoint_interval: int,
):
    """Generate embeddings for chunks and save to JSONL in S3."""
    s3_client = get_minio_client()
    checkpoint_key = f"{embedding_dir}.checkpoint"

    # ── 1. IDEMPOTENCY CHECK ─────────────────────────────────────
    if _s3_object_exists(s3_client, bucket_name, embedding_dir):
        logger.info(f"Embeddings already exist at s3://{bucket_name}/{embedding_dir}, skipping")
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=checkpoint_key)
        except Exception as e:
            logger.debug(f"Checkpoint cleanup skipped: {e}")
        return 0

    # ── 2. LOAD INPUT CHUNKS ─────────────────────────────────────
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=chunk_dir)
        chunk_data = response["Body"].read().decode("utf-8")
    except Exception as e:
        logger.error(f"Chunk file not found in s3://{bucket_name}/{chunk_dir}: {e}")
        raise RuntimeError(f"Chunk file not found: {chunk_dir}") from e

    all_records = []
    for line in chunk_data.strip().split("\n"):
        if line.strip():
            all_records.append(json.loads(line))

    if not all_records:
        logger.error("No chunks found for embedding")
        raise ValueError("No chunks found for embedding")

    total_records = len(all_records)
    logger.info(f"Loaded {total_records} chunks for embedding")

    # ── 3. RESUME FROM CHECKPOINT ────────────────────────────────
    jsonl_lines, processed_count = _load_checkpoint(s3_client, bucket_name, checkpoint_key)

    # ── 4. LOAD MODEL ONCE ───────────────────────────────────────
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # ── 5. BATCH PROCESSING LOOP ─────────────────────────────────
    for i in range(processed_count, total_records, batch_size):
        batch_end = min(i + batch_size, total_records)
        batch_records = all_records[i:batch_end]
        batch_texts = [r["text"] for r in batch_records]

        batch_num = i // batch_size + 1
        total_batches = (total_records + batch_size - 1) // batch_size
        logger.info(f"Batch {batch_num}/{total_batches} | Records {i} - {batch_end - 1}")

        embeddings = model.encode(
            batch_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32,
        )

        for record, emb in zip(batch_records, embeddings, strict=False):
            record["embedding"] = emb.tolist()
            jsonl_lines.append(json.dumps(record, ensure_ascii=False))

        # ── Checkpoint save ─────────────────────────────────────
        if batch_num % checkpoint_interval == 0 or batch_end == total_records:
            _save_checkpoint(s3_client, bucket_name, checkpoint_key, jsonl_lines)

    # ── 6. FINAL UPLOAD ──────────────────────────────────────────
    final_content = "\n".join(jsonl_lines)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=embedding_dir,
        Body=final_content.encode("utf-8"),
        ContentType="application/jsonl",
    )
    logger.info(f"Saved final embeddings to s3://{bucket_name}/{embedding_dir}")

    # ── 7. CLEANUP CHECKPOINT ────────────────────────────────────
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=checkpoint_key)
        logger.info("Checkpoint cleaned up")
    except Exception as e:
        logger.warning(f"Could not delete checkpoint: {e}")

    return len(jsonl_lines)
