import json
import tempfile
import time
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError, HTTPClientError
from sentence_transformers import SentenceTransformer

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


def get_minio_client():
    return boto3.client(
        "s3",
        config=Config(
            retries={"max_attempts": 3, "mode": "standard"},
            read_timeout=300,
            connect_timeout=60,
        ),
    )


def _s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def _load_checkpoint_meta(s3_client, bucket: str, checkpoint_key: str) -> int:
    """Load only the count from checkpoint, not the full data."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=checkpoint_key)
        meta = json.loads(response["Body"].read().decode("utf-8"))
        return meta.get("processed_count", 0)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return 0
        raise


def _save_checkpoint_meta(
    s3_client,
    bucket: str,
    checkpoint_key: str,
    processed_count: int,
):
    """Save only metadata, not the full JSONL."""
    meta = {"processed_count": processed_count}
    s3_client.put_object(
        Bucket=bucket,
        Key=checkpoint_key,
        Body=json.dumps(meta).encode("utf-8"),
        ContentType="application/json",
    )


def _download_with_retry(s3_client, bucket: str, key: str, dest_path: Path, max_retries: int = 3):
    """Download S3 object to local file with retry logic."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading s3://{bucket}/{key} (attempt {attempt + 1})")
            s3_client.download_file(
                Bucket=bucket,
                Key=key,
                Filename=str(dest_path),
                Config=TransferConfig(
                    multipart_threshold=8 * 1024 * 1024,  # 8MB
                    multipart_chunksize=8 * 1024 * 1024,  # 8MB chunks
                    max_concurrency=4,
                    use_threads=True,
                ),
            )
            logger.info(f"Downloaded {dest_path.stat().st_size} bytes")
            return True
        except (ConnectionError, HTTPClientError, Exception) as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
    return False


def generate_embeddings(  # noqa: C901, PLR0912
    bucket_name: str,
    chunk_dir: str,
    embedding_dir: str,
    model_name: str,
    batch_size: int,
    checkpoint_interval: int,
):
    """Generate embeddings with robust file handling."""
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

    # ── 2. DOWNLOAD CHUNK FILE TO TEMP LOCATION ──────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        local_chunk_path = Path(tmpdir) / "chunks.jsonl"

        if not _download_with_retry(s3_client, bucket_name, chunk_dir, local_chunk_path):
            raise RuntimeError(f"Failed to download {chunk_dir} after retries")

        with Path.open(local_chunk_path, "r", encoding="utf-8") as f:
            total_records = sum(1 for line in f if line.strip())

        # ── 3. RESUME FROM CHECKPOINT ────────────────────────────
        processed_count = _load_checkpoint_meta(s3_client, bucket_name, checkpoint_key)
        if processed_count > 0:
            logger.info(
                f"Resuming from checkpoint: {processed_count}/{total_records} records processed"
            )

        # ── 4. LOAD MODEL ONCE ───────────────────────────────────
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        # ── 5. PROCESS LINE-BY-LINE FROM LOCAL FILE ──────────────
        jsonl_lines = []
        uploaded_count = processed_count

        with Path.open(local_chunk_path, "r", encoding="utf-8") as f:
            batch_texts = []
            batch_records = []

            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                if line_idx < processed_count:
                    continue

                record = json.loads(line)
                batch_texts.append(record["text"])
                batch_records.append(record)

                if len(batch_texts) >= batch_size:
                    embeddings = model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=32,
                    )

                    for record, emb in zip(batch_records, embeddings, strict=False):
                        record["embedding"] = emb.tolist()
                        jsonl_lines.append(json.dumps(record, ensure_ascii=False))

                    batch_texts.clear()
                    batch_records.clear()
                    uploaded_count += batch_size

                    if uploaded_count % (checkpoint_interval * batch_size) == 0:
                        _save_checkpoint_meta(
                            s3_client, bucket_name, checkpoint_key, uploaded_count
                        )
                        _upload_embeddings_incremental(
                            s3_client, bucket_name, embedding_dir, jsonl_lines
                        )
                        jsonl_lines.clear()
                    logger.info(f"{uploaded_count}/{total_records} embeddings uploaded")

            if batch_texts:
                embeddings = model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=32,
                )
                for record, emb in zip(batch_records, embeddings, strict=False):
                    record["embedding"] = emb.tolist()
                    jsonl_lines.append(json.dumps(record, ensure_ascii=False))
                uploaded_count += len(batch_texts)

        # ── 6. FINAL UPLOAD ──────────────────────────────────────
        if jsonl_lines:
            _upload_embeddings_incremental(s3_client, bucket_name, embedding_dir, jsonl_lines)

        logger.info(f"Saved {uploaded_count} embeddings to s3://{bucket_name}/{embedding_dir}")

        # ── 7. CLEANUP ───────────────────────────────────────────
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=checkpoint_key)
            logger.info("Checkpoint cleaned up")
        except Exception as e:
            logger.warning(f"Could not delete checkpoint: {e}")

        return uploaded_count


def _upload_embeddings_incremental(
    s3_client, bucket: str, key: str, jsonl_lines: list[str], append: bool = True
):
    """Upload new embeddings to S3, appending to existing object if needed."""
    if not jsonl_lines:
        return

    content = "\n".join(jsonl_lines) + "\n"

    if append:
        try:
            existing = s3_client.get_object(Bucket=bucket, Key=key)
            existing_content = existing["Body"].read().decode("utf-8")
            full_content = existing_content + content if existing_content.strip() else content
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                full_content = content
            else:
                raise
    else:
        full_content = content

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=full_content.encode("utf-8"),
        ContentType="application/jsonl",
    )
