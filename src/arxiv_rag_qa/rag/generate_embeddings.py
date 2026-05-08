import json
import tempfile
import time
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError, HTTPClientError
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

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
                    multipart_threshold=8 * 1024 * 1024,
                    multipart_chunksize=8 * 1024 * 1024,
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


def _is_jina_model(model_name: str) -> bool:
    """Check if the model is a Jina model that requires special handling."""
    return "jina" in model_name.lower()


def _get_device() -> str:
    """Detect available device: cuda if available, else cpu."""
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return "cuda"
    logger.info("CUDA not available, using CPU")
    return "cpu"


def _load_embedding_model(model_name: str) -> tuple:
    """
    Load embedding model with GPU support and Jina model detection.

    Returns:
        Tuple of (model, tokenizer, encode_func, is_jina, device)
        For SentenceTransformer models: tokenizer is None, is_jina is False
        For Jina models: both model and tokenizer are returned, is_jina is True
    """
    device = _get_device()
    is_jina = _is_jina_model(model_name)
    use_gpu = device == "cuda"

    if is_jina:
        logger.info(f"Loading Jina model: {model_name} with trust_remote_code=True")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if use_gpu:
            model = model.half()
            model = model.to(device)
            logger.info(f"Jina model moved to {device} with half precision")

        model.eval()
        return model, tokenizer, None, True, device

    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    if use_gpu:
        model = model.half()
        model = model.to(device)
        logger.info(f"SentenceTransformer model moved to {device} with half precision")

    return model, None, None, False, device


def _encode_texts(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    is_jina: bool,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode a batch of texts into embeddings.

    For Jina models, uses AutoModel + mean pooling + L2 normalization.
    For SentenceTransformer models, uses the built-in encode method.
    """
    if is_jina:
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    return model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
    )


def generate_embeddings_for_model(
    chunks: list[dict[str, Any]],
    model_config: dict[str, Any],
    batch_size: int,
    checkpoint_interval: int,
    temp_dir: Path,
) -> list[dict[str, Any]]:
    """Generate embeddings for a single model configuration."""
    model_name = model_config["name"]
    vector_size = model_config["vector_size"]
    field_name = model_config["field_name"]

    logger.info(f"Generating embeddings for model: {model_name} (vector_size: {vector_size})")

    model, tokenizer, _, is_jina, device = _load_embedding_model(model_name)

    embeddings_with_chunks = []
    processed_count = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        embs = _encode_texts(texts, model, tokenizer, is_jina, device)

        for chunk, emb in zip(batch, embs, strict=False):
            chunk_with_emb = chunk.copy()
            chunk_with_emb[field_name] = emb.tolist()
            embeddings_with_chunks.append(chunk_with_emb)
            processed_count += 1

            if processed_count % (checkpoint_interval * batch_size) == 0:
                _save_checkpoint_meta(
                    get_minio_client(), "temp-bucket", f"{model_name}.checkpoint", processed_count
                )
                logger.info(f"Processed {processed_count} chunks")

    del model
    if tokenizer is not None:
        del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared after embedding generation")

    logger.info(f"Completed {processed_count} embeddings for {model_name}")
    return embeddings_with_chunks


def generate_hybrid_embeddings(
    bucket_name: str,
    chunk_dir: str,
    embedding_config: dict[str, Any],
    models_config: list[dict[str, Any]],
    batch_size: int,
    checkpoint_interval: int,
) -> dict[str, list[dict[str, Any]]]:
    """Generate embeddings for multiple models.

    Args:
        bucket_name: S3 bucket name
        chunk_dir: Directory containing chunks JSONL
        embedding_config: Embedding generation configuration
        models_config: List of model configurations
        batch_size: Batch size for processing
        checkpoint_interval: Checkpoint interval

    Returns:
        Dictionary mapping model names to embeddings data
    """
    s3_client = get_minio_client()
    checkpoint_key = f"{embedding_config['embedding_dir']}.checkpoint"

    if _s3_object_exists(s3_client, bucket_name, embedding_config["embedding_dir"]):
        logger.info(
            "Embeddings already exist at "
            f"s3://{bucket_name}/{embedding_config['embedding_dir']}, skipping"
        )
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        local_chunk_path = Path(tmpdir) / "chunks.jsonl"

        if not _download_with_retry(s3_client, bucket_name, chunk_dir, local_chunk_path):
            raise RuntimeError(f"Failed to download {chunk_dir} after retries")

        with Path.open(local_chunk_path, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]

        logger.info(f"Loaded {len(chunks)} chunks")

        processed_count = _load_checkpoint_meta(s3_client, bucket_name, checkpoint_key)
        if processed_count > 0:
            logger.info(f"Resuming from checkpoint: {processed_count} chunks processed")

        all_embeddings = {}

        for model_config in models_config:
            model_name = model_config["name"]

            model_key = f"{embedding_config['embedding_dir']}_{model_name}.jsonl"
            if _s3_object_exists(s3_client, bucket_name, model_key):
                logger.info(f"Embeddings for {model_name} already exist, skipping")
                continue

            try:
                embeddings = generate_embeddings_for_model(
                    chunks, model_config, batch_size, checkpoint_interval, Path(tmpdir)
                )

                model_embeddings_path = Path(tmpdir) / f"{model_name}_embeddings.jsonl"
                with Path.open(model_embeddings_path, "w", encoding="utf-8") as f:
                    lines = (
                        json.dumps(emb_data, ensure_ascii=False) + "\n" for emb_data in embeddings
                    )
                    f.writelines(lines)

                s3_client.upload_file(str(model_embeddings_path), bucket_name, model_key)

                all_embeddings[model_name] = embeddings

                try:
                    s3_client.delete_object(Bucket=bucket_name, Key=f"{model_name}.checkpoint")
                except Exception as e:
                    logger.debug(f"Checkpoint cleanup skipped: {e}")

                logger.info(
                    f"Saved {len(embeddings)} embeddings for {model_name} to s3://{bucket_name}/{model_key}"
                )

            except Exception as e:
                logger.error(f"Failed to generate embeddings for {model_name}: {e}")
                continue

        try:
            s3_client.delete_object(Bucket=bucket_name, Key=checkpoint_key)
        except Exception as e:
            logger.warning(f"Could not delete checkpoint: {e}")

        return all_embeddings


def generate_embeddings_single_model(  # noqa: C901, PLR0912, PLR0915
    bucket_name: str,
    chunk_dir: str,
    embedding_dir: str,
    model_name: str,
    batch_size: int,
    checkpoint_interval: int,
) -> int:
    s3_client = get_minio_client()
    checkpoint_key = f"{embedding_dir}.checkpoint"

    if _s3_object_exists(s3_client, bucket_name, embedding_dir):
        logger.info(f"Embeddings already exist at s3://{bucket_name}/{embedding_dir}, skipping")
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=checkpoint_key)
        except Exception as e:
            logger.debug(f"Checkpoint cleanup skipped: {e}")
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        local_chunk_path = Path(tmpdir) / "chunks.jsonl"

        if not _download_with_retry(s3_client, bucket_name, chunk_dir, local_chunk_path):
            raise RuntimeError(f"Failed to download {chunk_dir} after retries")

        with Path.open(local_chunk_path, "r", encoding="utf-8") as f:
            total_records = sum(1 for line in f if line.strip())

        processed_count = _load_checkpoint_meta(s3_client, bucket_name, checkpoint_key)
        if processed_count > 0:
            logger.info(
                f"Resuming from checkpoint: {processed_count}/{total_records} records processed"
            )

        logger.info(f"Loading model: {model_name}")
        model, tokenizer, _, is_jina, device = _load_embedding_model(model_name)

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
                    embeddings = _encode_texts(batch_texts, model, tokenizer, is_jina, device)

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
                embeddings = _encode_texts(batch_texts, model, tokenizer, is_jina, device)
                for record, emb in zip(batch_records, embeddings, strict=False):
                    record["embedding"] = emb.tolist()
                    jsonl_lines.append(json.dumps(record, ensure_ascii=False))
                uploaded_count += len(batch_texts)

        if jsonl_lines:
            _upload_embeddings_incremental(s3_client, bucket_name, embedding_dir, jsonl_lines)

        logger.info(f"Saved {uploaded_count} embeddings to s3://{bucket_name}/{embedding_dir}")

        try:
            s3_client.delete_object(Bucket=bucket_name, Key=checkpoint_key)
            logger.info("Checkpoint cleaned up")
        except Exception as e:
            logger.warning(f"Could not delete checkpoint: {e}")

        del model
        if tokenizer is not None:
            del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared after embedding generation")

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
