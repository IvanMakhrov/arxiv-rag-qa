import json

import boto3
from sentence_transformers import SentenceTransformer

from utils.setup_logger import setup_logger

# Logging setup
logger = setup_logger(__name__)


def get_minio_client():
    return boto3.client("s3")


def generate_embeddings(bucket_name: str, chunk_dir: str, embedding_dir: str, model_name: str):
    """Генерирует эмбеддинги для каждого чанка и сохраняет в новый JSONL"""

    s3_client = get_minio_client()

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=chunk_dir)
        chunk_data = response["Body"].read().decode("utf-8")
    except Exception as e:
        logger.error(f"Chunk file not found in s3://{bucket_name}/{chunk_dir}: {e}")
        raise Exception(f"Chunk file not found in s3://{bucket_name}/{chunk_dir}: {e}") from e

    texts = []
    records = []
    for line in chunk_data.strip().split("\n"):
        if line.strip():
            record = json.loads(line)
            records.append(record)
            texts.append(record["text"])

    if not texts:
        logger.error("No chunks found for embedding")
        raise ValueError("No chunks found for embedding")

    logger.info(f"Generating embeddings for {len(texts)} chunks using {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    jsonl_lines = []
    for record, emb in zip(records, embeddings, strict=False):
        record["embedding"] = emb.tolist()
        jsonl_lines.append(json.dumps(record, ensure_ascii=False))

    jsonl_content = "\n".join(jsonl_lines)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=embedding_dir,
        Body=jsonl_content.encode("utf-8"),
        ContentType="application/jsonl",
    )

    logger.info(f"Saved embeddings to s3://{bucket_name}/{embedding_dir}")

    return len(embeddings)
