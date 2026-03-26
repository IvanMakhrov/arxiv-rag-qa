import json
import re
from typing import Any

import boto3
import fitz

from utils.setup_logger import setup_logger

# Logging setup
logger = setup_logger(__name__)


def get_minio_client() -> boto3.client:
    return boto3.client("s3")


def extract_full_text_with_pymupdf(pdf_bytes: bytes) -> str:
    """
    Извлечение полного текста из PDF-документа в памяти.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text("text")
    doc.close()
    return full_text


def preprocess_text(text: str) -> str:
    """
    Базовая предобработка текста:
    - Удаление переносов слов через дефис
    - Нормализация пробелов и переносов строк
    - Удаление лишних цифр-нумераторов в начале строк
    """
    text = re.sub(r"-\n\s*", "", text)
    text = re.sub(r"\n[1-9]\s+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_recursive(  # noqa: C901
    text: str, chunk_size: int = 512, chunk_overlap: int = 50, separators: list[str] | None = None
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current_chunk = ""

            for part in parts:
                candidate = current_chunk + part + (sep if sep != " " else "")

                if len(candidate) <= chunk_size or not current_chunk:
                    current_chunk = candidate
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.rstrip())
                    current_chunk = part + (sep if sep != " " else "")

            if current_chunk.strip():
                chunks.append(current_chunk.rstrip())

            final_chunks = []
            for chunk in chunks:
                if len(chunk) > chunk_size:
                    final_chunks.extend(
                        split_text_recursive(chunk, chunk_size, chunk_overlap, separators)
                    )
                elif chunk.strip():
                    final_chunks.append(chunk)

            return final_chunks

    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]


def chunking(
    bucket_name: str,
    pdf_dir: str,
    chunk_dir: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> int:
    """
    Полный пайплайн: загрузка PDF из MinIO → извлечение текста → чанкинг → сохранение JSONL.
    """

    s3_client = get_minio_client()

    paginator = s3_client.get_paginator("list_objects_v2")
    pdf_files = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=pdf_dir):
        if "Contents" in page:
            pdf_files.extend(
                [obj["Key"] for obj in page["Contents"] if obj["Key"].endswith(".pdf")]
            )

    if not pdf_files:
        logger.warning(f"No PDF files found in s3://{bucket_name}/{pdf_dir}")
        return 0

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    all_chunks: list[dict[str, Any]] = []

    for pdf_key in pdf_files:
        try:
            arxiv_id = pdf_key.rstrip(".pdf").split("/")[-1]

            logger.info(f"Processing: {pdf_key}")
            pdf_obj = s3_client.get_object(Bucket=bucket_name, Key=pdf_key)
            pdf_bytes = pdf_obj["Body"].read()

            full_text = extract_full_text_with_pymupdf(pdf_bytes)
            if not full_text.strip():
                logger.warning(f"Empty text extracted from {pdf_key}")
                continue

            cleaned_text = preprocess_text(full_text)

            chunks = split_text_recursive(
                cleaned_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            for i, chunk_text in enumerate(chunks):
                if chunk_text.strip():
                    all_chunks.append(
                        {
                            "id": f"{arxiv_id}_chunk_{i:04d}",
                            "text": chunk_text,
                            "metadata": {
                                "arxiv_id": arxiv_id,
                                "source": pdf_key,
                                "chunk_idx": i,
                                "chunk_size": len(chunk_text),
                            },
                        }
                    )

            logger.info(f"  → {len(chunks)} chunks from {arxiv_id}")

        except Exception as e:
            logger.error(f"Failed to process {pdf_key}: {type(e).__name__}: {e}")
            continue

    if all_chunks:
        jsonl_lines = [json.dumps(chunk, ensure_ascii=False) for chunk in all_chunks]
        jsonl_content = "\n".join(jsonl_lines)

        s3_client.put_object(
            Bucket=bucket_name,
            Key=chunk_dir,
            Body=jsonl_content.encode("utf-8"),
            ContentType="application/x-ndjson",
        )

        logger.info(f"Saved {len(all_chunks)} chunks to s3://{bucket_name}/{chunk_dir}")
        return len(all_chunks)

    logger.warning("No chunks were generated")
    return 0
