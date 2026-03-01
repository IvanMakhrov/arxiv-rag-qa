import json
from typing import Any

import boto3
import fitz

from utils.setup_logger import setup_logger

# Logging setup
logger = setup_logger(__name__)


def get_minio_client():
    return boto3.client("s3")


def extract_full_text_with_pymupdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text("text")
    doc.close()
    return full_text


def load_metadata_from_minio(bucket: str, metadata_key: str, s3_client: Any):
    """Load metadata JSON from MinIO."""
    response = s3_client.get_object(Bucket=bucket, Key=metadata_key)
    metadata = json.loads(response["Body"].read().decode("utf-8"))
    logger.info(f"Metadata loaded from s3://{bucket}/{metadata_key}")
    return metadata


def parse_pdfs_to_json(
    bucket_name: str,
    metadata_dir: str,
    json_dir: str,
) -> int:
    """
    Convert all PDFs in raw_pdf_dir to JSON files with full text.
    Returns number of parsed papers.
    """
    s3_client = get_minio_client()
    metadata = load_metadata_from_minio(bucket_name, metadata_dir, s3_client)

    if not metadata:
        logger.error(f"No metadata found in s3://{bucket_name}/{metadata_dir}")
        raise ValueError(f"No metadata found in s3://{bucket_name}/{metadata_dir}")

    parsed_count = 0
    for meta in metadata:
        try:
            arxiv_id = meta["arxiv_id"]
            pdf_path = meta["pdf_path"]
            json_path = f"{json_dir}/{arxiv_id}.json"

            try:
                s3_client.head_object(Bucket=bucket_name, Key=str(json_path))
                continue
            except Exception:
                pass

            pdf_obj = s3_client.get_object(Bucket=bucket_name, Key=pdf_path)
            pdf_bytes = pdf_obj["Body"].read()

            full_text = extract_full_text_with_pymupdf(pdf_bytes)

            doc_json = {
                "arxiv_id": arxiv_id,
                "title": meta["title"],
                "abstract": meta["abstract"],
                "authors": meta["authors"],
                "pdf_path": pdf_path,
                "full_text": full_text.strip(),
            }

            json_str = json.dumps(doc_json, ensure_ascii=False, indent=2)
            s3_client.put_object(
                Bucket=bucket_name, Key=json_path, Body=json_str, ContentType="application/json"
            )
            parsed_count += 1

        except Exception as e:
            logger.error(f"Failed to parse {arxiv_id}: {e}")
            continue

        logger.info(f"Saved {parsed_count} JSON files to s3://{bucket_name}/{json_dir}")

    return parsed_count
