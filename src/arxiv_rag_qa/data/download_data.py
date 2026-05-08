import json
import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote

import boto3
import requests

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


def get_minio_client():
    return boto3.client("s3")


def save_metadata_to_minio(data: list, bucket_name: str, metadata_key: str, s3_client: Any):
    """Save metadata JSON to MinIO as a single object."""
    try:
        json_str = json.dumps(data, indent=2)
        s3_client.put_object(
            Bucket=bucket_name, Key=metadata_key, Body=json_str, ContentType="application/json"
        )
        logger.info(f"Metadata saved to s3://{bucket_name}/{metadata_key}")
    except Exception as e:
        logger.error(f"Error saving metadata to s3://{bucket_name}/{metadata_key}: {e}")


def create_bucket(bucket_name: str, s3_client: Any):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} already exists")
    except Exception:
        s3_client.create_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} created")


def fetch_arxiv_pdfs(
    category: str = "",
    start_date: str = "",
    target_count: int = 0,
    results_per_request: int = 0,
    bucket_name: str = "",
    pdf_dir: str = "",
    metadata_dir: str = "",
) -> list[dict]:
    """
    Download arXiv papers as PDFs and return metadata list.
    Does NOT parse to JSON.
    """

    s3_client = get_minio_client()
    create_bucket(bucket_name, s3_client)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15"
    }

    downloaded = {}
    start_index = 0

    logger.info(f"Target: {target_count} papers from '{category}' since {start_date[:8]}")

    while len(downloaded) < target_count:
        remaining = target_count - len(downloaded)
        batch_size = min(results_per_request, remaining)

        search_query = f"cat:{category} AND submittedDate:[{start_date} TO 999912312359]"
        encoded_query = quote(search_query)

        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={encoded_query}&"
            f"sortBy=submittedDate&"
            f"sortOrder=descending&"
            f"start={start_index}&"
            f"max_results={batch_size}"
        )

        logger.info(
            f"Requesting papers {start_index}-{start_index + batch_size - 1} "
            f"(total so far: {len(downloaded)})"
        )

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"API request failed: {e}. Retrying in 5s")
            time.sleep(5)
            continue

        root = ET.fromstring(response.content)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")

        if not entries:
            logger.info("No more papers available")
            break

        for entry in entries:
            try:
                paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
                title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
                summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
                authors = [
                    author.find("{http://www.w3.org/2005/Atom}name").text
                    for author in entry.findall("{http://www.w3.org/2005/Atom}author")
                ]
                pdf_path = f"{pdf_dir}/{paper_id}.pdf"

                try:
                    s3_client.head_object(Bucket=bucket_name, Key=pdf_path)
                    logger.warning(f"Already exists in MinIO: {paper_id}")
                    exists_in_minio = True
                except Exception:
                    exists_in_minio = False

                if paper_id not in downloaded and not exists_in_minio:
                    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                    pdf_resp = requests.get(pdf_url, headers=headers, timeout=10)
                    pdf_resp.raise_for_status()

                    s3_client.put_object(
                        Bucket=bucket_name,
                        Key=pdf_path,
                        Body=pdf_resp.content,
                        ContentType="application/pdf",
                    )
                    logger.info(f"Uploaded to MinIO: {paper_id}")

                    downloaded[paper_id] = {
                        "arxiv_id": paper_id,
                        "title": title,
                        "abstract": summary,
                        "authors": authors,
                        "pdf_path": pdf_path,
                    }
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing {paper_id}: {e}")
                continue

        start_index += batch_size
        if len(entries) < batch_size:
            break

        metadata_list = [downloaded[pid] for pid in downloaded]
        save_metadata_to_minio(metadata_list, bucket_name, metadata_dir, s3_client)

    return len(list(downloaded.values()))
