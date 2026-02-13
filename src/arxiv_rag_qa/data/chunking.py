import json
import re

import boto3


def get_minio_client():
    return boto3.client("s3")


def process_text(text):
    text = text.replace("-\n", "")
    text = re.sub(r"\n[1-9]\s", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_recursive(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
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
                    chunks.append(current_chunk.rstrip())
                    current_chunk = part + (sep if sep != " " else "")

            if current_chunk:
                chunks.append(current_chunk.rstrip())

            final_chunks = []
            for chunk in chunks:
                if len(chunk) > chunk_size:
                    final_chunks.extend(split_text_recursive(chunk, chunk_size, chunk_overlap))
                else:
                    final_chunks.append(chunk)
            return final_chunks

    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]


def process_all_papers_to_chunks(
    bucket_name: str,
    json_dir: str,
    chunk_dir: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> int:
    """
    Process all JSON files into chunks and save as JSONL.
    Returns total number of chunks.
    """
    s3_client = get_minio_client()

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=json_dir.rstrip("/") + "/")

    json_files = []
    for page in pages:
        if "Contents" in page:
            json_files.extend(
                [obj["Key"] for obj in page["Contents"] if obj["Key"].endswith(".json")]
            )

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in s3://{bucket_name}/{json_dir}")

    all_chunks = []
    for json_path in json_files:
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=json_path)
            paper = json.loads(response["Body"].read().decode("utf-8"))

            full_text = paper.get("full_text", "").strip()
            if not full_text:
                continue

            cleaned_text = process_text(full_text)
            chunks = split_text_recursive(
                cleaned_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    all_chunks.append(
                        {
                            "id": f"{paper['arxiv_id']}_chunk_{i}",
                            "text": chunk,
                            "metadata": {
                                "arxiv_id": paper["arxiv_id"],
                                "title": paper["title"],
                                "source": str(paper.get("pdf_path", "")),
                                "chunk_index": i,
                            },
                        }
                    )

        except Exception as e:
            print(f"Failed to process {json_path}: {e}")
            continue

    jsonl_content = "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in all_chunks)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=chunk_dir,
        Body=jsonl_content.encode("utf-8"),
        ContentType="application/jsonl",
    )

    print(f"Saved {len(all_chunks)} chunks to s3://{bucket_name}/{chunk_dir}")

    return len(all_chunks)
