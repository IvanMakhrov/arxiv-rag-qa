import json
import re
from pathlib import Path


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
    raw_json_dir: str,
    output_chunks_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> int:
    """
    Process all JSON files into chunks and save as JSONL.
    Returns total number of chunks.
    """
    output_chunks_path = Path(output_chunks_path)
    raw_json_dir = Path(raw_json_dir)
    output_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    json_files = list(raw_json_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {raw_json_dir}")

    all_chunks = []
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                paper = json.load(f)

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
            print(f"Failed to process {json_path.name}: {e}")
            continue

    with output_chunks_path.open("w", encoding="utf-8") as f:
        for doc in all_chunks:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return len(all_chunks)
