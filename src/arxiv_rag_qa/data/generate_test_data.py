import json
from pathlib import Path
from typing import Any


def load_chunks(file_path: Path) -> list[dict[str, Any]]:
    """Load chunked documents with metadata."""
    chunks = []
    with Path.open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_metadata(metadata_path: Path) -> dict[str, dict[str, Any]]:
    """Load metadata.json and return {arxiv_id: paper_data} mapping."""
    with Path.open(metadata_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    return {paper["arxiv_id"]: paper for paper in papers}


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
    chunks_path: str, test_data_path: str, metadata_path: str, test_data_size: int
):
    chunks = load_chunks(chunks_path)

    if not chunks:
        raise ValueError("No chunks found. Run ingestion first.")

    metadata_map = load_metadata(metadata_path)
    paper_to_chunks = build_paper_to_chunks(chunks)
    test_samples = generate_test_samples(
        chunks=chunks,
        paper_to_chunks=paper_to_chunks,
        metadata_map=metadata_map,
        test_data_size=test_data_size,
    )

    with Path.open(test_data_path, "w", encoding="utf-8") as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
