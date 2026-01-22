import json
from pathlib import Path

import fitz


def extract_full_text_with_pymupdf(pdf_path: str) -> str:
    """Извлекает весь текст из PDF, сохраняя порядок чтения."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text")
    doc.close()
    return full_text


def load_metadata(metadata_path: str):
    with Path.open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Metadata loaded from {metadata_path}")
    return metadata


def parse_pdfs_to_json(
    raw_pdf_dir: str,
    metadata_path: str,
    processed_json_dir: str,
) -> int:
    """
    Convert all PDFs in raw_pdf_dir to JSON files with full text.
    Returns number of parsed papers.
    """
    processed_json_dir = Path(processed_json_dir)
    raw_pdf_dir = Path(raw_pdf_dir)
    processed_json_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(metadata_path)
    pdf_files = list(raw_pdf_dir.glob("*.pdf"))

    if not metadata:
        raise FileNotFoundError(f"No metadata found in {metadata_path}")

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {raw_pdf_dir}")

    parsed_count = 0
    for meta in metadata:
        try:
            arxiv_id = meta["arxiv_id"]
            pdf_path = meta["pdf_path"]
            json_path = processed_json_dir / f"{arxiv_id}.json"

            if json_path.exists():
                continue

            full_text = extract_full_text_with_pymupdf(str(pdf_path))

            doc_json = {
                "arxiv_id": arxiv_id,
                "title": meta["title"],
                "abstract": meta["abstract"],
                "authors": meta["authors"],
                "pdf_path": pdf_path,
                "full_text": full_text.strip(),
            }

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(doc_json, f, ensure_ascii=False, indent=2)

            parsed_count += 1

        except Exception as e:
            print(f"Failed to parse {pdf_path.name}: {e}")
            continue

        print(f"Saved {parsed_count} JSON files to {processed_json_dir}")

    return parsed_count
