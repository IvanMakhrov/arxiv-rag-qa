import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

JSON_DIR = Path("data/raw/json")
OUTPUT_CHUNKS_PATH = Path("data/processed/basic_chunks.jsonl")
OUTPUT_EMBEDDINGS_PATH = Path("data/processed/basic_chunks_embeddings.jsonl")


def generate_embeddings():
    """Генерирует эмбеддинги для каждого чанка и сохраняет в новый JSONL"""
    OUTPUT_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    texts = []
    records = []
    with OUTPUT_CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                records.append(record)
                texts.append(record["text"])

    if not texts:
        raise ValueError("No chunks found for embedding!")

    print(f"Generating embeddings for {len(texts)} chunks using {model_name}")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    with OUTPUT_EMBEDDINGS_PATH.open("w", encoding="utf-8") as f:
        for record, emb in zip(records, embeddings, strict=False):
            record["embedding"] = emb.tolist()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved embeddings to {OUTPUT_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    generate_embeddings()
