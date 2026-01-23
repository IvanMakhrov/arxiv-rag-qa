import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


def generate_embeddings(json_chunks: str, json_embeddings: str, model_name: str):
    """Генерирует эмбеддинги для каждого чанка и сохраняет в новый JSONL"""
    json_chunks = Path(json_chunks)
    json_embeddings = Path(json_embeddings)

    model = SentenceTransformer(model_name)

    texts = []
    records = []
    with json_chunks.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                records.append(record)
                texts.append(record["text"])

    if not texts:
        raise ValueError("No chunks found for embedding!")

    print(f"Generating embeddings for {len(texts)} chunks using {model_name}")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    with json_embeddings.open("w", encoding="utf-8") as f:
        for record, emb in zip(records, embeddings, strict=False):
            record["embedding"] = emb.tolist()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved embeddings to {json_embeddings}")

    return len(embeddings)
