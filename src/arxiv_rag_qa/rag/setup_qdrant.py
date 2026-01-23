import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import PointStruct

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_rag"
VECTOR_SIZE = 384


def read_file(file_name):
    records = []

    file_name = Path(file_name)
    with file_name.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                records.append(record)

    return records


def create_collection():
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(
            size=VECTOR_SIZE, distance=rest.Distance.COSINE, on_disk=True
        ),
        hnsw_config=rest.HnswConfigDiff(m=16, ef_construct=100),
    )

    print(f"Collection '{COLLECTION_NAME}' created successfully!")


def add_data():
    file_path = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "processed"
        / "basic_chunks_embeddings.jsonl"
    )
    chunks = read_file(str(file_path))

    points = [
        PointStruct(
            id=i, vector=chunk["embedding"], payload={"text": chunk["text"], **chunk["metadata"]}
        )
        for i, chunk in enumerate(chunks)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)


if __name__ == "__main__":
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    create_collection()
    add_data()
