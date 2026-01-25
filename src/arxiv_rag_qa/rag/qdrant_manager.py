import json
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import PointStruct


class QdrantManager:
    _instance: Optional["QdrantManager"] = None
    _initialized: bool = False

    def __new__(
        cls,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "arxiv_rag",
        vector_size: int = 384,
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._host = host
            cls._instance._port = port
            cls._instance._collection_name = collection_name
            cls._instance._vector_size = vector_size
            cls._initialized = True
        else:
            pass
        return cls._instance

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "arxiv_rag",
        vector_size: int = 384,
    ):
        if self._initialized:
            return
        self._client: QdrantClient | None = None
        self._initialized = True

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=self._host, port=self._port)
        return self._client

    def read_file(self, file_name: str) -> list[dict[str, Any]]:
        records = []
        file_path = Path(file_name)
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    records.append(record)
        return records

    def create_collection(self) -> None:
        if self.client.collection_exists(self._collection_name):
            print(f"Collection '{self._collection_name}' already exists.")
            return

        self.client.create_collection(
            collection_name=self._collection_name,
            vectors_config=rest.VectorParams(
                size=self._vector_size,
                distance=rest.Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=rest.HnswConfigDiff(m=16, ef_construct=100),
        )
        print(f"Collection '{self._collection_name}' created successfully!")

    def add_data(self) -> None:
        file_path = (
            Path(__file__).parent.parent / "data" / "processed" / "basic_chunks_embeddings.jsonl"
        )
        chunks = self.read_file(str(file_path))

        points = [
            PointStruct(
                id=i,
                vector=chunk["embedding"],
                payload={"text": chunk["text"], **chunk["metadata"]},
            )
            for i, chunk in enumerate(chunks)
        ]

        self.client.upsert(collection_name=self._collection_name, points=points)
        print(f"Inserted {len(points)} points into '{self._collection_name}'.")

    def setup(self) -> None:
        """One-time setup: create collection + ingest data."""
        self.create_collection()
        self.add_data()
