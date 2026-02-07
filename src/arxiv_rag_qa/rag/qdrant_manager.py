import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import PointStruct


class QdrantManager:
    def __init__(
        self,
        host: str = "",
        port: int = 0,
        collection_name: str = "",
        vector_size: int = 0,
        embedding_dir: str = "",
        timeout: int = 5,
        batch_size: int = 256,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding_dir = embedding_dir
        self.timeout = timeout
        self.batch_size = batch_size

        self._client = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port, timeout=self.timeout)
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

    def _read_jsonl_lines(self) -> Iterator[dict[str, Any]]:
        """Generator that yields one record at a time from the JSONL file."""
        embedding_dir = Path(self.embedding_dir)

        with embedding_dir.open("r", encoding="utf-8") as f:
            for chunk in f:
                line = chunk.strip()
                if line:
                    yield json.loads(line)

    def create_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists.")
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(
                size=self.vector_size,
                distance=rest.Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=rest.HnswConfigDiff(m=16, ef_construct=100),
        )
        print(f"Collection '{self.collection_name}' created successfully!")

    def add_data(self) -> None:
        batch = []
        point_id = 0

        for record in self._read_jsonl_lines():
            point = PointStruct(
                id=point_id,
                vector=record["embedding"],
                payload={"text": record["text"], **record["metadata"]},
            )
            batch.append(point)
            point_id += 1

            if len(batch) >= self.batch_size:
                self.client.upsert(collection_name=self.collection_name, points=batch)
                print(f"Upserted batch of {len(batch)} points (up to ID {point_id - 1})")
                batch = []

        if batch:
            self.client.upsert(collection_name=self.collection_name, points=batch)
            print(f"Upserted final batch of {len(batch)} points")

        print(f"Total {point_id} points inserted into '{self.collection_name}'.")

    def setup(self) -> None:
        """One-time setup: create collection + ingest data."""
        self.create_collection()
        self.add_data()
