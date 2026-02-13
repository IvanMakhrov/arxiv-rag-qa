import json
from collections.abc import Iterator
from typing import Any

import boto3
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
        bucket_name: str = "",
        embedding_dir: str = "",
        timeout: int = 5,
        batch_size: int = 256,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.bucket_name = bucket_name
        self.embedding_dir = embedding_dir
        self.timeout = timeout
        self.batch_size = batch_size

        self._client = None
        self._s3_client = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port, timeout=self.timeout)
        return self._client

    @property
    def s3_client(self):
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    def _read_jsonl_lines(self) -> Iterator[dict[str, Any]]:
        """Efficiently stream JSONL from MinIO using buffered reads."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.embedding_dir)
            body = response["Body"]

            buffer = ""
            chunk_size = 1024 * 1024 * 100

            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break

                buffer += chunk.decode("utf-8")

                lines = buffer.split("\n")
                buffer = lines[-1]

                for line in lines[:-1]:
                    stripped = line.strip()
                    if stripped:
                        yield json.loads(stripped)

            if buffer.strip():
                yield json.loads(buffer.strip())

        except Exception as e:
            raise FileNotFoundError(
                f"Embedding file not found in s3://{self.bucket_name}/{self.embedding_dir}: {e}"
            ) from e

    def create_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists.")
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=rest.OptimizersConfigDiff(indexing_threshold=0),
            )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(
                size=self.vector_size,
                distance=rest.Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=rest.HnswConfigDiff(m=16, ef_construct=100),
            optimizers_config=rest.OptimizersConfigDiff(indexing_threshold=0),
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

        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=rest.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print("Indexing re-enabled. Vectors will be searchable shortly.")

        return point_id

    def setup(self) -> None:
        """One-time setup: create collection + ingest data."""
        self.create_collection()
        return self.add_data()
