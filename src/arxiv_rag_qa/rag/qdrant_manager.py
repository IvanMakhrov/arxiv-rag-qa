import json
import time
from collections.abc import Iterator
from typing import Any

import boto3
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import PointStruct

from utils.setup_logger import setup_logger

# Logging setup
logger = setup_logger(__name__)


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

            buffer_lines = []
            chunk_size = 1024 * 1024 * 10

            for chunk in iter(lambda: body.read(chunk_size), b""):
                buffer_lines.append(chunk.decode("utf-8"))

            full_text = "".join(buffer_lines)

            for line in full_text.split("\n"):
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)

        except Exception as e:
            logger.error(
                f"Embedding file not found in s3://{self.bucket_name}/{self.embedding_dir}: {e}"
            )
            raise FileNotFoundError(
                f"Embedding file not found in s3://{self.bucket_name}/{self.embedding_dir}: {e}"
            ) from e

    def create_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists.")
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
        logger.info(f"Collection '{self.collection_name}' created successfully!")

    def add_data(self) -> None:
        batch = []
        point_id = 0
        time_checkpoint = 30
        last_log_time = time.time()

        for i, record in enumerate(self._read_jsonl_lines()):
            if i % 1000 == 0 or (time.time() - last_log_time) > time_checkpoint:
                elapsed = time.time() - last_log_time
                rate = 1000 / elapsed if elapsed > 0 else 0
                logger.info(f"Read {i} lines | Rate: {rate:.1f} lines/sec")
                last_log_time = time.time()

            point = PointStruct(
                id=point_id,
                vector=record["embedding"],
                payload={"text": record["text"], **record["metadata"]},
            )
            batch.append(point)
            point_id += 1

            if len(batch) >= self.batch_size:
                self.client.upsert(collection_name=self.collection_name, points=batch)
                logger.info(f"Upserted batch of {len(batch)} points (up to ID {point_id - 1})")
                batch = []

        if batch:
            self.client.upsert(collection_name=self.collection_name, points=batch)
            logger.info(f"Upserted final batch of {len(batch)} points")

        logger.info(f"Total {point_id} points inserted into '{self.collection_name}'.")

        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=rest.OptimizersConfigDiff(indexing_threshold=20000),
        )
        logger.info("Indexing re-enabled. Vectors will be searchable shortly.")

        return point_id

    def setup(self) -> None:
        """One-time setup: create collection + ingest data."""
        self.create_collection()
        return self.add_data()
