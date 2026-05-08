import json
from datetime import datetime

from arxiv_rag_qa.celery_config import celery_app
from arxiv_rag_qa.rag.qdrant_manager import QdrantManager
from arxiv_rag_qa.tasks.base import DatabaseTask


@celery_app.task(bind=True, base=DatabaseTask, name="setup_qdrant")
def setup_qdrant_task(self, request_data):
    """Асинхронная задача для настройки Qdrant"""
    task_id = self.request.id

    try:
        self.update_task_status(task_id, status="processing", started_at=datetime.utcnow())

        retriever_type = request_data.get("retriever_type", "dense")

        qdrant = QdrantManager(
            host=request_data["host"],
            port=request_data["port"],
            collection_name=request_data["collection_name"],
            vector_size=request_data["vector_size"],
            bucket_name=request_data["bucket_name"],
            embedding_dir=request_data["embedding_dir"],
            chunk_dir=request_data.get("chunk_dir", ""),
            timeout=request_data["timeout"],
            batch_size=request_data["batch_size"],
        )

        count = qdrant.setup(retriever_type=retriever_type)

        result_data = {
            "points_number": count,
            "collection_name": request_data["collection_name"],
            "vector_size": request_data["vector_size"],
            "bucket_name": request_data["bucket_name"],
            "embedding_dir": request_data["embedding_dir"],
            "chunk_dir": request_data.get("chunk_dir", ""),
            "retriever_type": retriever_type,
            "batch_size": request_data["batch_size"],
        }

        self.update_task_status(
            task_id,
            status="completed",
            completed_at=datetime.utcnow(),
            result_data=json.dumps(result_data),
            progress=100,
        )

        return result_data

    except Exception as e:
        self.update_task_status(
            task_id, status="failed", error_message=str(e), completed_at=datetime.utcnow()
        )
        raise
