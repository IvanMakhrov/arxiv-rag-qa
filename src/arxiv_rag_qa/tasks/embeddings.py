import json
from datetime import datetime

from arxiv_rag_qa.celery_config import celery_app
from arxiv_rag_qa.rag.generate_embeddings import (
    generate_embeddings_single_model as generate_embeddings,
)
from arxiv_rag_qa.tasks.base import DatabaseTask


@celery_app.task(bind=True, base=DatabaseTask, name="create_embeddings")
def create_embeddings_task(self, request_data):
    """Асинхронная задача для генерации эмбеддингов"""
    task_id = self.request.id

    try:
        self.update_task_status(task_id, status="processing", started_at=datetime.utcnow())

        count = generate_embeddings(
            bucket_name=request_data["bucket_name"],
            chunk_dir=request_data["chunk_dir"],
            embedding_dir=request_data["embedding_dir"],
            model_name=request_data["model_name"],
            batch_size=request_data["batch_size"],
            checkpoint_interval=request_data["checkpoint_interval"],
        )

        result_data = {
            "embeddings_number": count,
            "bucket_name": request_data["bucket_name"],
            "model_name": request_data["model_name"],
            "chunk_dir": request_data["chunk_dir"],
            "embedding_dir": request_data["embedding_dir"],
            "batch_size": request_data["batch_size"],
            "checkpoint_interval": request_data["checkpoint_interval"],
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
