import json
from datetime import datetime

from arxiv_rag_qa.celery_config import celery_app
from arxiv_rag_qa.chunking import full_text_chunking, section_aware_chunking
from arxiv_rag_qa.tasks.base import DatabaseTask


@celery_app.task(bind=True, base=DatabaseTask, name="process_chunks")
def process_chunks_task(self, request_data):
    """Асинхронная задача для чанкирования"""
    task_id = self.request.id

    try:
        self.update_task_status(task_id, status="processing", started_at=datetime.utcnow())

        chunking_type = request_data["chunking_type"]

        if chunking_type == "full_text":
            total_chunks = full_text_chunking.chunking(
                bucket_name=request_data["bucket_name"],
                chunk_dir=request_data["chunk_dir"],
                pdf_dir=request_data["pdf_dir"],
                chunk_size=request_data["chunk_size"],
                chunk_overlap=request_data["chunk_overlap"],
            )
        elif chunking_type == "section_aware":
            total_chunks = section_aware_chunking.chunking(
                bucket_name=request_data["bucket_name"],
                chunk_dir=request_data["chunk_dir"],
                pdf_dir=request_data["pdf_dir"],
                chunk_size=request_data["chunk_size"],
                chunk_overlap=request_data["chunk_overlap"],
            )

        result_data = {
            "total_chunks": total_chunks,
            "bucket_name": request_data["bucket_name"],
            "chunk_dir": request_data["chunk_dir"],
            "pdf_dir": request_data["pdf_dir"],
            "chunk_size": request_data["chunk_size"],
            "chunk_overlap": request_data["chunk_overlap"],
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
