import json
from datetime import datetime

from arxiv_rag_qa.celery_config import celery_app
from arxiv_rag_qa.rag.rag import get_response
from arxiv_rag_qa.tasks.base import DatabaseTask


@celery_app.task(bind=True, base=DatabaseTask, name="get_rag_response")
def get_rag_response_task(self, request_data):
    """Асинхронная задача для получения RAG ответа"""
    task_id = self.request.id

    try:
        self.update_task_status(task_id, status="processing", started_at=datetime.utcnow())

        response = get_response(
            emb_model_name=request_data["emb_model_name"],
            collection_name=request_data["collection_name"],
            top_k=request_data["top_k"],
            gen_model_name=request_data["gen_model_name"],
            query=request_data["query"],
            qdrant_host=request_data["qdrant_host"],
            qdrant_port=request_data["qdrant_port"],
        )

        result_data = {
            "answer": response,
            "sources": [],
            "emb_model_name": request_data["emb_model_name"],
            "collection_name": request_data["collection_name"],
            "top_k": request_data["top_k"],
            "gen_model_name": request_data["gen_model_name"],
            "query": request_data["query"],
        }

        self.update_task_status(
            task_id,
            status="completed",
            completed_at=datetime.utcnow(),
            result=json.dumps(result_data),
            progress=100,
        )

        return result_data

    except Exception as e:
        self.update_task_status(
            task_id, status="failed", error_message=str(e), completed_at=datetime.utcnow()
        )
        raise
