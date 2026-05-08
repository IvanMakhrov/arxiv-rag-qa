import json
from datetime import datetime

from arxiv_rag_qa.celery_config import celery_app
from arxiv_rag_qa.eval.eval_generator import generator_eval
from arxiv_rag_qa.eval.eval_retriever import retriever_eval
from arxiv_rag_qa.tasks.base import DatabaseTask


@celery_app.task(bind=True, base=DatabaseTask, name="evaluate_retriever")
def evaluate_retriever_task(self, request_data):
    """Асинхронная задача для оценки ретривера"""
    task_id = self.request.id

    try:
        self.update_task_status(task_id, status="processing", started_at=datetime.utcnow())

        results = retriever_eval(
            bucket_name=request_data["bucket_name"],
            test_data_dir=request_data["test_data_dir"],
            collection_name=request_data["collection_name"],
            top_k=request_data["top_k"],
            model_name=request_data["model_name"],
            qdrant_host=request_data["qdrant_host"],
            qdrant_port=request_data["qdrant_port"],
            retriever_type=request_data.get("retriever_type", "dense"),
            sparse_method=request_data.get("sparse_method", "bm25"),
            use_qdrant_corpus=request_data.get("use_qdrant_corpus", True),
            hybrid_config=request_data.get("hybrid_config"),
            sparse_params=request_data.get("sparse_params"),
        )

        result_data = {
            "results": results,
            "bucket_name": request_data["bucket_name"],
            "test_data_dir": request_data["test_data_dir"],
            "collection_name": request_data["collection_name"],
            "top_k": request_data["top_k"],
            "model_name": request_data["model_name"],
            "retriever_type": request_data.get("retriever_type", "dense"),
            "sparse_method": request_data.get("sparse_method", "bm25"),
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


@celery_app.task(bind=True, base=DatabaseTask, name="evaluate_generator")
def evaluate_generator_task(self, request_data):
    """Асинхронная задача для оценки генератора"""
    task_id = self.request.id

    try:
        self.update_task_status(task_id, status="processing", started_at=datetime.utcnow())

        results = generator_eval(
            bucket_name=request_data["bucket_name"],
            test_data_dir=request_data["test_data_dir"],
            collection_name=request_data["collection_name"],
            top_k=request_data["top_k"],
            emb_model_name=request_data["emb_model_name"],
            gen_model_name=request_data["gen_model_name"],
            bertscore_model=request_data["bertscore_model"],
            qdrant_host=request_data["qdrant_host"],
            qdrant_port=request_data["qdrant_port"],
            llm_judge_model=request_data.get("llm_judge_model"),
            retriever_type=request_data.get("retriever_type", "dense"),
            sparse_method=request_data.get("sparse_method", "bm25"),
            use_qdrant_corpus=request_data.get("use_qdrant_corpus", True),
            hybrid_config=request_data.get("hybrid_config"),
            sparse_params=request_data.get("sparse_params"),
        )

        result_data = {
            "results": results,
            "bucket_name": request_data["bucket_name"],
            "test_data_dir": request_data["test_data_dir"],
            "collection_name": request_data["collection_name"],
            "top_k": request_data["top_k"],
            "emb_model_name": request_data["emb_model_name"],
            "gen_model_name": request_data["gen_model_name"],
            "bertscore_model": request_data["bertscore_model"],
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
