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

        emb_model_name = request_data.get("emb_model_name", "all-MiniLM-L6-v2")
        embedding_model = request_data.get("embedding_model", "all-MiniLM-L6-v2")
        retriever_type = request_data.get("retriever_type", "dense")
        sparse_method = request_data.get("sparse_method", "bm25")
        use_qdrant_corpus = request_data.get("use_qdrant_corpus", True)
        hybrid_config = request_data.get("hybrid_config", {})
        sparse_params = request_data.get("sparse_params", {})
        in_memory = request_data.get("in_memory", False)

        response = get_response(
            emb_model_name=emb_model_name,
            collection_name=request_data["collection_name"],
            top_k=request_data["top_k"],
            gen_model_name=request_data["gen_model_name"],
            query=request_data["query"],
            qdrant_host=request_data["qdrant_host"],
            qdrant_port=request_data["qdrant_port"],
            retriever_type=retriever_type,
            sparse_method=sparse_method,
            use_qdrant_corpus=use_qdrant_corpus,
            hybrid_config=hybrid_config,
            sparse_params=sparse_params,
            in_memory=in_memory,
            embedding_model=embedding_model,
        )

        timing = response.get("timing", {})
        token_usage = response.get("token_usage", {})

        result_data = {
            "answer": response.get("answer", ""),
            "sources": response.get("sources", []),
            "emb_model_name": emb_model_name,
            "collection_name": request_data["collection_name"],
            "top_k": request_data["top_k"],
            "gen_model_name": request_data["gen_model_name"],
            "query": request_data["query"],
            "retriever_type": retriever_type,
            "sparse_method": sparse_method,
            "use_qdrant_corpus": use_qdrant_corpus,
            "in_memory": in_memory,
            "hybrid_config": hybrid_config,
            "sparse_params": sparse_params,
            "embedding_model": embedding_model,
            "timing": {
                "retrieve_time_s": timing.get("retrieve_time_s"),
                "generate_time_s": timing.get("generate_time_s"),
                "total_time_s": timing.get("total_time_s"),
            },
            "token_usage": {
                "generated_tokens": token_usage.get("generated_tokens"),
                "tokens_per_second": token_usage.get("tokens_per_second"),
            },
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
