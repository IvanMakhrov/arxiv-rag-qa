import json
from datetime import datetime

from arxiv_rag_qa.celery_config import celery_app
from arxiv_rag_qa.data.generate_test_data import generate_test_data
from arxiv_rag_qa.data.generate_test_data_llm import generate_test_data_llm
from arxiv_rag_qa.tasks.base import DatabaseTask


@celery_app.task(bind=True, base=DatabaseTask, name="generate_test_data")
def generate_test_data_task(self, request_data):
    task_id = self.request.id

    try:
        self.update_task_status(task_id, status="processing", started_at=datetime.utcnow())

        test_type = request_data["test_type"]

        if test_type == "llm":
            count = generate_test_data_llm(
                bucket_name=request_data["bucket_name"],
                chunk_dir=request_data["chunk_dir"],
                test_data_dir=request_data["test_data_dir"],
                test_data_size=request_data["test_data_size"],
                max_questions_per_paper=request_data["max_questions_per_paper"],
            )
            result_data = {
                "test_data_size": request_data["test_data_size"],
                "max_questions_per_paper": request_data["max_questions_per_paper"],
                "bucket_name": request_data["bucket_name"],
                "chunk_dir": request_data["chunk_dir"],
                "test_data_dir": request_data["test_data_dir"],
                "test_type": test_type,
                "generated_samples": count,
            }
        elif test_type == "base":
            generate_test_data(
                bucket_name=request_data["bucket_name"],
                chunk_dir=request_data["chunk_dir"],
                test_data_dir=request_data["test_data_dir"],
                metadata_dir=request_data["metadata_dir"],
                test_data_size=request_data["test_data_size"],
                max_questions_per_paper=request_data["max_questions_per_paper"],
            )
            result_data = {
                "test_data_size": request_data["test_data_size"],
                "max_questions_per_paper": request_data["max_questions_per_paper"],
                "bucket_name": request_data["bucket_name"],
                "chunk_dir": request_data["chunk_dir"],
                "test_data_dir": request_data["test_data_dir"],
                "metadata_dir": request_data["metadata_dir"],
                "test_type": test_type,
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
