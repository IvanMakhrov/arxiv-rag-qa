import json
from datetime import datetime

from arxiv_rag_qa.celery_config import celery_app
from arxiv_rag_qa.data.download_data import fetch_arxiv_pdfs
from arxiv_rag_qa.tasks.base import DatabaseTask


@celery_app.task(bind=True, base=DatabaseTask, name="download_papers")
def download_papers_task(self, request_data):
    """Асинхронная задача для скачивания статей"""
    task_id = self.request.id

    try:
        # Обновление статуса: начало выполнения
        self.update_task_status(task_id, status="processing", started_at=datetime.utcnow())

        # Выполнение задачи
        count = fetch_arxiv_pdfs(
            category=request_data["category"],
            start_date=request_data["start_date"],
            target_count=request_data["target_count"],
            results_per_request=request_data["results_per_request"],
            bucket_name=request_data["bucket_name"],
            pdf_dir=request_data["pdf_dir"],
            metadata_dir=request_data["metadata_dir"],
        )

        # Сохранение результата
        result_data = {
            "downloaded_papers_number": count,
            "category": request_data["category"],
            "start_date": request_data["start_date"],
        }

        # Обновление статуса: завершение
        self.update_task_status(
            task_id,
            status="completed",
            completed_at=datetime.utcnow(),
            result_data=json.dumps(result_data),
            progress=100,
        )

        return result_data

    except Exception as e:
        # Обновление статуса: ошибка
        self.update_task_status(
            task_id, status="failed", error_message=str(e), completed_at=datetime.utcnow()
        )
        raise
