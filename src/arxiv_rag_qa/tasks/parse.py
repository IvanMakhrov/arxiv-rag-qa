import json
from datetime import datetime

from arxiv_rag_qa.celery_config import celery_app
from arxiv_rag_qa.data.parse_pdf_to_json import parse_pdfs_to_json
from arxiv_rag_qa.tasks.base import DatabaseTask


@celery_app.task(bind=True, base=DatabaseTask, name="parse_pdfs")
def parse_pdfs_task(self, request_data):
    """Асинхронная задача для парсинга PDF"""
    task_id = self.request.id

    try:
        # Обновление статуса: начало выполнения
        self.update_task_status(task_id, status="processing", started_at=datetime.utcnow())

        # Выполнение задачи
        count = parse_pdfs_to_json(
            bucket_name=request_data["bucket_name"],
            metadata_dir=request_data["metadata_dir"],
            json_dir=request_data["json_dir"],
        )

        # Подготовка результата
        result_data = {
            "parsed_papers_number": count,
            "bucket_name": request_data["bucket_name"],
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
