import os

from celery import Celery

broker_url = os.getenv("CELERY_BROKER_URL")
result_backend = os.getenv("CELERY_RESULT_BACKEND")

celery_app = Celery(
    "arxiv_rag_qa",
    broker=broker_url,
    backend=result_backend,
    include=[
        "arxiv_rag_qa.tasks.download",
        "arxiv_rag_qa.tasks.parse",
        "arxiv_rag_qa.tasks.chunk",
        "arxiv_rag_qa.tasks.embeddings",
        "arxiv_rag_qa.tasks.test_data",
        "arxiv_rag_qa.tasks.qdrant",
        "arxiv_rag_qa.tasks.eval",
        "arxiv_rag_qa.tasks.rag",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 24,  # 24 часа максимум на задачу
    task_soft_time_limit=3600 * 23,
    worker_prefetch_multiplier=1,  # Важно для долгих задач
    worker_max_tasks_per_child=10,  # Перезапуск воркера после 10 задач
)

app = celery_app

if __name__ == "__main__":
    celery_app.start()
