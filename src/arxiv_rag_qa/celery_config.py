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
    # Task tracking
    task_track_started=True,
    task_time_limit=3600 * 24,  # 24 hours max per task
    task_soft_time_limit=3600 * 23,
    # Worker optimization for long tasks
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,
    # Monitoring: Enable task events for celery-exporter
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Optional: Prevent event loss
    event_queue_ttl=0,
    event_queue_expires=0,
)

app = celery_app

if __name__ == "__main__":
    celery_app.start()
