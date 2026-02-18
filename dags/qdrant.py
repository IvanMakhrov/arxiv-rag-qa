from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from hydra import compose, initialize

from utils.dag_celery_manager import log_task_to_mlflow, trigger_task, wait_for_task

with initialize(version_base=None, config_path="../conf", job_name="rag"):
    cfg = compose(config_name="config")

default_args = {
    "owner": "ivan",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": cfg.dag.retries,
    "retry_delay": timedelta(minutes=cfg.dag.retry_delay),
}

with DAG(
    "qdrant_manager",
    default_args=default_args,
    description="Async Qdrant collection setup with Celery",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["arxiv", "rag", "async", "celery", "qdrant"],
) as dag:
    trigger_qdrant = PythonOperator(
        task_id="trigger_qdrant",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/qdrant-setup",
            "payload": {
                "host": cfg.qdrant.host,
                "port": cfg.qdrant.port,
                "collection_name": cfg.qdrant.collection_name,
                "vector_size": cfg.qdrant.vector_size,
                "bucket_name": cfg.minio.bucket_name,
                "embedding_dir": cfg.embeddings.embedding_dir,
                "timeout": cfg.qdrant.timeout,
                "batch_size": cfg.qdrant.batch_size,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_qdrant = PythonOperator(
        task_id="wait_qdrant",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_qdrant', key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.qdrant_timeout,
            "poll_interval": 30,
        },
    )

    log_qdrant = PythonOperator(
        task_id="log_qdrant",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "setup_qdrant",
            "experiment_name": cfg.mlflow.experiment_name,
        },
    )

    # ============ DAG DEPENDENCIES ============
    trigger_qdrant >> wait_qdrant >> log_qdrant
