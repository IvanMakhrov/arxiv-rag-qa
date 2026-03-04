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
    "email_on_retry": False,
    "retries": cfg.infrastructure.dag.retries,
    "retry_delay": timedelta(minutes=cfg.infrastructure.dag.retry_delay),
}

with DAG(
    "data_downloader",
    default_args=default_args,
    description="Async data pipeline with Celery",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["arxiv", "rag", "async"],
) as dag:
    # ============ DOWNLOAD ============
    trigger_download = PythonOperator(
        task_id="trigger_download",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/download-papers",
            "payload": {
                "category": cfg.data.arxiv_category,
                "start_date": str(cfg.data.start_date),
                "target_count": cfg.data.target_paper_count,
                "results_per_request": cfg.data.results_per_request,
                "bucket_name": cfg.infrastructure.minio.bucket_name,
                "pdf_dir": cfg.data.pdf_dir,
                "metadata_dir": cfg.data.metadata_dir,
            },
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
        },
    )

    wait_download = PythonOperator(
        task_id="wait_download",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_download', key='task_id') }}",
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
            "max_wait_time": cfg.infrastructure.dag.download_timeout,
            "poll_interval": 30,
        },
    )

    log_download = PythonOperator(
        task_id="log_download",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "download",
            "experiment_name": "download_data",
        },
    )

    # ============ PARSE ============
    trigger_parse = PythonOperator(
        task_id="trigger_parse",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/parse-pdf",
            "payload": {
                "bucket_name": cfg.infrastructure.minio.bucket_name,
                "metadata_dir": cfg.data.metadata_dir,
                "json_dir": cfg.data.json_dir,
            },
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
        },
    )

    wait_parse = PythonOperator(
        task_id="wait_parse",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_parse', key='task_id') }}",
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
            "max_wait_time": cfg.infrastructure.dag.parse_timeout,
            "poll_interval": 20,
        },
    )

    log_parse = PythonOperator(
        task_id="log_parse",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "parse",
            "experiment_name": "download_data",
        },
    )

    # ============ DAG DEPENDENCIES ============
    (trigger_download >> wait_download >> log_download >> trigger_parse >> wait_parse >> log_parse)
