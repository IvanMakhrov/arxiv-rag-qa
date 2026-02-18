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
    "retries": cfg.dag.retries,
    "retry_delay": timedelta(minutes=cfg.dag.retry_delay),
}

with DAG(
    "prepare_data",
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
                "category": cfg.download.arxiv_category,
                "start_date": str(cfg.download.start_date),
                "target_count": cfg.download.target_paper_count,
                "results_per_request": cfg.download.results_per_request,
                "bucket_name": cfg.minio.bucket_name,
                "pdf_dir": cfg.download.pdf_dir,
                "metadata_dir": cfg.download.metadata_dir,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_download = PythonOperator(
        task_id="wait_download",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_download', key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.download_timeout,
            "poll_interval": 30,
        },
    )

    log_download = PythonOperator(
        task_id="log_download",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "download",
            "experiment_name": cfg.mlflow.experiment_name,
        },
    )

    # ============ PARSE ============
    trigger_parse = PythonOperator(
        task_id="trigger_parse",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/parse-pdf",
            "payload": {
                "bucket_name": cfg.minio.bucket_name,
                "metadata_dir": cfg.download.metadata_dir,
                "json_dir": cfg.download.json_dir,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_parse = PythonOperator(
        task_id="wait_parse",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_parse', key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.parse_timeout,
            "poll_interval": 20,
        },
    )

    log_parse = PythonOperator(
        task_id="log_parse",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "parse",
            "experiment_name": cfg.mlflow.experiment_name,
        },
    )

    # ============ CHUNKING ============
    trigger_chunking = PythonOperator(
        task_id="trigger_chunking",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/chunking",
            "payload": {
                "bucket_name": cfg.minio.bucket_name,
                "chunk_dir": cfg.chunking.chunk_dir,
                "json_dir": cfg.download.json_dir,
                "chunk_size": cfg.chunking.chunk_size,
                "chunk_overlap": cfg.chunking.chunk_overlap,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_chunking = PythonOperator(
        task_id="wait_chunking",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_chunking', key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.chunking_timeout,
            "poll_interval": 25,
        },
    )

    log_chunking = PythonOperator(
        task_id="log_chunking",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "chunking",
            "experiment_name": cfg.mlflow.experiment_name,
        },
    )

    # ============ EMBEDDINGS ============
    trigger_embeddings = PythonOperator(
        task_id="trigger_embeddings",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/embeddings",
            "payload": {
                "bucket_name": cfg.minio.bucket_name,
                "chunk_dir": cfg.chunking.chunk_dir,
                "embedding_dir": cfg.embeddings.embedding_dir,
                "model_name": cfg.embeddings.model_name,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_embeddings = PythonOperator(
        task_id="wait_embeddings",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_embeddings', "
            "key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.embeddings_timeout,
            "poll_interval": 45,
        },
    )

    log_embeddings = PythonOperator(
        task_id="log_embeddings",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "embedding",
            "experiment_name": cfg.mlflow.experiment_name,
        },
    )

    # ============ TEST DATA ============
    trigger_test_data = PythonOperator(
        task_id="trigger_test_data",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/generate-test-data",
            "payload": {
                "bucket_name": cfg.minio.bucket_name,
                "chunk_dir": cfg.chunking.chunk_dir,
                "test_data_dir": cfg.test_data.test_data_dir,
                "metadata_dir": cfg.download.metadata_dir,
                "test_data_size": cfg.test_data.test_data_size,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_test_data = PythonOperator(
        task_id="wait_test_data",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_test_data', key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.test_data_timeout,
            "poll_interval": 15,
        },
    )

    log_test_data = PythonOperator(
        task_id="log_test_data",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "generate_test_data",
            "experiment_name": cfg.mlflow.experiment_name,
        },
    )

    # ============ DAG DEPENDENCIES ============
    (
        trigger_download
        >> wait_download
        >> log_download
        >> trigger_parse
        >> wait_parse
        >> log_parse
        >> trigger_chunking
        >> wait_chunking
        >> log_chunking
        >> trigger_embeddings
        >> wait_embeddings
        >> log_embeddings
        >> trigger_test_data
        >> wait_test_data
        >> log_test_data
    )
