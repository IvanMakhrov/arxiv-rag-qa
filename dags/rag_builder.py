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

retriever_type = cfg.experiments.retriever.get("type", "dense")
needs_embeddings = retriever_type in ("dense", "hybrid")

with DAG(
    "build_rag",
    default_args=default_args,
    description="RAG builder",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["arxiv", "rag", "async"],
) as dag:
    # ============ CHUNKING ============
    trigger_chunking = PythonOperator(
        task_id="trigger_chunking",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/chunking",
            "payload": {
                "bucket_name": cfg.infrastructure.minio.bucket_name,
                "chunk_dir": cfg.experiments.chunking.chunk_dir,
                "pdf_dir": cfg.data.pdf_dir,
                "chunk_size": cfg.experiments.chunking.chunk_size,
                "chunk_overlap": cfg.experiments.chunking.chunk_overlap,
                "chunking_type": cfg.experiments.chunking.chunking_type,
            },
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
        },
    )

    wait_chunking = PythonOperator(
        task_id="wait_chunking",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_chunking', key='task_id') }}",
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
            "max_wait_time": cfg.infrastructure.dag.chunking_timeout,
            "poll_interval": 25,
            "max_retries": 3,
            "retry_delay": 60,
        },
    )

    log_chunking = PythonOperator(
        task_id="log_chunking",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "chunking",
            "experiment_name": cfg.experiments.mlflow.experiment_name,
        },
    )

    # ============ EMBEDDINGS ============
    if needs_embeddings:
        trigger_embeddings = PythonOperator(
            task_id="trigger_embeddings",
            python_callable=trigger_task,
            op_kwargs={
                "endpoint": "/embeddings",
                "payload": {
                    "bucket_name": cfg.infrastructure.minio.bucket_name,
                    "chunk_dir": cfg.experiments.chunking.chunk_dir,
                    "embedding_dir": cfg.experiments.embeddings.embedding_dir,
                    "model_name": cfg.experiments.embeddings.default_model,
                    "batch_size": cfg.experiments.embeddings.batch_size,
                    "checkpoint_interval": cfg.experiments.embeddings.checkpoint_interval,
                },
                "http_conn_id": cfg.infrastructure.dag.http_conn_id,
            },
        )

        wait_embeddings = PythonOperator(
            task_id="wait_embeddings",
            python_callable=wait_for_task,
            op_kwargs={
                "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_embeddings', "
                "key='task_id') }}",
                "http_conn_id": cfg.infrastructure.dag.http_conn_id,
                "max_wait_time": cfg.infrastructure.dag.embeddings_timeout,
                "poll_interval": 45,
                "max_retries": 5,
                "retry_delay": 120,
            },
        )

        log_embeddings = PythonOperator(
            task_id="log_embeddings",
            python_callable=log_task_to_mlflow,
            op_kwargs={
                "task_stage": "embedding",
                "experiment_name": cfg.experiments.mlflow.experiment_name,
            },
        )

    # ============ TEST DATA ============
    trigger_test_data = PythonOperator(
        task_id="trigger_test_data",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/generate-test-data",
            "payload": {
                "bucket_name": cfg.infrastructure.minio.bucket_name,
                "chunk_dir": cfg.experiments.chunking.chunk_dir,
                "test_data_dir": cfg.experiments.test_data.test_data_dir,
                "metadata_dir": cfg.data.metadata_dir,
                "test_data_size": cfg.experiments.test_data.test_data_size,
                "max_questions_per_paper": cfg.experiments.test_data.max_questions_per_paper,
                "test_type": cfg.experiments.test_data.test_type,
            },
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
        },
    )

    wait_test_data = PythonOperator(
        task_id="wait_test_data",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_test_data', key='task_id') }}",
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
            "max_wait_time": cfg.infrastructure.dag.test_data_timeout,
            "poll_interval": 15,
            "max_retries": 3,
            "retry_delay": 60,
        },
    )

    log_test_data = PythonOperator(
        task_id="log_test_data",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "generate_test_data",
            "experiment_name": cfg.experiments.mlflow.experiment_name,
        },
    )

    # ============ QDRANT ============
    trigger_qdrant = PythonOperator(
        task_id="trigger_qdrant",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/qdrant-setup",
            "payload": {
                "host": cfg.experiments.qdrant.host,
                "port": cfg.experiments.qdrant.port,
                "collection_name": cfg.experiments.qdrant.collection_name,
                "vector_size": cfg.experiments.qdrant.vector_size,
                "bucket_name": cfg.infrastructure.minio.bucket_name,
                "embedding_dir": cfg.experiments.embeddings.embedding_dir,
                "chunk_dir": cfg.experiments.chunking.chunk_dir,
                "retriever_type": retriever_type,
                "timeout": cfg.experiments.qdrant.timeout,
                "batch_size": cfg.experiments.qdrant.batch_size,
            },
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
        },
    )

    wait_qdrant = PythonOperator(
        task_id="wait_qdrant",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_qdrant', key='task_id') }}",
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
            "max_wait_time": cfg.infrastructure.dag.qdrant_timeout,
            "poll_interval": 30,
            "max_retries": 3,
            "retry_delay": 60,
        },
    )

    log_qdrant = PythonOperator(
        task_id="log_qdrant",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "setup_qdrant",
            "experiment_name": cfg.experiments.mlflow.experiment_name,
        },
    )

    # ============ DAG DEPENDENCIES ============
    if needs_embeddings:
        (
            trigger_chunking
            >> wait_chunking
            >> log_chunking
            >> trigger_embeddings
            >> wait_embeddings
            >> log_embeddings
            >> trigger_test_data
            >> wait_test_data
            >> log_test_data
            >> trigger_qdrant
            >> wait_qdrant
            >> log_qdrant
        )
    else:
        (
            trigger_chunking
            >> wait_chunking
            >> log_chunking
            >> trigger_test_data
            >> wait_test_data
            >> log_test_data
            >> trigger_qdrant
            >> wait_qdrant
            >> log_qdrant
        )
