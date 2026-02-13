import json
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import HttpOperator
from airflow.utils.dates import days_ago
from hydra import compose, initialize

from utils.mlflow_logger import log_to_mlflow

with initialize(version_base=None, config_path="../conf", job_name="rag"):
    cfg = compose(config_name="config")

default_args = {
    "owner": "ivan",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": cfg.dag.retries,
    "retry_delay": timedelta(minutes=cfg.dag.retry_delay),
}


def log_task(task_id, experiment_name, stage, run_name, **context):
    http_response = context["task_instance"].xcom_pull(task_ids=task_id)
    run_id = context["task_instance"].xcom_pull(task_ids=task_id, key="mlflow_run_id")

    run_id = log_to_mlflow(
        task_id=task_id,
        http_response=http_response or {},
        experiment_name=experiment_name,
        run_name=run_name,
        stage=stage,
    )

    context["task_instance"].xcom_push(key="mlflow_run_id", value=run_id)


with DAG(
    "prepare_data",
    default_args=default_args,
    description="Data download and chunking",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["arxiv", "rag", "data", "download", "chunking"],
) as dag:
    download_data = HttpOperator(
        task_id="download_data",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/download-papers",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        response_filter=lambda response: response.json(),
        do_xcom_push=True,
        data=json.dumps(
            {
                "category": cfg.download.arxiv_category,
                "start_date": str(cfg.download.start_date),
                "target_count": cfg.download.target_paper_count,
                "results_per_request": cfg.download.results_per_request,
                "bucket_name": cfg.minio.bucket_name,
                "pdf_dir": cfg.download.pdf_dir,
                "metadata_dir": cfg.download.metadata_dir,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    log_download = PythonOperator(
        task_id="log_download_to_mlflow",
        python_callable=log_task,
        op_kwargs={
            "task_id": "download_data",
            "experiment_name": "base_rag",
            "stage": "download",
            "run_name": "download",
        },
    )

    parse_pdf_to_json = HttpOperator(
        task_id="parse_pdf_to_json",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/parse-pdf",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        response_filter=lambda response: response.json(),
        do_xcom_push=True,
        data=json.dumps(
            {
                "bucket_name": cfg.minio.bucket_name,
                "metadata_dir": cfg.download.metadata_dir,
                "json_dir": cfg.download.json_dir,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    log_parse = PythonOperator(
        task_id="log_parse_to_mlflow",
        python_callable=log_task,
        op_kwargs={
            "task_id": "parse_pdf_to_json",
            "experiment_name": "base_rag",
            "stage": "parse",
            "run_name": "parse_data",
        },
    )

    trigger_chunking = HttpOperator(
        task_id="chunking",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/chunking",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        response_filter=lambda response: response.json(),
        do_xcom_push=True,
        data=json.dumps(
            {
                "bucket_name": cfg.minio.bucket_name,
                "chunk_dir": cfg.chunking.chunk_dir,
                "json_dir": cfg.download.json_dir,
                "chunk_size": cfg.chunking.chunk_size,
                "chunk_overlap": cfg.chunking.chunk_overlap,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    log_chunking = PythonOperator(
        task_id="log_chunking_to_mlflow",
        python_callable=log_task,
        op_kwargs={
            "task_id": "chunking",
            "experiment_name": "base_rag",
            "stage": "chunking",
            "run_name": "chunking",
        },
    )

    create_embeddings = HttpOperator(
        task_id="create_embeddings",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/embeddings",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        response_filter=lambda response: response.json(),
        do_xcom_push=True,
        data=json.dumps(
            {
                "bucket_name": cfg.minio.bucket_name,
                "chunk_dir": cfg.chunking.chunk_dir,
                "embedding_dir": cfg.embeddings.embedding_dir,
                "model_name": cfg.embeddings.model_name,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    log_embedding = PythonOperator(
        task_id="log_embedding_to_mlflow",
        python_callable=log_task,
        op_kwargs={
            "task_id": "create_embeddings",
            "experiment_name": "base_rag",
            "stage": "embedding",
            "run_name": "embedding",
        },
    )

    generate_test_data = HttpOperator(
        task_id="generate_test_data",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/generate-test-data",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        response_filter=lambda response: response.json(),
        do_xcom_push=True,
        data=json.dumps(
            {
                "bucket_name": cfg.minio.bucket_name,
                "chunk_dir": cfg.chunking.chunk_dir,
                "test_data_dir": cfg.test_data.test_data_dir,
                "metadata_dir": cfg.download.metadata_dir,
                "test_data_size": cfg.test_data.test_data_size,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    log_generate_test_data = PythonOperator(
        task_id="log_test_data_to_mlflow",
        python_callable=log_task,
        op_kwargs={
            "task_id": "generate_test_data",
            "experiment_name": "base_rag",
            "stage": "generate_test_data",
            "run_name": "generate_test_data",
        },
    )

    (
        download_data
        >> log_download
        >> parse_pdf_to_json
        >> log_parse
        >> trigger_chunking
        >> log_chunking
        >> create_embeddings
        >> log_embedding
        >> generate_test_data
        >> log_generate_test_data
    )
