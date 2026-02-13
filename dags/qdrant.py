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
    "qdrant_manager",
    default_args=default_args,
    description="Create qdrant collection and add data",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["arxiv", "rag", "data", "qdrant"],
) as dag:
    qdrant_setup = HttpOperator(
        task_id="qdrant_setup",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/qdrant-setup",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        response_filter=lambda response: response.json(),
        do_xcom_push=True,
        data=json.dumps(
            {
                "host": cfg.qdrant.host,
                "port": cfg.qdrant.port,
                "collection_name": cfg.qdrant.collection_name,
                "vector_size": cfg.qdrant.vector_size,
                "bucket_name": cfg.minio.bucket_name,
                "embedding_dir": cfg.embeddings.embedding_dir,
                "timeout": cfg.qdrant.timeout,
                "batch_size": cfg.qdrant.batch_size,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    log_qdrant = PythonOperator(
        task_id="log_qdrant_to_mlflow",
        python_callable=log_task,
        op_kwargs={
            "task_id": "qdrant_setup",
            "experiment_name": "base_rag",
            "stage": "setup_qdrant",
            "run_name": "setup_qdrant",
        },
    )

    qdrant_setup >> log_qdrant
