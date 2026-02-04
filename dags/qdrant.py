import json
from datetime import timedelta

from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator
from airflow.utils.dates import days_ago
from hydra import compose, initialize

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
    description="Create qdrant collection and add data",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["arxiv", "rag", "data", "qdrant"],
) as dag:
    qdrant_setup = HttpOperator(
        task_id="qdrant-setup",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/qdrant-setup",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "host": cfg.qdrant.host,
                "port": cfg.qdrant.port,
                "collection_name": cfg.qdrant.collection_name,
                "vector_size": cfg.qdrant.vector_size,
                "file_path": cfg.embeddings.json_embeddings,
                "timeout": cfg.qdrant.timeout,
                "batch_size": cfg.qdrant.batch_size,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    # qdrant_setup
