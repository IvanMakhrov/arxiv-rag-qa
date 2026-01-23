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
        data=json.dumps(
            {
                "category": cfg.download.arxiv_category,
                "start_date": str(cfg.download.start_date),
                "target_count": cfg.download.target_paper_count,
                "results_per_request": cfg.download.results_per_request,
                "raw_pdf_dir": cfg.download.raw_pdf_dir,
                "metadata_path": cfg.download.metadata_path,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    parse_pdf_to_json = HttpOperator(
        task_id="parse_pdf_to_json",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/parse-pdfs",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "raw_pdf_dir": cfg.download.raw_pdf_dir,
                "metadata_path": cfg.download.metadata_path,
                "processed_json_dir": cfg.download.processed_json_dir,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    trigger_chunking = HttpOperator(
        task_id="chunking",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/process-all-papers",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "output_chunks_path": cfg.chunking.output_path,
                "raw_json_dir": cfg.download.processed_json_dir,
                "chunk_size": cfg.chunking.chunk_size,
                "chunk_overlap": cfg.chunking.chunk_overlap,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    create_embeddings = HttpOperator(
        task_id="create_embeddings",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/embeddings",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "json_chunks": cfg.chunking.output_path,
                "json_embeddings": cfg.embeddings.json_embeddings,
                "model_name": cfg.embeddings.model_name,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )
    download_data >> parse_pdf_to_json >> trigger_chunking >> create_embeddings
