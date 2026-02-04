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
    "evaluate_rag",
    default_args=default_args,
    description="Evaluate RAG quality",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["arxiv", "rag", "evaluate", "retriever", "generator"],
) as dag:
    retriever_eval = HttpOperator(
        task_id="eval_retriever",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/retriever-eval",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "test_file": cfg.eval.test_file,
                "collection_name": cfg.qdrant.collection_name,
                "top_k": cfg.retriever.top_k,
                "model_name": cfg.embeddings.model_name,
                "qdrant_host": cfg.qdrant_setup.host,
                "qdrant_port": cfg.qdrant_setup.port,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    generator_eval = HttpOperator(
        task_id="eval_generator",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/generator-eval",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "test_file": cfg.eval.test_file,
                "collection_name": cfg.qdrant.collection_name,
                "top_k": cfg.retriever.top_k,
                "emb_model_name": cfg.embeddings.model_name,
                "gen_model_name": cfg.generator.model_name,
                "bertscore_model": cfg.eval.bertscore_model,
                "qdrant_host": cfg.qdrant_setup.host,
                "qdrant_port": cfg.qdrant_setup.port,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    retriever_eval >> generator_eval
