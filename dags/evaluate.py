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
        response_filter=lambda response: response.json(),
        do_xcom_push=True,
        data=json.dumps(
            {
                "test_data_dir": cfg.eval.test_data_dir,
                "collection_name": cfg.qdrant.collection_name,
                "top_k": cfg.retriever.top_k,
                "model_name": cfg.embeddings.model_name,
                "qdrant_host": cfg.qdrant_setup.host,
                "qdrant_port": cfg.qdrant_setup.port,
            }
        ),
        response_check=lambda response: response.status_code == cfg.dag.response_check,
    )

    log_retriever_eval = PythonOperator(
        task_id="log_retriever_to_mlflow",
        python_callable=log_task,
        op_kwargs={
            "task_id": "eval_retriever",
            "experiment_name": "base_rag",
            "stage": "retriever_eval",
            "run_name": "retriever_eval",
        },
    )

    generator_eval = HttpOperator(
        task_id="eval_generator",  # Name of DAG in AirFlow UI
        http_conn_id=cfg.dag.http_conn_id,  # Connection_id в AirFlow UI
        endpoint="/generator-eval",  # Router
        method="POST",
        headers={"Content-Type": "application/json"},
        response_filter=lambda response: response.json(),
        do_xcom_push=True,
        data=json.dumps(
            {
                "test_data_dir": cfg.eval.test_data_dir,
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

    log_generator_eval = PythonOperator(
        task_id="log_generator_to_mlflow",
        python_callable=log_task,
        op_kwargs={
            "task_id": "eval_generator",
            "experiment_name": "base_rag",
            "stage": "generator_eval",
            "run_name": "generator_eval",
        },
    )

    retriever_eval >> log_retriever_eval >> generator_eval >> log_generator_eval
