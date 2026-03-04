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
    "retries": cfg.infrastructure.dag.retries,
    "retry_delay": timedelta(minutes=cfg.infrastructure.dag.retry_delay),
}


with DAG(
    "evaluate_rag",
    default_args=default_args,
    description="Async RAG quality evaluation with Celery",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["arxiv", "rag", "async", "celery", "evaluate", "retriever", "generator"],
) as dag:
    # ============ RETRIEVER EVALUATION ============
    trigger_retriever_eval = PythonOperator(
        task_id="trigger_retriever_eval",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/retriever-eval",
            "payload": {
                "bucket_name": cfg.infrastructure.minio.bucket_name,
                "test_data_dir": cfg.experiments.test_data.test_data_dir,
                "collection_name": cfg.experiments.qdrant.collection_name,
                "top_k": cfg.experiments.retriever.top_k,
                "model_name": cfg.experiments.embeddings.model_name,
                "qdrant_host": cfg.experiments.qdrant.host,
                "qdrant_port": cfg.experiments.qdrant.port,
            },
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
        },
    )

    wait_retriever_eval = PythonOperator(
        task_id="wait_retriever_eval",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_retriever_eval', "
            "key='task_id') }}",
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
            "max_wait_time": cfg.infrastructure.dag.retriever_eval_timeout,
            "poll_interval": 30,
        },
    )

    log_retriever_eval = PythonOperator(
        task_id="log_retriever_eval",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "retriever_eval",
            "experiment_name": cfg.experiments.mlflow.experiment_name,
        },
    )

    # ============ GENERATOR EVALUATION ============
    trigger_generator_eval = PythonOperator(
        task_id="trigger_generator_eval",
        python_callable=trigger_task,
        op_kwargs={
            "endpoint": "/generator-eval",
            "payload": {
                "bucket_name": cfg.infrastructure.minio.bucket_name,
                "test_data_dir": cfg.experiments.test_data.test_data_dir,
                "collection_name": cfg.experiments.qdrant.collection_name,
                "top_k": cfg.experiments.retriever.top_k,
                "emb_model_name": cfg.experiments.embeddings.model_name,
                "gen_model_name": cfg.experiments.generator.model_name,
                "bertscore_model": cfg.eval.eval.bertscore_model,
                "qdrant_host": cfg.experiments.qdrant.host,
                "qdrant_port": cfg.experiments.qdrant.port,
            },
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
        },
    )

    wait_generator_eval = PythonOperator(
        task_id="wait_generator_eval",
        python_callable=wait_for_task,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_generator_eval', "
            "key='task_id') }}",
            "http_conn_id": cfg.infrastructure.dag.http_conn_id,
            "max_wait_time": cfg.infrastructure.dag.generator_eval_timeout,
            "poll_interval": 30,
        },
    )

    log_generator_eval = PythonOperator(
        task_id="log_generator_eval",
        python_callable=log_task_to_mlflow,
        op_kwargs={
            "task_stage": "generator_eval",
            "experiment_name": cfg.experiments.mlflow.experiment_name,
        },
    )

    # ============ DAG DEPENDENCIES ============
    (
        trigger_retriever_eval
        >> wait_retriever_eval
        >> log_retriever_eval
        >> trigger_generator_eval
        >> wait_generator_eval
        >> log_generator_eval
    )
