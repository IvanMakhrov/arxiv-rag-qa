from datetime import timedelta

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from hydra import compose, initialize

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
    dag_id="experiment_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["rag", "orchestration"],
) as dag:
    trigger_building_rag = TriggerDagRunOperator(
        task_id="trigger_build_rag",
        trigger_dag_id="build_rag",
        wait_for_completion=True,
    )

    trigger_evaluation = TriggerDagRunOperator(
        task_id="trigger_evaluate_rag",
        trigger_dag_id="evaluate_rag",
        wait_for_completion=True,
    )

    trigger_building_rag >> trigger_evaluation
