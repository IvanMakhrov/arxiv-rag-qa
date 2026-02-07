from typing import Any

import mlflow


def log_to_mlflow(
    task_id: str,
    http_response: dict[str, Any],
    experiment_name: str,
    run_id: str | None = None,
    run_name: str | None = None,
    stage: str = "unknown",
) -> str:
    """
    Log Airflow HTTP task results to MLflow.

    Args:
        task_id: Airflow task ID (e.g., 'download_data')
        http_response: JSON response from HttpOperator
        config_params: Relevant config subset (e.g., cfg.download)
        experiment_name: MLflow experiment name
        run_id: Optional run ID to resume
        stage: Semantic stage name (e.g., 'download', 'chunking')
    """

    mlflow.set_experiment(experiment_name)

    metrics = http_response["results"]["metrics"] if "results" in http_response else {}
    params = {f"{stage}_{k}": v for k, v in http_response.items() if k != "results"}

    with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
        mlflow.set_tags({"airflow_task_id": task_id, "stage": stage, "source": "airflow_http_task"})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    return run.info.run_id
