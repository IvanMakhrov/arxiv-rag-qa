import json
import time
from typing import Any

from airflow.exceptions import AirflowException
from airflow.providers.http.hooks.http import HttpHook

from utils.mlflow_logger import log_to_mlflow


def trigger_task(
    endpoint: str,
    payload: dict[str, Any],
    http_conn_id: str = "rag_service",
    **context,
) -> str:
    """Триггер асинхронной задачи"""
    hook = HttpHook(method="POST", http_conn_id=http_conn_id)
    try:
        response = hook.run(
            endpoint=endpoint,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        result = response.json()
        if "task_id" not in result:
            raise AirflowException(f"Missing 'task_id' in response: {result}")
        task_id = result["task_id"]
        context["task_instance"].xcom_push(key="task_id", value=task_id)
        return task_id
    except Exception as e:
        raise AirflowException(f"Trigger failed: {e!s}") from e


def wait_for_task(
    task_id: str,
    http_conn_id: str = "rag_service",
    max_wait_time: int = 3600,
    poll_interval: int = 15,
    **context,
) -> str:
    """Ожидание завершения задачи. Возвращает ТОЛЬКО task_id"""
    hook = HttpHook(method="GET", http_conn_id=http_conn_id)
    start_time = time.time()
    print(f"Waiting for task {task_id} (timeout: {max_wait_time}s)")

    while time.time() - start_time < max_wait_time:
        try:
            response = hook.run(
                endpoint=f"/tasks/{task_id}",
                headers={"Content-Type": "application/json"},
            )
            status_data = response.json()
            status = status_data.get("status", "unknown")
            progress = status_data.get("progress", 0)
            print(f"   Status: {status} | Progress: {progress}%")

            context["task_instance"].xcom_push(key="task_status", value=status)
            context["task_instance"].xcom_push(key="task_progress", value=progress)

            if status == "completed":
                context["task_instance"].xcom_push(key="task_full_response", value=status_data)
                print(f"Task completed: {task_id}")
                return task_id

            if status == "failed":
                error = status_data.get("errorMessage") or status_data.get(
                    "error_message", "Unknown"
                )
                raise AirflowException(f"Task failed: {error}")

            time.sleep(poll_interval)
        except Exception as e:
            print(f"⚠️  Polling error: {e!s}")
            time.sleep(poll_interval)

    raise AirflowException(f"Task timeout after {max_wait_time}s")


def log_task_to_mlflow(
    task_stage: str,
    experiment_name: str = "base_rag",
    **context,
) -> str:
    task_instance = context["task_instance"]
    upstream_task_id = task_instance.task_id.replace("log_", "wait_")

    full_response = task_instance.xcom_pull(task_ids=upstream_task_id, key="task_full_response")

    data = json.loads(full_response["result_data"])

    params = {k: v for k, v in data.items() if k != "results"}
    metrics = data.get("results", {}).get("metrics", {})

    http_response = {"params": params, "metrics": metrics}

    task_id = full_response["id"]
    run_id = log_to_mlflow(
        task_id=task_id,
        http_response=http_response,
        experiment_name=experiment_name,
        run_name=task_stage,
        stage=task_stage,
    )
    task_instance.xcom_push(key="mlflow_run_id", value=run_id)
