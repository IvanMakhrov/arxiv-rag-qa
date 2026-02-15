import json
import time
from datetime import timedelta
from typing import Any

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from airflow.providers.http.hooks.http import HttpHook
from airflow.utils.dates import days_ago
from hydra import compose, initialize

from utils.mlflow_logger import log_to_mlflow

# Инициализация конфигурации
with initialize(version_base=None, config_path="../conf", job_name="rag"):
    cfg = compose(config_name="config")

default_args = {
    "owner": "ivan",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": cfg.dag.retries,
    "retry_delay": timedelta(minutes=cfg.dag.retry_delay),
}


def trigger_async_task(
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


def wait_for_task_completion(
    task_id: str,
    http_conn_id: str = "rag_service",
    max_wait_time: int = 3600,
    poll_interval: int = 15,
    **context,
) -> str:
    """Ожидание завершения задачи. Возвращает ТОЛЬКО task_id для простоты."""
    hook = HttpHook(method="GET", http_conn_id=http_conn_id)
    start_time = time.time()
    print(f"⏳ Waiting for task {task_id} (timeout: {max_wait_time}s)")

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

            # Сохраняем ТОЛЬКО необходимое для логирования
            context["task_instance"].xcom_push(key="task_status", value=status)
            context["task_instance"].xcom_push(key="task_progress", value=progress)

            if status == "completed":
                # КРИТИЧЕСКИ ВАЖНО: сохраняем ПОЛНЫЙ ответ для логирования
                context["task_instance"].xcom_push(key="task_full_response", value=status_data)
                print(f"✅ Task completed: {task_id}")
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


def log_completed_task_to_mlflow(  # noqa: C901
    task_stage: str,
    experiment_name: str = "base_rag",
    **context,
) -> str:
    """
    Надёжное логирование результатов асинхронной задачи в MLflow.
    Поддерживает ОБА формата: camelCase (от API) и snake_case (от моделей).
    """
    task_instance = context["task_instance"]
    upstream_task_id = task_instance.task_id.replace("_log", "")  # wait_download → log_download

    # 1. Получаем ПОЛНЫЙ ответ от эндпоинта /tasks/{id}
    full_response = task_instance.xcom_pull(task_ids=upstream_task_id, key="task_full_response")

    if not full_response:
        raise AirflowException(
            f"No task_full_response in XCom from {upstream_task_id}. "
            f"Available XCom keys: {task_instance.xcom_pull(task_ids=upstream_task_id, key=None)}"
        )

    # 2. Вспомогательная функция для безопасного извлечения по разным форматам ключей
    def safe_get(data: dict, *key_variants: str):
        """Извлекает значение по первому найденному ключу (поддержка camelCase/snake_case)"""
        for key in key_variants:
            if key in data:
                return data[key]
            if key.lower() in {k.lower() for k in data}:
                for k in data:  # noqa: PLC0206
                    if k.lower() == key.lower():
                        return data[k]
        return None

    result_data_str = safe_get(full_response, "resultData", "result_data", "resultdata")

    if not result_data_str:
        raise AirflowException(
            f"No result data found in response. Available keys: {list(full_response.keys())}\n"
            f"Full response: {json.dumps(full_response, indent=2, default=str)}"
        )

    try:
        # Сначала пытаемся распарсить как есть
        result_data = json.loads(result_data_str)
    except json.JSONDecodeError:
        try:
            result_data = json.loads(json.loads(result_data_str))
        except Exception as e:
            raise AirflowException(
                f"Cannot parse result data (tried double parsing):\n"
                f"Raw value: {result_data_str[:200]}...\n"
                f"Error: {e}"
            ) from e

    # 5. Формируем структуру для MLflow
    metrics = {}
    params = {}

    for key, value in result_data.items():
        # Числовые значения (но не булевы!) → метрики
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = float(value)
        # Простые типы → параметры
        elif isinstance(value, (str, int, float, bool, type(None))):
            params[key] = value

    http_response = {"results": {"metrics": metrics}, **params}

    # 7. Логирование в MLflow
    try:
        task_id = safe_get(full_response, "id", "taskId", "task_id") or "unknown"
        run_id = log_to_mlflow(
            task_id=task_instance.task_id,
            http_response=http_response,
            experiment_name=experiment_name,
            run_name=f"{task_stage}_{str(task_id)[:8]}",
            stage=task_stage,
        )
        task_instance.xcom_push(key="mlflow_run_id", value=run_id)

        # Подробный лог для отладки
        print("✅ MLflow logging SUCCESS")
        print(f"   Run ID: {run_id}")
        print(f"   Stage: {task_stage}")
        print(f"   Metrics logged: {list(metrics.keys()) or 'none'}")
        print(f"   Params logged: {list(params.keys())[:5]}{'...' if len(params) > 5 else ''}")  # noqa: PLR2004

        return run_id

    except Exception as e:
        # Детальный дамп для отладки
        print(f"❌ MLflow logging FAILED: {e}")
        print(f"   Full response keys: {list(full_response.keys())}")
        print(f"   Result data type: {type(result_data)}")
        print(f"   Result data preview: {str(result_data)[:300]}")
        print(f"   Metrics prepared: {metrics}")
        print(f"   Params prepared: {params}")
        raise AirflowException(f"MLflow logging failed: {e}") from e


# ==================== DAG DEFINITION ====================

with DAG(
    "prepare_data_async",
    default_args=default_args,
    description="Async data pipeline with Celery",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["arxiv", "rag", "async"],
) as dag:
    # ============ DOWNLOAD ============
    trigger_download = PythonOperator(
        task_id="trigger_download",
        python_callable=trigger_async_task,
        op_kwargs={
            "endpoint": "/download-papers",
            "payload": {
                "category": cfg.download.arxiv_category,
                "start_date": str(cfg.download.start_date),
                "target_count": cfg.download.target_paper_count,
                "results_per_request": cfg.download.results_per_request,
                "bucket_name": cfg.minio.bucket_name,
                "pdf_dir": cfg.download.pdf_dir,
                "metadata_dir": cfg.download.metadata_dir,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_download = PythonOperator(
        task_id="wait_download",
        python_callable=wait_for_task_completion,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_download', key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.get("download_timeout", 7200),
            "poll_interval": 30,
        },
    )

    log_download = PythonOperator(
        task_id="log_download",
        python_callable=log_completed_task_to_mlflow,
        op_kwargs={
            "task_stage": "download",
            "experiment_name": "base_rag",
        },
    )

    # ============ PARSE ============
    trigger_parse = PythonOperator(
        task_id="trigger_parse",
        python_callable=trigger_async_task,
        op_kwargs={
            "endpoint": "/parse-pdf",
            "payload": {
                "bucket_name": cfg.minio.bucket_name,
                "metadata_dir": cfg.download.metadata_dir,
                "json_dir": cfg.download.json_dir,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_parse = PythonOperator(
        task_id="wait_parse",
        python_callable=wait_for_task_completion,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_parse', key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.get("parse_timeout", 3600),
            "poll_interval": 20,
        },
    )

    log_parse = PythonOperator(
        task_id="log_parse",
        python_callable=log_completed_task_to_mlflow,
        op_kwargs={
            "task_stage": "parse",
            "experiment_name": "base_rag",
        },
    )

    # ============ CHUNKING ============
    trigger_chunking = PythonOperator(
        task_id="trigger_chunking",
        python_callable=trigger_async_task,
        op_kwargs={
            "endpoint": "/chunking",
            "payload": {
                "bucket_name": cfg.minio.bucket_name,
                "chunk_dir": cfg.chunking.chunk_dir,
                "json_dir": cfg.download.json_dir,
                "chunk_size": cfg.chunking.chunk_size,
                "chunk_overlap": cfg.chunking.chunk_overlap,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_chunking = PythonOperator(
        task_id="wait_chunking",
        python_callable=wait_for_task_completion,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_chunking', key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.get("chunking_timeout", 5400),
            "poll_interval": 25,
        },
    )

    log_chunking = PythonOperator(
        task_id="log_chunking",
        python_callable=log_completed_task_to_mlflow,
        op_kwargs={
            "task_stage": "chunking",
            "experiment_name": "base_rag",
        },
    )

    # ============ EMBEDDINGS ============
    trigger_embeddings = PythonOperator(
        task_id="trigger_embeddings",
        python_callable=trigger_async_task,
        op_kwargs={
            "endpoint": "/embeddings",
            "payload": {
                "bucket_name": cfg.minio.bucket_name,
                "chunk_dir": cfg.chunking.chunk_dir,
                "embedding_dir": cfg.embeddings.embedding_dir,
                "model_name": cfg.embeddings.model_name,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_embeddings = PythonOperator(
        task_id="wait_embeddings",
        python_callable=wait_for_task_completion,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_embeddings', "
            "key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.get("embeddings_timeout", 10800),
            "poll_interval": 45,
        },
    )

    log_embeddings = PythonOperator(
        task_id="log_embeddings",
        python_callable=log_completed_task_to_mlflow,
        op_kwargs={
            "task_stage": "embedding",
            "experiment_name": "base_rag",
        },
    )

    # ============ TEST DATA ============
    trigger_test_data = PythonOperator(
        task_id="trigger_test_data",
        python_callable=trigger_async_task,
        op_kwargs={
            "endpoint": "/generate-test-data",
            "payload": {
                "bucket_name": cfg.minio.bucket_name,
                "chunk_dir": cfg.chunking.chunk_dir,
                "test_data_dir": cfg.test_data.test_data_dir,
                "metadata_dir": cfg.download.metadata_dir,
                "test_data_size": cfg.test_data.test_data_size,
            },
            "http_conn_id": cfg.dag.http_conn_id,
        },
    )

    wait_test_data = PythonOperator(
        task_id="wait_test_data",
        python_callable=wait_for_task_completion,
        op_kwargs={
            "task_id": "{{ task_instance.xcom_pull(task_ids='trigger_test_data', key='task_id') }}",
            "http_conn_id": cfg.dag.http_conn_id,
            "max_wait_time": cfg.dag.get("test_data_timeout", 1800),
            "poll_interval": 15,
        },
    )

    log_test_data = PythonOperator(
        task_id="log_test_data",
        python_callable=log_completed_task_to_mlflow,
        op_kwargs={
            "task_stage": "generate_test_data",
            "experiment_name": "base_rag",
        },
    )

    # ============ DAG DEPENDENCIES ============
    (
        trigger_download
        >> wait_download
        >> log_download
        >> trigger_parse
        >> wait_parse
        >> log_parse
        >> trigger_chunking
        >> wait_chunking
        >> log_chunking
        >> trigger_embeddings
        >> wait_embeddings
        >> log_embeddings
        >> trigger_test_data
        >> wait_test_data
        >> log_test_data
    )
