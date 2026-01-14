import time
import xml.etree.ElementTree as ET
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import mlflow
import requests
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Configuration
ARXIV_CATEGORY = "cs.CL"
START_DATE = "202501010000"
TARGET_PAPER_COUNT = 3000
RESULTS_PER_REQUEST = 10

RAW_PDF_DIR = "/opt/airflow/data/raw"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15"
}

default_args = {
    "owner": "ivan",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def fetch_arxiv_pdfs(**context):
    """Download NLP papers from arXiv"""
    Path(RAW_PDF_DIR).mkdir(parents=True)

    downloaded = set()
    start_index = 0

    print(f"Target: {TARGET_PAPER_COUNT} papers from '{ARXIV_CATEGORY}' since {START_DATE[:8]}")

    while len(downloaded) < TARGET_PAPER_COUNT:
        remaining = TARGET_PAPER_COUNT - len(downloaded)
        batch_size = min(RESULTS_PER_REQUEST, remaining)

        search_query = f"cat:{ARXIV_CATEGORY} AND submittedDate:[{START_DATE} TO 999912312359]"
        encoded_query = quote(search_query)

        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={encoded_query}&"
            f"sortBy=submittedDate&"
            f"sortOrder=descending&"
            f"start={start_index}&"
            f"max_results={batch_size}"
        )

        print(
            f"→ Requesting papers {start_index}-{start_index + batch_size - 1} "
            f"(total so far: {len(downloaded)})"
        )

        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"API request failed: {e}. Retrying in 5s")
            time.sleep(5)
            continue

        root = ET.fromstring(response.content)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")

        if not entries:
            print("No more papers available")
            break

        for entry in entries:
            try:
                paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
                pdf_path = Path(RAW_PDF_DIR) / f"{paper_id}.pdf"

                if paper_id not in downloaded:
                    if not Path(pdf_path).exists():
                        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                        pdf_resp = requests.get(pdf_url, headers=HEADERS, timeout=10)
                        pdf_resp.raise_for_status()
                        with Path(pdf_path).open("wb") as f:
                            f.write(pdf_resp.content)
                        print(f"Downloaded: {paper_id}")
                    else:
                        print(f"Already exists: {paper_id}")
                    downloaded.add(paper_id)
            except Exception as e:
                print(f"Error processing {paper_id}: {e}")
                continue

        start_index += batch_size
        time.sleep(3)

        if len(entries) < batch_size:
            print("Reached end of results")
            break

    downloaded_list = sorted(list(downloaded))
    full_paths = [Path(RAW_PDF_DIR) / f"{pid}.pdf" for pid in downloaded_list]

    context["ti"].xcom_push(key="downloaded_pdfs", value=full_paths)
    print(f"\nTotal papers: {len(full_paths)}")
    return len(full_paths)


def log_mlflow(**context):
    """Log to MLflow"""
    downloaded_pdfs = context["ti"].xcom_pull(key="downloaded_pdfs", task_ids="fetch_pdfs")

    mlflow.set_tracking_uri("http://mlflow-service:5000")
    with mlflow.start_run(run_name="arxiv_nlp_parse"):
        mlflow.log_param("arxiv_category", ARXIV_CATEGORY)
        mlflow.log_param("start_date", START_DATE[:8])
        mlflow.log_param("num_papers", len(downloaded_pdfs))


with DAG(
    "arxiv_nlp_pipeline",
    default_args=default_args,
    description="Download NLP papers from arXiv",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["nlp", "arxiv", "rag"],
) as dag:
    fetch_task = PythonOperator(
        task_id="fetch_pdfs",
        python_callable=fetch_arxiv_pdfs,
    )

    parse_task = PythonOperator(
        task_id="log_mlflow",
        python_callable=log_mlflow,
    )

    dvc_track_task = BashOperator(
        task_id="dvc_track_data",
        bash_command="""
        set -a
        source .env
        set +a
        cd /opt/airflow && \
        dvc add data/raw && \
        git add data/*.dvc && \
        git commit -m 'Update arXiv NLP dataset''
        """,
    )

    fetch_task >> parse_task >> dvc_track_task
