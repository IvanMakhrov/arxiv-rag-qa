import json
import os
import uuid
from datetime import datetime

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from arxiv_rag_qa.api.data_model import (
    ChunkRequest,
    DownloadRequest,
    EmbeddingsRequest,
    ParseRequest,
    TaskListResponse,
    TaskStatusResponse,
    TestDataRequest,
)
from arxiv_rag_qa.api.eval_model import (
    GeneratorEvalRequest,
    RetrieverEvalRequest,
)
from arxiv_rag_qa.api.middleware import LatencyMiddleware
from arxiv_rag_qa.api.qdrant_model import QdrantRequest
from arxiv_rag_qa.celery_config import celery_app
from arxiv_rag_qa.db.models import Base, TaskStatus
from utils.setup_logger import setup_logger

# Logging setup
logger = setup_logger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Создание таблиц
Base.metadata.create_all(bind=engine)

# Production settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# FastAPI setup
app = FastAPI(
    title="RAG Service with Async Tasks",
    debug=DEBUG,
    docs_url=None if not DEBUG else "/docs",  # Disable Swagger in production
    redoc_url=None if not DEBUG else "/redoc",
)
app.add_middleware(LatencyMiddleware)

# Mount static files (nginx will handle this, but keep for local dev)
if DEBUG:
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get-response", response_class=HTMLResponse)
async def get_rag_response(
    request: Request,
    query: str = Form(..., min_length=1, max_length=2000),
    top_k: int = Form(5, ge=1, le=50),
    emb_model_name: str = Form("all-MiniLM-L6-v2"),
    collection_name: str = Form("arxiv_rag"),
    gen_model_name: str = Form("Qwen/Qwen2.5-0.5B-Instruct"),
    qdrant_host: str = Form("qdrant"),
    qdrant_port: int = Form(6333),
):
    """
    Асинхронный RAG-запрос через HTML-форму.
    Принимает form-data (без JS), отправляет задачу в Celery,
    перенаправляет на страницу статуса meta-refresh.
    """
    try:
        task_id = str(uuid.uuid4())

        db = SessionLocal()
        task = TaskStatus(
            id=task_id,
            task_type="get_rag_response",
            status="pending",
            created_at=datetime.utcnow(),
            query=query,
        )
        db.add(task)
        db.commit()
        db.close()

        task_data = {
            "emb_model_name": emb_model_name,
            "collection_name": collection_name,
            "top_k": top_k,
            "gen_model_name": gen_model_name,
            "query": query,
            "qdrant_host": qdrant_host,
            "qdrant_port": qdrant_port,
        }

        celery_app.send_task("get_rag_response", args=[task_data], task_id=task_id)

        logger.info(f"RAG query task queued: {task_id}, query: {query[:100]}...")

        return RedirectResponse(url=f"/task/{task_id}", status_code=303)

    except Exception as e:
        logger.error(f"Failed to queue RAG query task: {e}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Ошибка обработки запроса: {e!s}", "query": query},
        )


@app.get("/task/{task_id}", response_class=HTMLResponse)
async def task_status_page(request: Request, task_id: str):
    try:
        db = SessionLocal()
        task = db.query(TaskStatus).filter(TaskStatus.id == task_id).first()
        db.close()

        if not task:
            raise HTTPException(status_code=404, detail="Задача не найдена")

        refresh_seconds = 3 if task.status in ["pending", "processing"] else 0

        result = None
        if task.status == "completed" and task.result:
            try:
                result = json.loads(task.result) if isinstance(task.result, str) else task.result
            except Exception:
                result = {"answer": task.result, "sources": []}

        return templates.TemplateResponse(
            "task_status.html",
            {
                "request": request,
                "task_id": task_id,
                "task": task,
                "result": result,
                "refresh_seconds": refresh_seconds,
                "query": task.query,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to render task status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки статуса: {e!s}") from e


@app.get("/health")
async def health_check():
    """Health endpoint for nginx/Docker"""
    return {"status": "healthy", "service": "rag-service"}


@app.post("/download-papers", response_model=dict)
async def download_papers(request: DownloadRequest):
    """Асинхронное скачивание статей - возвращает ID задачи"""
    try:
        task_id = str(uuid.uuid4())
        db = SessionLocal()
        task = TaskStatus(
            id=task_id, task_type="download_papers", status="pending", created_at=datetime.utcnow()
        )
        db.add(task)
        db.commit()
        db.close()

        task_data = {
            "category": request.category,
            "start_date": request.start_date,
            "target_count": request.target_count,
            "results_per_request": request.results_per_request,
            "bucket_name": request.bucket_name,
            "pdf_dir": request.pdf_dir,
            "metadata_dir": request.metadata_dir,
        }

        celery_app.send_task("download_papers", args=[task_data], task_id=task_id)

        logger.info("Data downloaded successfully")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Download task queued successfully",
        }

    except Exception as e:
        logger.error(f"Failed to queue download task: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/parse-pdf", response_model=dict)
async def parse_pdfs(request: ParseRequest):
    """Асинхронный парсинг PDF - возвращает ID задачи"""
    try:
        task_id = str(uuid.uuid4())
        db = SessionLocal()
        task = TaskStatus(
            id=task_id, task_type="parse_pdfs", status="pending", created_at=datetime.utcnow()
        )
        db.add(task)
        db.commit()
        db.close()

        task_data = {
            "bucket_name": request.bucket_name,
            "metadata_dir": request.metadata_dir,
            "json_dir": request.json_dir,
        }

        celery_app.send_task("parse_pdfs", args=[task_data], task_id=task_id)

        logger.info("Data parsed successfully")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Parse task queued successfully",
        }

    except Exception as e:
        logger.error(f"Failed to queue parse task: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/chunking", response_model=dict)
async def process_all_papers(request: ChunkRequest):
    """Асинхронное чанкирование - возвращает ID задачи"""
    try:
        task_id = str(uuid.uuid4())
        db = SessionLocal()
        task = TaskStatus(
            id=task_id, task_type="process_chunks", status="pending", created_at=datetime.utcnow()
        )
        db.add(task)
        db.commit()
        db.close()

        task_data = {
            "bucket_name": request.bucket_name,
            "chunk_dir": request.chunk_dir,
            "json_dir": request.json_dir,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
        }

        celery_app.send_task("process_chunks", args=[task_data], task_id=task_id)

        logger.info("Data chunked successfully")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Chunking task queued successfully",
        }

    except Exception as e:
        logger.error(f"Failed to queue chunking task: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/embeddings", response_model=dict)
async def create_embeddings(request: EmbeddingsRequest):
    """Асинхронная генерация эмбеддингов - возвращает ID задачи"""
    try:
        task_id = str(uuid.uuid4())
        db = SessionLocal()
        task = TaskStatus(
            id=task_id,
            task_type="create_embeddings",
            status="pending",
            created_at=datetime.utcnow(),
        )
        db.add(task)
        db.commit()
        db.close()

        task_data = {
            "bucket_name": request.bucket_name,
            "chunk_dir": request.chunk_dir,
            "embedding_dir": request.embedding_dir,
            "model_name": request.model_name,
            "batch_size": request.batch_size,
            "checkpoint_interval": request.checkpoint_interval,
        }

        celery_app.send_task("create_embeddings", args=[task_data], task_id=task_id)

        logger.info("Embeddings created successfully")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Embeddings generation task queued successfully",
        }

    except Exception as e:
        logger.error(f"Failed to queue embeddings task: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/qdrant-setup", response_model=dict)
async def qdrant_setup(request: QdrantRequest):
    """Асинхронная настройка Qdrant - возвращает ID задачи"""
    try:
        task_id = str(uuid.uuid4())
        db = SessionLocal()
        task = TaskStatus(
            id=task_id, task_type="setup_qdrant", status="pending", created_at=datetime.utcnow()
        )
        db.add(task)
        db.commit()
        db.close()

        task_data = {
            "host": request.host,
            "port": request.port,
            "collection_name": request.collection_name,
            "vector_size": request.vector_size,
            "bucket_name": request.bucket_name,
            "embedding_dir": request.embedding_dir,
            "timeout": request.timeout,
            "batch_size": request.batch_size,
        }

        celery_app.send_task("setup_qdrant", args=[task_data], task_id=task_id)
        logger.info("Qdrant setup done successfully")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Qdrant setup task queued successfully",
        }

    except Exception as e:
        logger.error(f"Failed to queue Qdrant setup task: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/generate-test-data", response_model=dict)
async def generate_test_dataset(request: TestDataRequest):
    """Асинхронная генерация тестовых данных - возвращает ID задачи"""
    try:
        task_id = str(uuid.uuid4())
        db = SessionLocal()
        task = TaskStatus(
            id=task_id,
            task_type="generate_test_data",
            status="pending",
            created_at=datetime.utcnow(),
        )
        db.add(task)
        db.commit()
        db.close()

        task_data = {
            "bucket_name": request.bucket_name,
            "chunk_dir": request.chunk_dir,
            "test_data_dir": request.test_data_dir,
            "metadata_dir": request.metadata_dir,
            "test_data_size": request.test_data_size,
        }

        celery_app.send_task("generate_test_data", args=[task_data], task_id=task_id)
        logger.info("Test data generated successfully")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Test data generation task queued successfully",
        }

    except Exception as e:
        logger.error(f"Failed to queue test data generation task: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/retriever-eval", response_model=dict)
async def eval_retriever(request: RetrieverEvalRequest):
    """Асинхронная оценка ретривера - возвращает ID задачи"""
    try:
        task_id = str(uuid.uuid4())
        db = SessionLocal()
        task = TaskStatus(
            id=task_id,
            task_type="evaluate_retriever",
            status="pending",
            created_at=datetime.utcnow(),
        )
        db.add(task)
        db.commit()
        db.close()

        task_data = {
            "bucket_name": request.bucket_name,
            "test_data_dir": request.test_data_dir,
            "collection_name": request.collection_name,
            "top_k": request.top_k,
            "model_name": request.model_name,
            "qdrant_host": request.qdrant_host,
            "qdrant_port": request.qdrant_port,
        }

        celery_app.send_task("evaluate_retriever", args=[task_data], task_id=task_id)
        logger.info("Retriever evaluated successfully")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Retriever evaluation task queued successfully",
        }

    except Exception as e:
        logger.error(f"Failed to queue retriever evaluation task: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/generator-eval", response_model=dict)
async def eval_generator(request: GeneratorEvalRequest):
    """Асинхронная оценка генератора - возвращает ID задачи"""
    try:
        task_id = str(uuid.uuid4())
        db = SessionLocal()
        task = TaskStatus(
            id=task_id,
            task_type="evaluate_generator",
            status="pending",
            created_at=datetime.utcnow(),
        )
        db.add(task)
        db.commit()
        db.close()

        task_data = {
            "bucket_name": request.bucket_name,
            "test_data_dir": request.test_data_dir,
            "collection_name": request.collection_name,
            "top_k": request.top_k,
            "emb_model_name": request.emb_model_name,
            "gen_model_name": request.gen_model_name,
            "bertscore_model": request.bertscore_model,
            "qdrant_host": request.qdrant_host,
            "qdrant_port": request.qdrant_port,
        }

        celery_app.send_task("evaluate_generator", args=[task_data], task_id=task_id)
        logger.info("Generator evaluated successfully")

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Generator evaluation task queued successfully",
        }

    except Exception as e:
        logger.error(f"Failed to queue generator evaluation task: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ===== Эндпоинты для отслеживания статуса задач =====


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Получить статус конкретной задачи"""
    try:
        db = SessionLocal()
        task = db.query(TaskStatus).filter(TaskStatus.id == task_id).first()
        db.close()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return TaskStatusResponse(**task.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    task_type: str | None = None, status: str | None = None, limit: int = 100, offset: int = 0
):
    try:
        db = SessionLocal()

        query = select(TaskStatus)

        if task_type:
            query = query.where(TaskStatus.task_type == task_type)
        if status:
            query = query.where(TaskStatus.status == status)

        query = query.order_by(TaskStatus.created_at.desc()).limit(limit).offset(offset)

        result = db.execute(query)
        tasks = [task.to_dict() for task in result.scalars().all()]

        db.close()

        return TaskListResponse(tasks=tasks, total=len(tasks))

    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    try:
        db = SessionLocal()
        task = db.query(TaskStatus).filter(TaskStatus.id == task_id).first()

        if not task:
            db.close()
            raise HTTPException(status_code=404, detail="Task not found")

        db.delete(task)
        db.commit()
        db.close()

        return {"message": "Task deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete task: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/tasks/stats")
async def get_tasks_stats():
    """Получить статистику по задачам"""
    try:
        db = SessionLocal()

        total = db.query(TaskStatus).count()
        pending = db.query(TaskStatus).filter(TaskStatus.status == "pending").count()
        processing = db.query(TaskStatus).filter(TaskStatus.status == "processing").count()
        completed = db.query(TaskStatus).filter(TaskStatus.status == "completed").count()
        failed = db.query(TaskStatus).filter(TaskStatus.status == "failed").count()

        db.close()

        return {
            "total": total,
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
        }

    except Exception as e:
        logger.error(f"Failed to get tasks stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
