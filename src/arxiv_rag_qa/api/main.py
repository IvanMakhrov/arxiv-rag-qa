import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import BaseLoader, Environment, TemplateNotFound
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from starlette.templating import _TemplateResponse

from arxiv_rag_qa.api.data_model import (
    ChunkRequest,
    DownloadRequest,
    EmbeddingsRequest,
    ParseRequest,
    TaskListResponse,
    TestDataRequest,
)
from arxiv_rag_qa.api.eval_model import (
    GeneratorEvalRequest,
    RetrieverEvalRequest,
)
from arxiv_rag_qa.api.middleware import LatencyMiddleware
from arxiv_rag_qa.api.qdrant_model import QdrantRequest
from arxiv_rag_qa.api.rag_model import RagRequest
from arxiv_rag_qa.celery_config import celery_app
from arxiv_rag_qa.db.models import Base, TaskStatus
from utils.setup_logger import setup_logger

logger = setup_logger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(bind=engine)

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

STATIC_DIR = "/app/static"


class DirectFileLoader(BaseLoader):
    """Loader that bypasses Jinja2's internal splitting logic"""

    def __init__(self, searchpath):
        self.searchpath = searchpath

    def get_source(self, environment, template):
        if not isinstance(template, str):
            template = str(template)

        template = template.lstrip("/")

        filepath = Path(self.searchpath) / template

        if not filepath.exists():
            raise TemplateNotFound(template)

        with Path.open(filepath, encoding="utf-8") as f:
            source = f.read()

        return source, str(filepath), lambda: True


loader = DirectFileLoader(STATIC_DIR)
jinja_env = Environment(
    loader=loader,
    autoescape=True,
    cache_size=0,
)
templates = Jinja2Templates(env=jinja_env)

try:
    test = jinja_env.get_template("index.html")
    print("Template engine works")
except Exception as e:
    print(f"Template error: {e}")

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

app = FastAPI(
    title="RAG Service with Async Tasks",
    debug=DEBUG,
    docs_url=None if not DEBUG else "/docs",
    redoc_url=None if not DEBUG else "/redoc",
)
app.add_middleware(LatencyMiddleware)

if DEBUG:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SafeFileSystemLoader(BaseLoader):
    """A safe file system loader that ensures template names are strings"""

    def __init__(self, searchpath):
        self.searchpath = searchpath

    def get_source(self, environment, template):
        if not isinstance(template, str):
            template = str(template)

        template = template.lstrip("/")

        template_path = Path(self.searchpath) / template

        if not template_path.exists():
            raise TemplateNotFound(template)

        with Path.open(template_path, encoding="utf-8") as f:
            source = f.read()

        return source, template_path, lambda: template_path.stat().st_mtime

    def list_templates(self):
        """List all templates in the directory"""
        templates = []
        for entry in Path(self.searchpath).iterdir():
            if entry.suffix == ".html":
                templates.append(entry.name)
        return templates


class SafeTemplateResponse(_TemplateResponse):
    """Custom TemplateResponse that avoids caching issues"""

    def __init__(
        self, template, context, status_code=200, headers=None, media_type=None, background=None
    ):
        clean_context = {}
        for key, value in context.items():
            if key == "request" or not isinstance(value, dict):
                clean_context[key] = value
        super().__init__(template, clean_context, status_code, headers, media_type, background)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve main page"""
    try:
        template = jinja_env.get_template("index.html")
        content = template.render(request=request)
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Error rendering template: {e}", exc_info=True)
        with Path.open(Path(STATIC_DIR) / "index.html", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())


_RAG_DEFAULTS = {name: field.default for name, field in RagRequest.model_fields.items()}


@app.post("/get-response", response_class=HTMLResponse)
async def get_rag_response(  # noqa: PLR0913
    request: Request,
    query: str = Form(..., min_length=1, max_length=2000),
    top_k: int = Form(_RAG_DEFAULTS["top_k"], ge=1, le=50),
    emb_model_name: str = Form(_RAG_DEFAULTS["emb_model_name"]),
    collection_name: str = Form(_RAG_DEFAULTS["collection_name"]),
    gen_model_name: str = Form(_RAG_DEFAULTS["gen_model_name"]),
    qdrant_host: str = Form(_RAG_DEFAULTS["qdrant_host"]),
    qdrant_port: int = Form(_RAG_DEFAULTS["qdrant_port"]),
    retriever_type: str = Form(_RAG_DEFAULTS["retriever_type"], pattern="^(dense|sparse|hybrid)$"),
    sparse_method: str = Form(_RAG_DEFAULTS["sparse_method"], pattern="^(bm25|tfidf)$"),
    use_qdrant_corpus: bool = Form(_RAG_DEFAULTS["use_qdrant_corpus"]),
    in_memory: bool = Form(_RAG_DEFAULTS["in_memory"]),
    hybrid_config: str = Form("{}"),
    sparse_params: str = Form("{}"),
    embedding_model: str = Form(_RAG_DEFAULTS["embedding_model"]),
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
            "retriever_type": retriever_type,
            "sparse_method": sparse_method,
            "use_qdrant_corpus": use_qdrant_corpus,
            "in_memory": in_memory,
            "hybrid_config": json.loads(hybrid_config),
            "sparse_params": json.loads(sparse_params),
            "embedding_model": embedding_model,
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


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status - simplified without response model"""
    try:
        db = SessionLocal()
        task = db.query(TaskStatus).filter(TaskStatus.id == task_id).first()
        db.close()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        result = None
        if task.status == "completed" and task.result:
            try:
                result = json.loads(task.result) if isinstance(task.result, str) else task.result
            except Exception as e:
                logger.error(f"Error parsing result: {e}")
                result = {"answer": str(task.result), "sources": []}

        return {
            "id": task.id,
            "status": task.status,
            "task_type": task.task_type,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": result,
            "error": task.error,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/get-response-json")
async def get_rag_response_json(  # noqa: PLR0913
    query: str = Form(..., min_length=1, max_length=2000),
    top_k: int = Form(_RAG_DEFAULTS["top_k"], ge=1, le=50),
    emb_model_name: str = Form(_RAG_DEFAULTS["emb_model_name"]),
    collection_name: str = Form(_RAG_DEFAULTS["collection_name"]),
    gen_model_name: str = Form(_RAG_DEFAULTS["gen_model_name"]),
    qdrant_host: str = Form(_RAG_DEFAULTS["qdrant_host"]),
    qdrant_port: int = Form(_RAG_DEFAULTS["qdrant_port"]),
    retriever_type: str = Form(_RAG_DEFAULTS["retriever_type"]),
    sparse_method: str = Form(_RAG_DEFAULTS["sparse_method"]),
    use_qdrant_corpus: bool = Form(_RAG_DEFAULTS["use_qdrant_corpus"]),
    in_memory: bool = Form(_RAG_DEFAULTS["in_memory"]),
    hybrid_config: str = Form("{}"),
    sparse_params: str = Form("{}"),
    embedding_model: str = Form(_RAG_DEFAULTS["embedding_model"]),
):
    """JSON endpoint for async form submission"""
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
            "retriever_type": retriever_type,
            "sparse_method": sparse_method,
            "use_qdrant_corpus": use_qdrant_corpus,
            "in_memory": in_memory,
            "hybrid_config": json.loads(hybrid_config),
            "sparse_params": json.loads(sparse_params),
            "embedding_model": embedding_model,
        }

        celery_app.send_task("get_rag_response", args=[task_data], task_id=task_id)

        logger.info(f"RAG query task queued: {task_id}, query: {query[:100]}...")

        return {"task_id": task_id, "status": "pending"}

    except Exception as e:
        logger.error(f"Failed to queue RAG query task: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


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
            "pdf_dir": request.pdf_dir,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "chunking_type": request.chunking_type,
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
            "chunk_dir": request.chunk_dir,
            "retriever_type": request.retriever_type,
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
            "max_questions_per_paper": request.max_questions_per_paper,
            "test_type": request.test_type,
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
            "retriever_type": request.retriever_type,
            "sparse_method": request.sparse_method,
            "use_qdrant_corpus": request.use_qdrant_corpus,
            "hybrid_config": request.hybrid_config,
            "sparse_params": request.sparse_params,
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
            "llm_judge_model": request.llm_judge_model,
            "retriever_type": request.retriever_type,
            "sparse_method": request.sparse_method,
            "use_qdrant_corpus": request.use_qdrant_corpus,
            "hybrid_config": request.hybrid_config,
            "sparse_params": request.sparse_params,
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
