from datetime import datetime

from pydantic import BaseModel


class DownloadRequest(BaseModel):
    category: str
    start_date: str
    results_per_request: int
    bucket_name: str
    target_count: int
    pdf_dir: str
    metadata_dir: str


class DownloadResponse(BaseModel):
    message: str
    downloaded_papers_number: int
    category: str
    start_date: str
    results_per_request: int
    bucket_name: str
    target_count: int
    pdf_dir: str
    metadata_dir: str


class ParseRequest(BaseModel):
    bucket_name: str
    metadata_dir: str
    json_dir: str


class ParseResponse(BaseModel):
    message: str
    parsed_papers_number: int
    bucket_name: str
    metadata_dir: str
    json_dir: str


class ChunkRequest(BaseModel):
    bucket_name: str
    chunk_dir: str
    pdf_dir: str
    chunk_size: int
    chunk_overlap: int
    chunking_type: str


class ChunkResponse(BaseModel):
    message: str
    bucket_name: str
    total_chunks: int
    chunk_dir: str
    json_dir: str
    chunk_size: int
    chunk_overlap: int


class EmbeddingsRequest(BaseModel):
    bucket_name: str
    chunk_dir: str
    embedding_dir: str
    model_name: str
    batch_size: int
    checkpoint_interval: int


class EmbeddingsResponse(BaseModel):
    message: str
    bucket_name: str
    embeddings_number: int
    chunk_dir: str
    embedding_dir: str
    model_name: str


class TestDataRequest(BaseModel):
    bucket_name: str
    chunk_dir: str
    test_data_dir: str
    metadata_dir: str
    test_data_size: int


class TestDataResponse(BaseModel):
    message: str
    bucket_name: str
    chunk_dir: str
    test_data_dir: str
    metadata_dir: str
    test_data_size: int


class TaskStatusResponse(BaseModel):
    id: str
    task_type: str
    status: str  # pending, processing, completed, failed
    created_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    progress: int
    result_data: str | None  # JSON string


class TaskListResponse(BaseModel):
    tasks: list[TaskStatusResponse]
    total: int
