from pydantic import BaseModel


class DownloadRequest(BaseModel):
    category: str
    start_date: str
    results_per_request: int
    target_count: int
    pdf_dir: str
    metadata_dir: str


class DownloadResponse(BaseModel):
    message: str
    downloaded_papers_number: int
    category: str
    start_date: str
    results_per_request: int
    target_count: int
    pdf_dir: str
    metadata_dir: str
