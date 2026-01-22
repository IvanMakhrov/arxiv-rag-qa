from pydantic import BaseModel


class DownloadRequest(BaseModel):
    category: str
    start_date: str
    results_per_request: int
    target_count: int
    raw_pdf_dir: str
    metadata_path: str


class DownloadResponse(BaseModel):
    downloaded_papers_number: int
    output_dir: str
