from pydantic import BaseModel


class ProcessRequest(BaseModel):
    raw_json_dir: str
    output_chunks_path: str
    chunk_size: int
    chunk_overlap: int


class ProcessResponse(BaseModel):
    total_chunks: int
    output_file: str
