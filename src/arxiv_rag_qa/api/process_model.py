from pydantic import BaseModel


class ChunkRequest(BaseModel):
    chunk_dir: str
    json_dir: str
    chunk_size: int
    chunk_overlap: int


class ChunkResponse(BaseModel):
    message: str
    total_chunks: int
    chunk_dir: str
    json_dir: str
    chunk_size: int
    chunk_overlap: int
