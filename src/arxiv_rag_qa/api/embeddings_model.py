from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    chunk_dir: str
    embedding_dir: str
    model_name: str


class EmbeddingsResponse(BaseModel):
    message: str
    embeddings_number: int
    chunk_dir: str
    embedding_dir: str
    model_name: str
