from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    json_chunks: str
    json_embeddings: str
    model_name: str


class EmbeddingsResponse(BaseModel):
    embeddings_number: int
    output_dir: str
