from pydantic import BaseModel


class RagRequest(BaseModel):
    emb_model_name: str
    collection_name: str
    top_k: int
    gen_model_name: str
    query: str
    qdrant_host: str
    qdrant_port: int


class RagResponse(BaseModel):
    message: str
    results: dict
    emb_model_name: str
    collection_name: str
    top_k: int
    gen_model_name: str
    query: str
