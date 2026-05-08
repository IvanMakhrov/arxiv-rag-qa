from pydantic import BaseModel, Field


class RagRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=7, ge=1, le=50)
    emb_model_name: str = Field(default="all-MiniLM-L6-v2")
    collection_name: str = Field(default="dense")
    gen_model_name: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    qdrant_host: str = Field(default="qdrant")
    qdrant_port: int = Field(default=6333)
    retriever_type: str = Field(default="dense", pattern="^(dense|sparse|hybrid)$")
    sparse_method: str = Field(default="bm25", pattern="^(bm25|tfidf)$")
    use_qdrant_corpus: bool = Field(default=True)
    in_memory: bool = Field(default=False)
    hybrid_config: dict = Field(default_factory=dict)
    sparse_params: dict = Field(default_factory=dict)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")


class RagResponse(BaseModel):
    message: str
    results: dict
    emb_model_name: str
    collection_name: str
    top_k: int
    gen_model_name: str
    query: str
