from pydantic import BaseModel


class GeneratorEvalRequest(BaseModel):
    bucket_name: str
    test_data_dir: str
    collection_name: str
    top_k: int
    emb_model_name: str
    gen_model_name: str
    bertscore_model: str
    qdrant_host: str
    qdrant_port: int
    llm_judge_model: str | None = None
    retriever_type: str = "dense"
    sparse_method: str = "bm25"
    use_qdrant_corpus: bool = True
    hybrid_config: dict | None = None
    sparse_params: dict | None = None


class GeneratorEvalResponse(BaseModel):
    message: str
    results: dict
    bucket_name: str
    test_data_dir: str
    collection_name: str
    top_k: int
    emb_model_name: str
    gen_model_name: str
    bertscore_model: str
    retriever_type: str = "dense"


class RetrieverEvalRequest(BaseModel):
    bucket_name: str
    test_data_dir: str
    collection_name: str
    top_k: int
    model_name: str
    qdrant_host: str
    qdrant_port: int
    retriever_type: str = "dense"
    sparse_method: str = "bm25"
    use_qdrant_corpus: bool = True
    hybrid_config: dict | None = None
    sparse_params: dict | None = None


class RetrieverEvalResponse(BaseModel):
    message: str
    results: dict
    bucket_name: str
    test_data_dir: str
    collection_name: str
    top_k: int
    model_name: str
    retriever_type: str = "dense"
