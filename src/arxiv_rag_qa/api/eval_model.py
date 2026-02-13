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


class RetrieverEvalRequest(BaseModel):
    bucket_name: str
    test_data_dir: str
    collection_name: str
    top_k: int
    model_name: str
    qdrant_host: str
    qdrant_port: int


class RetrieverEvalResponse(BaseModel):
    message: str
    results: dict
    bucket_name: str
    test_data_dir: str
    collection_name: str
    top_k: int
    model_name: str
