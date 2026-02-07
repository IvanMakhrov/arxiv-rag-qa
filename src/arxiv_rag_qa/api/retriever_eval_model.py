from pydantic import BaseModel


class RetrieverEvalRequest(BaseModel):
    test_data_dir: str
    collection_name: str
    top_k: int
    model_name: str
    qdrant_host: str
    qdrant_port: int


class RetrieverEvalResponse(BaseModel):
    message: str
    results: dict
    test_data_dir: str
    collection_name: str
    top_k: int
    model_name: str
