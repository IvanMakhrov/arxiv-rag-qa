from pydantic import BaseModel


class GeneratorEvalRequest(BaseModel):
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
    test_data_dir: str
    collection_name: str
    top_k: int
    emb_model_name: str
    gen_model_name: str
    bertscore_model: str
