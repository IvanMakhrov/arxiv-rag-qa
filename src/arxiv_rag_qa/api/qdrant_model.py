from pydantic import BaseModel


class QdrantRequest(BaseModel):
    host: str
    port: int
    collection_name: str
    vector_size: int
    bucket_name: str
    embedding_dir: str
    chunk_dir: str = ""
    retriever_type: str = "dense"
    timeout: int
    batch_size: int


class QdrantResponse(BaseModel):
    message: str
    points_number: int
    collection_name: str
    vector_size: int
    bucket_name: str
    embedding_dir: str
    chunk_dir: str = ""
    retriever_type: str = "dense"
    batch_size: int
