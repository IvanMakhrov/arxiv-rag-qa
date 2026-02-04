from pydantic import BaseModel


class QdrantRequest(BaseModel):
    host: str = ""
    port: int = 0
    collection_name: str = ""
    vector_size: int = 0
    file_path: str
    timeout: int
    batch_size: int


class QdrantResponse(BaseModel):
    collection_name: str = ""
    vector_size: int = 0
