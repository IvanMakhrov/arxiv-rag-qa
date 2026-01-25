from pydantic import BaseModel


class QdrantRequest(BaseModel):
    host: str = ""
    port: int = 0
    collection_name: str = ""
    vector_size: int = 0


class QdrantResponse(BaseModel):
    collection_name: str = ""
    vector_size: int = 0
