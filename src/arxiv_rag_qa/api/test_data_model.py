from pydantic import BaseModel


class TestDataRequest(BaseModel):
    chunk_dir: str
    test_data_dir: str
    metadata_dir: str
    test_data_size: int


class TestDataResponse(BaseModel):
    message: str
    chunk_dir: str
    test_data_dir: str
    metadata_dir: str
    test_data_size: int
