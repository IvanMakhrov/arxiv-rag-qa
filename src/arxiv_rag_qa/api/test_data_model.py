from pydantic import BaseModel


class TestDataRequest(BaseModel):
    chunks_path: str
    test_data_path: str
    metadata_path: str
    test_data_size: int


class TestDataResponse(BaseModel):
    test_data_path: str
    message: str
