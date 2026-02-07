from pydantic import BaseModel


class ParseRequest(BaseModel):
    pdf_dir: str
    metadata_dir: str
    json_dir: str


class ParseResponse(BaseModel):
    parsed_papers_number: int
    json_dir: str
