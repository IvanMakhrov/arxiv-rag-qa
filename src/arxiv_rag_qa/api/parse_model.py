from pydantic import BaseModel


class ParseRequest(BaseModel):
    raw_pdf_dir: str
    metadata_path: str
    processed_json_dir: str


class ParseResponse(BaseModel):
    parsed_count: int
    output_dir: str
