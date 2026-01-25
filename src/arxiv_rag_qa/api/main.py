import logging

from fastapi import FastAPI, HTTPException

from arxiv_rag_qa.api.download_model import DownloadRequest, DownloadResponse
from arxiv_rag_qa.api.embeddings_model import EmbeddingsRequest, EmbeddingsResponse
from arxiv_rag_qa.api.parse_model import ParseRequest, ParseResponse
from arxiv_rag_qa.api.process_model import ProcessRequest, ProcessResponse
from arxiv_rag_qa.api.qdrant_model import QdrantRequest, QdrantResponse
from arxiv_rag_qa.data.chunking import process_all_papers_to_chunks
from arxiv_rag_qa.data.download_data import fetch_arxiv_pdfs
from arxiv_rag_qa.data.generate_embeddings import generate_embeddings
from arxiv_rag_qa.data.parse_pdf_to_json import parse_pdfs_to_json
from arxiv_rag_qa.rag.qdrant_manager import QdrantManager

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastApi setup
app = FastAPI(title="RAG service")


@app.post("/download-papers", response_model=DownloadResponse)
def download_papers(request: DownloadRequest):
    """Download and parse arXiv papers."""
    try:
        count = fetch_arxiv_pdfs(
            category=request.category,
            start_date=request.start_date,
            target_count=request.target_count,
            results_per_request=request.results_per_request,
            raw_pdf_dir=request.raw_pdf_dir,
            metadata_path=request.metadata_path,
        )
        return DownloadResponse(downloaded_papers_number=count, output_dir=str(request.raw_pdf_dir))
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/parse-pdfs", response_model=ParseResponse)
def parse_pdfs(request: ParseRequest):
    """Convert PDFs to JSON with full text."""
    try:
        count = parse_pdfs_to_json(
            raw_pdf_dir=request.raw_pdf_dir,
            metadata_path=request.metadata_path,
            processed_json_dir=request.processed_json_dir,
        )
        return ParseResponse(parsed_count=count, output_dir=request.processed_json_dir)
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/process-all-papers", response_model=ProcessResponse)
def process_all_papers(request: ProcessRequest):
    """Chunk all JSON files into embeddings-ready format."""
    try:
        total_chunks = process_all_papers_to_chunks(
            output_chunks_path=request.output_chunks_path,
            raw_json_dir=request.raw_json_dir,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return ProcessResponse(total_chunks=total_chunks, output_file=request.output_chunks_path)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/embeddings", response_model=EmbeddingsResponse)
def create_embeddings(request: EmbeddingsRequest):
    """Create embeddings of chunked data texts"""
    try:
        count = generate_embeddings(
            json_chunks=request.json_chunks,
            json_embeddings=request.json_embeddings,
            model_name=request.model_name,
        )
        return EmbeddingsResponse(embeddings_number=count, output_dir=str(request.json_embeddings))
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/qdrant_setup", response_model=DownloadResponse)
def qdrant_setup(request: QdrantRequest):
    """Setup Qdrant db"""
    try:
        qdrant = QdrantManager(
            host=request.host,
            port=request.port,
            collection_name=request.collection_name,
            vector_size=request.vector_size,
        )
        qdrant.setup()
        return QdrantResponse(
            collection_name=request.collection_name,
            vector_size=request.vector_size,
        )
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
