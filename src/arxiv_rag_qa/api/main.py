import logging

from fastapi import FastAPI, HTTPException

from arxiv_rag_qa.api.download_model import DownloadRequest, DownloadResponse
from arxiv_rag_qa.api.embeddings_model import EmbeddingsRequest, EmbeddingsResponse
from arxiv_rag_qa.api.generator_eval_model import GeneratorEvalRequest, GeneratorEvalResponse
from arxiv_rag_qa.api.parse_model import ParseRequest, ParseResponse
from arxiv_rag_qa.api.process_model import ProcessRequest, ProcessResponse
from arxiv_rag_qa.api.qdrant_model import QdrantRequest, QdrantResponse
from arxiv_rag_qa.api.rag_model import RagRequest, RagResponse
from arxiv_rag_qa.api.retriever_eval_model import RetrieverEvalRequest, RetrieverEvalResponse
from arxiv_rag_qa.api.test_data_model import TestDataRequest, TestDataResponse
from arxiv_rag_qa.data.chunking import process_all_papers_to_chunks
from arxiv_rag_qa.data.download_data import fetch_arxiv_pdfs
from arxiv_rag_qa.data.generate_embeddings import generate_embeddings
from arxiv_rag_qa.data.generate_test_data import generate_test_data
from arxiv_rag_qa.data.parse_pdf_to_json import parse_pdfs_to_json
from arxiv_rag_qa.eval.eval_generator import generator_eval
from arxiv_rag_qa.eval.eval_retriever import retriever_eval
from arxiv_rag_qa.rag.qdrant_manager import QdrantManager
from arxiv_rag_qa.rag.rag import get_response

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
        logger.error(f"Embeddings generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/qdrant-setup", response_model=QdrantResponse)
def qdrant_setup(request: QdrantRequest):
    """Setup Qdrant db"""
    try:
        qdrant = QdrantManager(
            host=request.host,
            port=request.port,
            collection_name=request.collection_name,
            vector_size=request.vector_size,
            file_path=request.file_path,
            timeout=request.timeout,
            batch_size=request.batch_size,
        )
        qdrant.setup()
        return QdrantResponse(
            collection_name=request.collection_name,
            vector_size=request.vector_size,
        )
    except Exception as e:
        logger.error(f"Qdrant setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/generate-test-data", response_model=TestDataResponse)
def generate_test_dataset(request: TestDataRequest):
    """Generate test data"""
    try:
        generate_test_data(
            chunks_path=request.chunks_path,
            test_data_path=request.test_data_path,
            metadata_path=request.metadata_path,
            test_data_size=request.test_data_size,
        )
        return TestDataResponse(
            test_data_path=request.test_data_path, message="Test data was generated"
        )
    except Exception as e:
        logger.error(f"Test data generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/retriever-eval", response_model=RetrieverEvalResponse)
def eval_retriever(request: RetrieverEvalRequest):
    """Evaluate retriever"""
    try:
        count = retriever_eval(
            test_file=request.test_file,
            collection_name=request.collection_name,
            top_k=request.top_k,
            model_name=request.model_name,
            qdrant_host=request.qdrant_host,
            qdrant_port=request.qdrant_port,
        )
        return RetrieverEvalResponse(results=count)
    except Exception as e:
        logger.error(f"Retriever evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/generator-eval", response_model=GeneratorEvalResponse)
def eval_generator(request: GeneratorEvalRequest):
    """Evaluate generator"""
    try:
        count = generator_eval(
            test_file=request.test_file,
            collection_name=request.collection_name,
            top_k=request.top_k,
            emb_model_name=request.emb_model_name,
            gen_model_name=request.gen_model_name,
            bertscore_model=request.bertscore_model,
            qdrant_host=request.qdrant_host,
            qdrant_port=request.qdrant_port,
        )
        return GeneratorEvalResponse(results=count)
    except Exception as e:
        logger.error(f"Generator evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/get-response", response_model=RagResponse)
def get_rag_response(request: RagRequest):
    """Get RAG response"""
    try:
        count = get_response(
            emb_model_name=request.emb_model_name,
            collection_name=request.collection_name,
            top_k=request.top_k,
            gen_model_name=request.gen_model_name,
            query=request.query,
            qdrant_host=request.qdrant_host,
            qdrant_port=request.qdrant_port,
        )
        return RagResponse(results=count)
    except Exception as e:
        logger.error(f"Failed to get response from RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
