import logging

from fastapi import FastAPI, HTTPException

from arxiv_rag_qa.api.data_model import (
    ChunkRequest,
    ChunkResponse,
    DownloadRequest,
    DownloadResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ParseRequest,
    ParseResponse,
    TestDataRequest,
    TestDataResponse,
)
from arxiv_rag_qa.api.eval_model import (
    GeneratorEvalRequest,
    GeneratorEvalResponse,
    RetrieverEvalRequest,
    RetrieverEvalResponse,
)
from arxiv_rag_qa.api.qdrant_model import QdrantRequest, QdrantResponse
from arxiv_rag_qa.api.rag_model import RagRequest, RagResponse
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
            bucket_name=request.bucket_name,
            pdf_dir=request.pdf_dir,
            metadata_dir=request.metadata_dir,
        )
        return DownloadResponse(
            message="PDF downloaded successfully",
            downloaded_papers_number=count,
            category=request.category,
            start_date=request.start_date,
            target_count=request.target_count,
            results_per_request=request.results_per_request,
            bucket_name=request.bucket_name,
            pdf_dir=request.pdf_dir,
            metadata_dir=request.metadata_dir,
        )
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/parse-pdf", response_model=ParseResponse)
def parse_pdfs(request: ParseRequest):
    """Convert PDFs to JSON with full text."""
    try:
        count = parse_pdfs_to_json(
            bucket_name=request.bucket_name,
            metadata_dir=request.metadata_dir,
            json_dir=request.json_dir,
        )
        return ParseResponse(
            message="PDF to json parsed successfully",
            parsed_papers_number=count,
            bucket_name=request.bucket_name,
            metadata_dir=request.metadata_dir,
            json_dir=request.json_dir,
        )
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/chunking", response_model=ChunkResponse)
def process_all_papers(request: ChunkRequest):
    """Chunk all JSON files into embeddings-ready format."""
    try:
        total_chunks = process_all_papers_to_chunks(
            bucket_name=request.bucket_name,
            chunk_dir=request.chunk_dir,
            json_dir=request.json_dir,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return ChunkResponse(
            message="Chunking success",
            bucket_name=request.bucket_name,
            total_chunks=total_chunks,
            chunk_dir=request.chunk_dir,
            json_dir=request.json_dir,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/embeddings", response_model=EmbeddingsResponse)
def create_embeddings(request: EmbeddingsRequest):
    """Create embeddings of chunked data texts"""
    try:
        count = generate_embeddings(
            bucket_name=request.bucket_name,
            chunk_dir=request.chunk_dir,
            embedding_dir=request.embedding_dir,
            model_name=request.model_name,
        )
        return EmbeddingsResponse(
            message="Embedding success",
            bucket_name=request.bucket_name,
            embeddings_number=count,
            chunk_dir=request.chunk_dir,
            embedding_dir=request.embedding_dir,
            model_name=request.model_name,
        )
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
            bucket_name=request.bucket_name,
            embedding_dir=request.embedding_dir,
            timeout=request.timeout,
            batch_size=request.batch_size,
        )
        count = qdrant.setup()
        return QdrantResponse(
            message="Collection created and data added",
            points_number=count,
            collection_name=request.collection_name,
            vector_size=request.vector_size,
            bucket_name=request.bucket_name,
            embedding_dir=request.embedding_dir,
            batch_size=request.batch_size,
        )
    except Exception as e:
        logger.error(f"Qdrant setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/generate-test-data", response_model=TestDataResponse)
def generate_test_dataset(request: TestDataRequest):
    """Generate test data"""
    try:
        generate_test_data(
            bucket_name=request.bucket_name,
            chunk_dir=request.chunk_dir,
            test_data_dir=request.test_data_dir,
            metadata_dir=request.metadata_dir,
            test_data_size=request.test_data_size,
        )
        return TestDataResponse(
            message="Test data was generated",
            bucket_name=request.bucket_name,
            chunk_dir=request.chunk_dir,
            test_data_dir=request.test_data_dir,
            metadata_dir=request.metadata_dir,
            test_data_size=request.test_data_size,
        )
    except Exception as e:
        logger.error(f"Test data generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/retriever-eval", response_model=RetrieverEvalResponse)
def eval_retriever(request: RetrieverEvalRequest):
    """Evaluate retriever"""
    try:
        count = retriever_eval(
            bucket_name=request.bucket_name,
            test_data_dir=request.test_data_dir,
            collection_name=request.collection_name,
            top_k=request.top_k,
            model_name=request.model_name,
            qdrant_host=request.qdrant_host,
            qdrant_port=request.qdrant_port,
        )
        return RetrieverEvalResponse(
            message="Retriever evaluated",
            results=count,
            bucket_name=request.bucket_name,
            test_data_dir=request.test_data_dir,
            collection_name=request.collection_name,
            top_k=request.top_k,
            model_name=request.model_name,
        )
    except Exception as e:
        logger.error(f"Retriever evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/generator-eval", response_model=GeneratorEvalResponse)
def eval_generator(request: GeneratorEvalRequest):
    """Evaluate generator"""
    try:
        count = generator_eval(
            bucket_name=request.bucket_name,
            test_data_dir=request.test_data_dir,
            collection_name=request.collection_name,
            top_k=request.top_k,
            emb_model_name=request.emb_model_name,
            gen_model_name=request.gen_model_name,
            bertscore_model=request.bertscore_model,
            qdrant_host=request.qdrant_host,
            qdrant_port=request.qdrant_port,
        )
        return GeneratorEvalResponse(
            message="Generator evaluated",
            results=count,
            bucket_name=request.bucket_name,
            test_data_dir=request.test_data_dir,
            collection_name=request.collection_name,
            top_k=request.top_k,
            emb_model_name=request.emb_model_name,
            gen_model_name=request.gen_model_name,
            bertscore_model=request.bertscore_model,
        )
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
        return RagResponse(
            message="Query was processed",
            results=count,
            emb_model_name=request.emb_model_name,
            collection_name=request.collection_name,
            top_k=request.top_k,
            gen_model_name=request.gen_model_name,
            query=request.query,
        )
    except Exception as e:
        logger.error(f"Failed to get response from RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
