from sentence_transformers import SentenceTransformer

from arxiv_rag_qa.rag.generator import QwenGenerator
from arxiv_rag_qa.rag.pipeline import RAGPipeline
from arxiv_rag_qa.rag.retriever import DenseRetriever


def get_response(
    emb_model_name: str,
    collection_name: str,
    top_k: int,
    gen_model_name: str,
    query: str,
    qdrant_host: str,
    qdrant_port: int,
):
    embedder = SentenceTransformer(emb_model_name)

    def embedding_model(x):
        return embedder.encode(x, normalize_embeddings=True).tolist()

    retriever = DenseRetriever(
        collection_name=collection_name,
        embedding_model=embedding_model,
        top_k=top_k,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )
    generator = QwenGenerator(
        model_name=gen_model_name,
        load_in_4bit=True,
    )

    rag = RAGPipeline(retriever=retriever, generator=generator)

    return rag.run(query)
