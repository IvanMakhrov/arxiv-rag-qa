from sentence_transformers import SentenceTransformer

from arxiv_rag_qa.rag.generator import QwenGenerator
from arxiv_rag_qa.rag.pipeline import RAGPipeline
from arxiv_rag_qa.rag.retriever import DenseRetriever


def main():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def embedding_func(x):
        return embedder.encode(x, normalize_embeddings=True).tolist()

    retriever = DenseRetriever(
        collection_name="arxiv_rag",
        embedding_model=embedding_func,
        top_k=5,
    )
    generator = QwenGenerator(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        load_in_4bit=True,
    )

    rag = RAGPipeline(retriever=retriever, generator=generator)

    query = "What is a critical challenge in CPT?"
    result = rag.run(query)

    print("Query:", result["query"])
    print("Answer:", result["answer"])
    print("\nRetrieved Contexts:")
    for i, ctx in enumerate(result["retrieved_contexts"]):
        print(f"{i + 1}. {ctx[:200]}...")


if __name__ == "__main__":
    main()
