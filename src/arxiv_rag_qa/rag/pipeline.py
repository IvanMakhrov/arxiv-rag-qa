from arxiv_rag_qa.rag.generator import QwenGenerator
from arxiv_rag_qa.rag.retriever import DenseRetriever


class RAGPipeline:
    def __init__(self, retriever: DenseRetriever, generator: QwenGenerator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, top_k: int = 5) -> dict:
        docs = self.retriever.retrieve(query, top_k=top_k)
        context = "\n\n".join([doc["payload"]["text"] for doc in docs])

        answer = self.generator.generate(query, context)

        return {
            "query": query,
            "answer": answer,
            "retrieved_contexts": [doc["payload"]["text"] for doc in docs],
            "scores": [doc["score"] for doc in docs],
        }
