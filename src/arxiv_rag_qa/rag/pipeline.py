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

        sources = []
        seen_arxiv_ids = set()
        for doc in docs:
            payload = doc["payload"]
            arxiv_id = payload.get("arxiv_id", "")
            if arxiv_id and arxiv_id not in seen_arxiv_ids:
                seen_arxiv_ids.add(arxiv_id)
                sources.append(
                    {
                        "arxiv_id": arxiv_id,
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "score": doc["score"],
                    }
                )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_contexts": [doc["payload"]["text"] for doc in docs],
            "scores": [doc["score"] for doc in docs],
        }
