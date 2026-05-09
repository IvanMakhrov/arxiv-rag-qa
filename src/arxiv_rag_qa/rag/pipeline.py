import time

from arxiv_rag_qa.rag.generator import OpenRouterGenerator, QwenGenerator
from arxiv_rag_qa.rag.retriever import DenseRetriever


class RAGPipeline:
    def __init__(self, retriever: DenseRetriever, generator: QwenGenerator | OpenRouterGenerator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        docs = self.retriever.retrieve(query, top_k=top_k)
        retrieve_time = time.perf_counter() - t0

        context = "\n\n".join([doc["payload"]["text"] for doc in docs])

        t0 = time.perf_counter()
        answer, num_tokens = self.generator.generate(query, context)
        generate_time = time.perf_counter() - t0

        total_time = retrieve_time + generate_time
        tokens_per_sec = num_tokens / generate_time if generate_time > 0 else 0.0

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
            "timing": {
                "retrieve_time_s": round(retrieve_time, 4),
                "generate_time_s": round(generate_time, 4),
                "total_time_s": round(total_time, 4),
            },
            "token_usage": {
                "generated_tokens": num_tokens,
                "tokens_per_second": round(tokens_per_sec, 2),
            },
        }
