import json
import time
from typing import Any

import boto3
import sacrebleu
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import CountVectorizer

from arxiv_rag_qa.eval.eval_retriever import create_eval_retriever
from arxiv_rag_qa.eval.llm_judge import evaluate_llm_metrics
from arxiv_rag_qa.rag.generator import create_generator
from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


def get_minio_client():
    return boto3.client("s3")


def load_test_set_from_minio(bucket: str, test_data_key: str, s3_client: Any) -> list:
    """Load test samples from MinIO JSONL file."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=test_data_key)
        samples = []
        for line in response["Body"].iter_lines():
            if line:
                samples.append(json.loads(line.decode("utf-8")))
        return samples
    except Exception as e:
        logger.error(f"Test file not found in s3://{bucket}/{test_data_key}: {e}")
        raise FileNotFoundError(f"Test file not found in s3://{bucket}/{test_data_key}: {e}") from e


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions, strict=False)]
    return {"rougeL": sum(s["rougeL"].fmeasure for s in scores) / len(scores)}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, refs)
    return bleu.score / 100.0


def compute_bertscore(predictions: list[str], references: list[str], bertscore_model: str) -> float:
    _, _, f1 = bert_score(
        predictions, references, model_type=bertscore_model, lang="en", verbose=False
    )
    return f1.mean().item()


def compute_faithfulness(answers: list[str], contexts: list[str]) -> float:
    """
    Compute faithfulness as the percentage of answer unigrams that appear in the context.
    """
    scores = []
    for ans, ctx in zip(answers, contexts, strict=False):
        if not ans.strip():
            scores.append(0.0)
            continue
        vec = CountVectorizer(ngram_range=(1, 1), lowercase=True, token_pattern=r"\b\w+\b")
        try:
            vec.fit([ctx])
            ctx_vocab = set(vec.get_feature_names_out())
            ans_tokens = vec.build_analyzer()(ans)
            if not ans_tokens:
                scores.append(0.0)
                continue
            in_context = sum(1 for token in ans_tokens if token in ctx_vocab)
            scores.append(in_context / len(ans_tokens))
        except ValueError:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def generator_eval(  # noqa: PLR0913
    bucket_name: str,
    test_data_dir: str,
    collection_name: str,
    top_k: int,
    emb_model_name: str,
    gen_model_name: str,
    bertscore_model: str,
    qdrant_host: str,
    qdrant_port: int,
    llm_judge_model: str | None = None,
    retriever_type: str = "dense",
    sparse_method: str = "bm25",
    use_qdrant_corpus: bool = True,
    hybrid_config: dict | None = None,
    sparse_params: dict | None = None,
):
    s3_client = get_minio_client()
    test_samples = load_test_set_from_minio(bucket_name, test_data_dir, s3_client)

    retriever = create_eval_retriever(
        retriever_type=retriever_type,
        collection_name=collection_name,
        top_k=top_k,
        model_name=emb_model_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        sparse_method=sparse_method,
        use_qdrant_corpus=use_qdrant_corpus,
        hybrid_config=hybrid_config or {},
        sparse_params=sparse_params or {},
    )
    generator_config = {
        "type": "local",
        "model_name": gen_model_name,
    }
    generator = create_generator(generator_config)

    predictions = []
    references = []
    contexts = []
    questions_list = []
    all_num_tokens: list[int] = []

    total_retrieve_time = 0.0
    total_generate_time = 0.0
    sample_count = len(test_samples)

    for i, sample in enumerate(test_samples):
        logger.info(f"\rGenerating {i + 1}/{sample_count}")

        t0 = time.perf_counter()
        docs = retriever.retrieve(sample["question"], top_k=top_k)
        retrieve_time = time.perf_counter() - t0
        total_retrieve_time += retrieve_time

        context = "\n\n".join([doc["payload"]["text"] for doc in docs])

        t0 = time.perf_counter()
        pred, num_tokens = generator.generate(sample["question"], context)
        generate_time = time.perf_counter() - t0
        total_generate_time += generate_time

        predictions.append(pred)
        all_num_tokens.append(num_tokens)
        references.append(sample["answer"])
        contexts.append(context)
        questions_list.append(sample["question"])

    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    bert_f1 = compute_bertscore(predictions, references, bertscore_model)

    llm_metrics = evaluate_llm_metrics(
        predictions=predictions,
        questions=questions_list,
        ground_truths=references,
        contexts=contexts,
        model=llm_judge_model,
    )

    total_time = total_retrieve_time + total_generate_time
    avg_latency_seconds = total_time / sample_count
    avg_retrieve_time = total_retrieve_time / sample_count
    avg_generate_time = total_generate_time / sample_count

    total_tokens = sum(all_num_tokens)
    avg_tokens_per_sample = total_tokens / sample_count if sample_count else 0
    tokens_per_second = total_tokens / total_generate_time if total_generate_time > 0 else 0.0
    tokens_per_second_e2e = total_tokens / total_time if total_time > 0 else 0.0

    throughput_qps = sample_count / total_time if total_time > 0 else 0.0

    all_metrics = {
        "rougeL": rouge["rougeL"],
        "bleu": bleu,
        "bertscore_f1": bert_f1,
        **llm_metrics,
        "e2e_latency_avg": avg_latency_seconds,
        "e2e_latency_avg_ms": avg_latency_seconds * 1000,
        "retrieve_latency_avg": avg_retrieve_time,
        "generate_latency_avg": avg_generate_time,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_e2e": tokens_per_second_e2e,
        "total_tokens_generated": total_tokens,
        "avg_tokens_per_sample": avg_tokens_per_sample,
        "throughput_qps": throughput_qps,
        "total_eval_time": total_time,
        "sample_count": sample_count,
    }

    logger.info(
        f"Retriever type: {retriever_type}, "
        f"Metrics: ROUGE-L: {rouge['rougeL']:.4f}, BLEU: {bleu:.4f}, "
        f"BERTScore F1: {bert_f1:.4f}, "
        f"LLM Judge: {', '.join([f'{k}: {v:.4f}' for k, v in llm_metrics.items()])}"
    )
    logger.info(
        f"Perf: Avg E2E latency: {avg_latency_seconds:.2f}s, "
        f"Token throughput: {tokens_per_second:.2f} tok/s (generate), "
        f"{tokens_per_second_e2e:.2f} tok/s (E2E), "
        f"Avg {avg_tokens_per_sample:.1f} tok/sample, "
        f"Avg retrieve: {avg_retrieve_time:.2f}s, Avg generate: {avg_generate_time:.2f}s"
    )

    return {
        "config": {
            "emb_model": emb_model_name,
            "gen_model": gen_model_name,
            "top_k": top_k,
            "test_file": test_data_dir,
            "retriever_type": retriever_type,
            "sparse_method": sparse_method,
        },
        "metrics": all_metrics,
    }
