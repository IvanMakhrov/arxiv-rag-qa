import json
from typing import Any

import boto3
import sacrebleu
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# from arxiv_rag_qa.eval.llm_judge import evaluate_llm_metrics  # Disabled: LLM-as-judge removed
from arxiv_rag_qa.rag.generator import QwenGenerator
from arxiv_rag_qa.rag.retriever import DenseRetriever
from utils.setup_logger import setup_logger

# Logging setup
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
        # Use CountVectorizer to tokenize and get vocabulary
        vec = CountVectorizer(ngram_range=(1, 1), lowercase=True, token_pattern=r"\b\w+\b")
        try:
            # Fit on context to get vocabulary
            vec.fit([ctx])
            ctx_vocab = set(vec.get_feature_names_out())
            # Tokenize answer using the same analyzer
            ans_tokens = vec.build_analyzer()(ans)
            if not ans_tokens:
                scores.append(0.0)
                continue
            # Count how many answer tokens are in context vocabulary
            in_context = sum(1 for token in ans_tokens if token in ctx_vocab)
            scores.append(in_context / len(ans_tokens))
        except ValueError:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def generator_eval(
    bucket_name: str,
    test_data_dir: str,
    collection_name: str,
    top_k: int,
    emb_model_name: str,
    gen_model_name: str,
    bertscore_model: str,
    qdrant_host: str,
    qdrant_port: int,
):
    s3_client = get_minio_client()
    test_samples = load_test_set_from_minio(bucket_name, test_data_dir, s3_client)

    embedder = SentenceTransformer(emb_model_name)

    def embedding_func(x):
        return embedder.encode(x, normalize_embeddings=True).tolist()

    retriever = DenseRetriever(
        collection_name=collection_name,
        embedding_model=embedding_func,
        top_k=top_k,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )
    generator = QwenGenerator(model_name=gen_model_name)

    predictions = []
    references = []
    contexts = []
    # questions = []  # Not used without LLM-as-judge

    for i, sample in enumerate(test_samples):
        logger.info(f"\rGenerating {i + 1}/{len(test_samples)}")

        docs = retriever.retrieve(sample["question"], top_k=top_k)
        context = "\n\n".join([doc["payload"]["text"] for doc in docs])

        pred = generator.generate(sample["question"], context)

        predictions.append(pred)
        references.append(sample["answer"])
        contexts.append(context)
        # questions.append(sample["question"])  # Not used without LLM-as-judge

    # Traditional metrics
    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    bert_f1 = compute_bertscore(predictions, references, bertscore_model)
    faithfulness = compute_faithfulness(predictions, contexts)

    # LLM-based metrics
    # llm_metrics = evaluate_llm_metrics(
    #     predictions=predictions,
    #     questions=questions,
    #     ground_truths=references,
    #     contexts=contexts,
    # )

    # Combine all metrics
    all_metrics = {
        "rougeL": rouge["rougeL"],
        "bleu": bleu,
        "bertscore_f1": bert_f1,
        "faithfulness_ngram": faithfulness,
        # **llm_metrics,  # answer_relevance, correctness, faithfulness, conciseness
    }

    logger.info(
        f"Metrics: ROUGE-L: {rouge['rougeL']:.4f}, BLEU: {bleu:.4f}, "
        f"BERTScore F1: {bert_f1:.4f}, Faithfulness (ngram): {faithfulness:.4f}, "
        # f"LLM Judge: {', '.join([f'{k}: {v:.4f}' for k, v in llm_metrics.items()])}"
    )

    return {
        "config": {
            "emb_model": emb_model_name,
            "gen_model": gen_model_name,
            "top_k": top_k,
            "test_file": test_data_dir,
        },
        "metrics": all_metrics,
    }
