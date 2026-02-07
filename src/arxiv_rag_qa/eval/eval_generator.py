import json
from pathlib import Path

import sacrebleu
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from arxiv_rag_qa.rag.generator import QwenGenerator
from arxiv_rag_qa.rag.retriever import DenseRetriever


def load_test_set(path: str) -> list:
    with Path.open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


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
    Simple n-gram coverage: % of answer unigrams found in context.
    """
    scores = []
    for ans, ctx in zip(answers, contexts, strict=False):
        if not ans.strip():
            scores.append(0.0)
            continue
        vec = CountVectorizer(ngram_range=(1, 1), lowercase=True, token_pattern=r"\b\w+\b")
        try:
            vec.fit_transform([ctx])
            ans_matrix = vec.transform([ans])
            ctx_vocab = set(vec.get_feature_names_out())
            ans_vocab = set(vec.inverse_transform(ans_matrix)[0])
            if not ans_vocab:
                scores.append(0.0)
            else:
                scores.append(len(ans_vocab & ctx_vocab) / len(ans_vocab))
        except ValueError:
            scores.append(0.0)
    return sum(scores) / len(scores)


def generator_eval(
    test_data_dir: str,
    collection_name: str,
    top_k: int,
    emb_model_name: str,
    gen_model_name: str,
    bertscore_model: str,
    qdrant_host: str,
    qdrant_port: int,
):
    test_samples = load_test_set(test_data_dir)

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

    for i, sample in enumerate(test_samples):
        print(f"\rGenerating {i + 1}/{len(test_samples)}", end="", flush=True)

        docs = retriever.retrieve(sample["question"], top_k=top_k)
        context = "\n\n".join([doc["payload"]["text"] for doc in docs])

        pred = generator.generate(sample["question"], context)

        predictions.append(pred)
        references.append(sample["answer"])
        contexts.append(context)

    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    bert_f1 = compute_bertscore(predictions, references, bertscore_model)
    faithfulness = compute_faithfulness(predictions, contexts)

    print(f"ROUGE-L:      {rouge['rougeL']:.4f}")
    print(f"BLEU:         {bleu:.4f}")
    print(f"BERTScore F1: {bert_f1:.4f}")
    print(f"Faithfulness: {faithfulness:.4f}")

    return {
        "config": {
            "emb_model": emb_model_name,
            "gen_model": gen_model_name,
            "top_k": top_k,
            "test_file": test_data_dir,
        },
        "metrics": {
            "rougeL": rouge["rougeL"],
            "bleu": bleu,
            "bertscore_f1": bert_f1,
            "faithfulness": faithfulness,
        },
    }
