import json
import os
import random
import re
from collections import defaultdict
from typing import Any

import boto3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)

# Expected lengths for chunk ID components
HASH_LENGTH = 8
CHUNK_IDX_LENGTH = 4
MIN_CHUNK_ID_PARTS = 4
MIN_SECTION_PARTS_FOR_SEPARATION = 2
MIN_KEYWORD_LENGTH = 3


def get_minio_client():
    """Create S3 client for MinIO."""
    return boto3.client("s3")


def load_chunks_from_minio(
    bucket_name: str, chunk_key: str, s3_client: Any
) -> list[dict[str, Any]]:
    """Load chunked documents with metadata from MinIO."""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=chunk_key)
        chunks = []
        for line in response["Body"].iter_lines():
            if line:
                chunks.append(json.loads(line.decode("utf-8")))
        logger.info(f"Loaded {len(chunks)} chunks from s3://{bucket_name}/{chunk_key}")
        return chunks
    except Exception as e:
        logger.error(f"Chunk file not found in s3://{bucket_name}/{chunk_key}: {e}")
        raise FileNotFoundError(
            f"Chunk file not found in s3://{bucket_name}/{chunk_key}: {e}"
        ) from e


def parse_chunk_id(chunk_id: str) -> dict[str, Any]:
    """
    Parse chunk ID to extract arxiv_id, section, chunk_idx, and hash.
    Expected format: {arxiv_id}_{section}_{chunk_idx:04d}_{hash}
    Example: 2603.25537_abstract_0000_bee2a393

    Returns:
        dict with keys: arxiv_id, section, chunk_idx, hash
    """
    if not chunk_id:
        return {"arxiv_id": "", "section": "", "chunk_idx": 0, "hash": ""}

    parts = chunk_id.split("_")
    if len(parts) < MIN_CHUNK_ID_PARTS:
        logger.warning(f"Invalid chunk ID format (too few parts): {chunk_id}")
        return {"arxiv_id": "", "section": "", "chunk_idx": 0, "hash": ""}

    hash_part = parts[-1]
    chunk_idx_part = parts[-2]
    section_parts = parts[:-2]

    if not (len(hash_part) == HASH_LENGTH and re.match(r"^[a-f0-9]+$", hash_part)):
        logger.warning(f"Invalid hash in chunk ID: {chunk_id}")
        return {"arxiv_id": "", "section": "", "chunk_idx": 0, "hash": ""}

    if not (len(chunk_idx_part) == CHUNK_IDX_LENGTH and chunk_idx_part.isdigit()):
        logger.warning(f"Invalid chunk_idx in chunk ID: {chunk_id}")
        return {"arxiv_id": "", "section": "", "chunk_idx": 0, "hash": ""}

    if len(section_parts) >= MIN_SECTION_PARTS_FOR_SEPARATION:
        arxiv_id = ".".join(section_parts[:-1])
        section = section_parts[-1]
    elif len(section_parts) == 1:
        arxiv_id = section_parts[0]
        section = ""
    else:
        arxiv_id = ""
        section = ""

    if not arxiv_id or "." not in arxiv_id:
        logger.warning(f"Invalid arxiv_id (missing dot or empty) in chunk ID: {chunk_id}")
        return {"arxiv_id": "", "section": "", "chunk_idx": 0, "hash": ""}

    return {
        "arxiv_id": arxiv_id,
        "section": section,
        "chunk_idx": int(chunk_idx_part),
        "hash": hash_part,
    }


def group_chunks_by_arxiv(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    paper_chunks = defaultdict(list)
    for chunk in chunks:
        chunk_id = chunk.get("id", "")
        if not chunk_id:
            continue
        parsed = parse_chunk_id(chunk_id)
        arxiv_id = parsed["arxiv_id"]
        if arxiv_id:
            paper_chunks[arxiv_id].append(chunk)
        else:
            logger.debug(f"Skipping chunk with invalid ID: {chunk_id}")

    logger.info(f"Grouped {len(chunks)} chunks into {len(paper_chunks)} papers")
    return dict(paper_chunks)


class LLMGenerator:
    """LLM Generator for test data generation using OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base = api_base or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        self.model = model or os.getenv("OPENROUTER_MODEL")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        logger.info(f"LLM Generator initialized with model: {self.model}")

    @staticmethod
    def extract_json_from_text(text: str) -> str:  # noqa: C901, PLR0912
        """
        Extract JSON string from LLM response text.
        Handles code blocks, finds balanced braces for nested objects.
        """
        if text is None:
            return ""

        text = text.strip()
        code_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            content = max(matches, key=len).strip()
            text = content

        if not text.startswith("{"):
            start = text.find("{")
            if start == -1:
                return ""
            text = text[start:]

        brace_count = 0
        in_string = False
        escape_next = False
        result_end = None

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    result_end = i + 1
                    break

        if result_end is not None:
            json_str = text[:result_end].strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass

        brace_count = 0
        start_idx = 0
        for i, char in enumerate(text):
            if char == "{":
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start_idx : i + 1].strip()
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "qa_pairs" in parsed:
                            return candidate
                    except json.JSONDecodeError:
                        continue

        if '"qa_pairs"' in text:
            return text

        return ""

    def generate_qa_pairs(self, prompt: str) -> list[dict[str, str]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful academic Q&A generator. "
                    "Respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 2500,
            "stream": False,
        }

        try:
            response = self.session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            if "choices" not in result or not result["choices"]:
                logger.error(f"Invalid API response: missing 'choices' or empty: {result}")
                raise ValueError("Invalid API response: missing 'choices' or empty")

            message = result["choices"][0]["message"]
            content = message.get("content")

            if content is None:
                content = (
                    message.get("thinking")
                    or message.get("reasoning")
                    or message.get("reasoning_content")
                )
                if content:
                    logger.info("Using content from 'reasoning' field")
                else:
                    logger.warning("LLM returned None content with no reasoning fallback")
                    raise ValueError("LLM returned empty content")

            if not isinstance(content, str):
                raise ValueError(f"LLM returned non-string content: {type(content)}")

            if not content.strip():
                raise ValueError("LLM returned empty or whitespace-only content")

            json_str = LLMGenerator.extract_json_from_text(content)
            if not json_str:
                logger.warning(f"Failed to extract JSON from content: {content}")
                raise ValueError("No JSON found in LLM response")

            parsed = json.loads(json_str)
            qa_pairs = parsed.get("qa_pairs", [])

            if not isinstance(qa_pairs, list):
                logger.warning(f"qa_pairs is not a list: {type(qa_pairs)}")
                return []

            return qa_pairs

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API call failed: {e}")
            raise
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Unexpected response or JSON parse error: {e}")
            raise ValueError(f"Invalid response from LLM: {e}") from e


def generate_questions_for_paper(  # noqa: C901, PLR0912
    arxiv_id: str,
    chunks: list[dict[str, Any]],
    num_questions: int = 3,
    llm_generator: LLMGenerator | None = None,
) -> list[dict[str, Any]]:
    """
    Generate question-answer pairs for a single paper using LLM.

    Args:
        arxiv_id: arXiv identifier (e.g., "2603.25537")
        chunks: List of all chunks for this paper
        num_questions: Number of questions to generate (default 3)
        llm_generator: Optional LLMGenerator instance (created if None)

    Returns:
        List of Q&A dicts with question, answer, relevant_chunk_ids, category
    """
    if llm_generator is None:
        llm_generator = LLMGenerator()

    sorted_chunks = sorted(chunks, key=lambda c: c.get("metadata", {}).get("chunk_idx", 0))

    paper_text_parts = []
    for chunk in sorted_chunks:
        text = chunk.get("text", "").strip()
        if text:
            paper_text_parts.append(text)

    full_text = "\n\n".join(paper_text_parts)

    if not full_text:
        logger.warning(f"Empty text for paper {arxiv_id}, skipping")
        return []

    prompt = f"""You are an expert academic researcher. Given the following academic paper text,
    generate exactly {num_questions} question-answer pairs.

Requirements:
1. Each question should be answerable FROM THE PROVIDED TEXT.
2. Questions should cover different aspects: main contribution, methodology, experiments/results,
specific details.
3. Answers must be directly extractable from the text (verbatim or near-verbatim).
4. Include specific details, numbers, names where appropriate.

Output format (strict JSON only):
{{
  "qa_pairs": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    ...
  ]
}}

Paper text (first 12000 characters):
-------------------
{full_text[:12000]}
-------------------

Generate {num_questions} diverse Q&A pairs:"""

    try:
        logger.debug(f"Calling LLM API for paper {arxiv_id}")
        qa_pairs = llm_generator.generate_qa_pairs(prompt)

        if not qa_pairs:
            logger.warning(f"No Q&A pairs returned from LLM for {arxiv_id}")
            return []

        qa_pairs = qa_pairs[:num_questions]

        all_chunk_ids = [c.get("id", "") for c in sorted_chunks if c.get("id")]

        results = []
        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()

            if not question or not answer:
                logger.warning(f"Skipping Q&A with empty question/answer for {arxiv_id}")
                continue

            relevant_chunk_ids = []
            answer_snippet = answer[:150].lower()

            for chunk in sorted_chunks:
                chunk_text = chunk.get("text", "").strip().lower()
                chunk_id = chunk.get("id", "")
                if not chunk_text or not chunk_id:
                    continue
                if answer_snippet in chunk_text:
                    relevant_chunk_ids.append(chunk_id)

            if not relevant_chunk_ids:
                answer_keywords = set(
                    word for word in answer_snippet.split() if len(word) > MIN_KEYWORD_LENGTH
                )
                if answer_keywords:
                    keyword_scores = []
                    for chunk in sorted_chunks:
                        chunk_text = chunk.get("text", "").strip().lower()
                        chunk_id = chunk.get("id", "")
                        if not chunk_text or not chunk_id:
                            continue
                        chunk_keywords = set(
                            word for word in chunk_text.split() if len(word) > MIN_KEYWORD_LENGTH
                        )
                        overlap = len(answer_keywords.intersection(chunk_keywords))
                        if overlap > 0:
                            keyword_scores.append((chunk_id, overlap))

                    if keyword_scores:
                        keyword_scores.sort(key=lambda x: x[1], reverse=True)
                        relevant_chunk_ids = [cid for cid, _ in keyword_scores[:5]]

            if not relevant_chunk_ids and all_chunk_ids:
                mid_idx = len(all_chunk_ids) // 2
                window_size = min(5, len(all_chunk_ids))
                start = max(0, mid_idx - window_size // 2)
                end = min(len(all_chunk_ids), start + window_size)
                relevant_chunk_ids = all_chunk_ids[start:end]

            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "relevant_chunk_ids": relevant_chunk_ids,
                    "category": "generated",
                }
            )

        logger.info(f"Generated {len(results)} Q&A pairs for paper {arxiv_id}")
        return results

    except Exception as e:
        logger.error(f"LLM generation failed for paper {arxiv_id}: {type(e).__name__}: {e}")
        return []


def generate_test_data_llm(
    bucket_name: str,
    chunk_dir: str,
    test_data_dir: str,
    test_data_size: int = 0,
    max_questions_per_paper: int = 3,
) -> int:
    """
    Generate test data using LLM from chunks stored in MinIO.

    Args:
        bucket_name: MinIO bucket name (default from config: rag-data)
        chunk_dir: S3 key to chunks JSONL file (e.g., "data/chunks/chunks.jsonl")
        test_data_dir: S3 key where test data JSONL will be saved
        test_data_size: Percentage of papers to sample (0-100), 0 means all papers.
                       The function will keep processing papers until it successfully
                       generates questions for the target number of papers, or exhausts
                       all available papers.
        max_questions_per_paper: Number of questions to generate per paper (default 3)

    Returns:
        Number of test samples generated
    """
    s3_client = get_minio_client()

    logger.info(f"Loading chunks from s3://{bucket_name}/{chunk_dir}")
    chunks = load_chunks_from_minio(bucket_name, chunk_dir, s3_client)

    if not chunks:
        raise ValueError("No chunks loaded. Check chunk file exists and is not empty.")

    paper_to_chunks = group_chunks_by_arxiv(chunks)

    if not paper_to_chunks:
        raise ValueError("No valid paper groupings found in chunks")

    total_papers = len(paper_to_chunks)
    if test_data_size > 0:
        target_successful_papers = max(1, int(total_papers * test_data_size / 100))
    else:
        target_successful_papers = total_papers

    target_successful_papers = min(target_successful_papers, total_papers)

    logger.info(
        f"Target: successfully generate test data for {target_successful_papers} papers "
        f"(out of {total_papers} total), with up to {max_questions_per_paper} questions each"
    )

    paper_ids = list(paper_to_chunks.keys())
    random.shuffle(paper_ids)

    llm_generator = LLMGenerator()

    all_samples = []
    successful_papers = 0
    processed_papers = 0
    max_attempts = total_papers

    while successful_papers < target_successful_papers and processed_papers < max_attempts:
        arxiv_id = paper_ids[processed_papers]
        processed_papers += 1

        logger.info(
            f"Processing paper {processed_papers}/{total_papers}: {arxiv_id} "
            f"(successful: {successful_papers}/{target_successful_papers})"
        )

        paper_chunks = paper_to_chunks[arxiv_id]
        samples = generate_questions_for_paper(
            arxiv_id=arxiv_id,
            chunks=paper_chunks,
            num_questions=max_questions_per_paper,
            llm_generator=llm_generator,
        )

        if samples:
            all_samples.extend(samples)
            successful_papers += 1
        else:
            logger.warning(f"No samples generated for {arxiv_id}, continuing...")

    logger.info(
        f"Finished processing: {successful_papers} successful papers out of "
        f"{processed_papers} attempted (target: {target_successful_papers})"
    )

    if all_samples:
        jsonl_content = "\n".join(json.dumps(sample, ensure_ascii=False) for sample in all_samples)

        s3_client.put_object(
            Bucket=bucket_name,
            Key=test_data_dir,
            Body=jsonl_content.encode("utf-8"),
            ContentType="application/jsonl",
        )

        logger.info(f"Saved {len(all_samples)} test samples to s3://{bucket_name}/{test_data_dir}")
    else:
        logger.warning("No test samples were generated")

    return len(all_samples)
