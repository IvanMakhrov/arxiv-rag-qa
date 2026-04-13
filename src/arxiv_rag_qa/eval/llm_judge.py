import json
import os
import re
from typing import Any, ClassVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


class LLMJudge:
    """LLM Judge for evaluating RAG outputs using OpenRouter."""

    MAX_SCORE: ClassVar[int] = 5
    PROMPTS: ClassVar[dict[str, str]] = {
        "answer_relevance": """You are an expert evaluator for question-answering systems.
        Evaluate how well the answer addresses the question.

        IMPORTANT: Output ONLY a valid JSON object with exactly this key:
        "score": <integer between 1 and 5>
        Do not include any additional text, reasoning, or explanations outside the JSON.

        Question: {question}
        Answer: {answer}
        """,
        "answer_correctness": """You are an expert evaluator for factual accuracy.
        Compare the predicted answer with the ground truth and rate factual correctness.
        IMPORTANT: Output ONLY a valid JSON object with exactly this key:
        "score": <integer between 1 and 5>
        Do not include any additional text, reasoning, or explanations outside the JSON.

        Question: {question}
        Ground Truth: {ground_truth}
        Prediction: {prediction}
        """,
        "faithfulness": """You are an expert evaluator for faithfulness/grounding.
        Verify whether all claims in the answer are supported by the provided context.
        IMPORTANT: Output ONLY a valid JSON object with exactly this key:
        "score": <integer between 1 and 5>
        Do not include any additional text, reasoning, or explanations outside the JSON.

        Context: {context}
        Answer: {answer}
        """,
        "conciseness": """You are an expert evaluator for answer conciseness.
        Evaluate whether the answer is appropriately concise without unnecessary information.
        IMPORTANT: Output ONLY a valid JSON object with exactly this key:
        "score": <integer between 1 and 5>
        Do not include any additional text, reasoning, or explanations outside the JSON.

        Question: {question}
        Answer: {answer}
        """,
    }

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

        logger.info(f"LLM Judge initialized with model: {self.model}")

    @staticmethod
    def extract_json_from_text(text: str) -> str:  # noqa: C901, PLR0911, PLR0912, PLR0915
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
                        if isinstance(parsed, dict) and "score" in parsed:
                            return candidate
                    except json.JSONDecodeError:
                        continue

        if '"score"' in text:
            score_match = re.search(r'"score"\s*[:=]\s*(\d+)', text)
            if score_match:
                score = int(score_match.group(1))
                constructed = json.dumps({"score": score})
                logger.info(f"Constructed JSON from key-value pairs: {constructed}")
                return constructed

        score_match = re.search(r'(?<!["\'])score\s+(?:of\s+)?(\d)', text, re.IGNORECASE)
        if not score_match:
            score_match = re.search(r'(?<!["\'])score\s*[:=]\s*(\d)', text, re.IGNORECASE)
        if score_match:
            try:
                score = int(score_match.group(1))
                if 1 <= score <= LLMJudge.MAX_SCORE:
                    constructed = json.dumps({"score": score})
                    logger.info(f"Constructed JSON from plain text: {constructed}")
                    return constructed
            except Exception as e:
                logger.warning(f"Failed to construct JSON from plain text: {e}")

        return ""

    def _call_llm(self, prompt: str) -> dict[str, Any]:  # noqa: C901
        """Make API call to OpenRouter and parse JSON response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 4096,
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

            if "message" not in result["choices"][0]:
                logger.error(f"Invalid API response: missing 'message' in first choice: {result}")
                raise ValueError("Invalid API response: missing 'message' in first choice")

            message = result["choices"][0]["message"]
            content = message.get("content")

            if content is None:
                content = (
                    message.get("thinking")
                    or message.get("reasoning")
                    or message.get("reasoning_content")
                )
                if content:
                    finish_reason = result["choices"][0].get("finish_reason")
                    logger.info(
                        f"Using content from 'reasoning' field "
                        f"(content was None, finish_reason: {finish_reason})"
                    )
                else:
                    logger.warning(
                        f"LLM returned None content and no reasoning fallback."
                        f" Full response: {result}"
                    )
                    raise ValueError("LLM returned empty content (None) with no reasoning fallback")

            if not isinstance(content, str):
                logger.warning(
                    f"LLM returned non-string content type: {type(content)}, content: {content}"
                )
                raise ValueError(f"LLM returned non-string content: {type(content)}")

            if not content.strip():
                logger.warning(f"LLM returned empty/whitespace-only content: '{content}'")
                raise ValueError("LLM returned empty or whitespace-only content")

            try:
                json_str = LLMJudge.extract_json_from_text(content)
                if not json_str:
                    logger.warning(f"Failed to extract JSON from content: {content}")
                    raise ValueError("No JSON found in LLM response")
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse LLM response as JSON. "
                    f"Extracted: '{json_str}'. "
                    f"Original: {content}"
                )
                raise ValueError(f"Invalid JSON response from LLM: {e}") from e

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API call failed: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(
                f"Unexpected API response structure: "
                f"{result if 'result' in locals() else 'No result'}. Error: {e}"
            )
            raise ValueError(f"Invalid API response structure: {e}") from e

    def evaluate(
        self,
        metric: str,
        question: str,
        prediction: str,
        ground_truth: str | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        if metric not in self.PROMPTS:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(self.PROMPTS.keys())}")

        prompt_templates = {
            "answer_relevance": self.PROMPTS["answer_relevance"].format(
                question=question, answer=prediction
            ),
            "answer_correctness": self.PROMPTS["answer_correctness"].format(
                question=question, ground_truth=ground_truth, prediction=prediction
            ),
            "faithfulness": self.PROMPTS["faithfulness"].format(context=context, answer=prediction),
            "conciseness": self.PROMPTS["conciseness"].format(question=question, answer=prediction),
        }

        prompt = prompt_templates[metric]
        result = self._call_llm(prompt)

        if "score" not in result:
            raise ValueError(f"LLM response missing 'score' field: {result}")

        return {"score": result["score"]}


def evaluate_llm_metrics(  # noqa: C901, PLR0912
    predictions: list[str],
    questions: list[str],
    ground_truths: list[str] | None = None,
    contexts: list[str] | None = None,
    metrics: list[str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
) -> dict[str, float]:
    if metrics is None:
        metrics = ["answer_relevance", "answer_correctness", "faithfulness", "conciseness"]

    if not predictions or not questions:
        return {}

    valid_indices = []
    for i, (pred, ques) in enumerate(zip(predictions, questions, strict=False)):
        if pred is not None and ques is not None:
            valid_indices.append(i)

    if not valid_indices:
        logger.warning("No valid samples with non-None predictions and questions")
        return {}

    try:
        judge = LLMJudge(api_key=api_key, api_base=api_base, model=model)
    except ValueError as e:
        logger.warning(f"LLM Judge not available: {e}. Skipping LLM metrics.")
        return {}

    results = {metric: [] for metric in metrics}

    for i in valid_indices:
        question = questions[i]
        prediction = predictions[i]
        ground_truth = (
            ground_truths[i]
            if ground_truths and i < len(ground_truths) and ground_truths[i] is not None
            else None
        )
        context = (
            contexts[i] if contexts and i < len(contexts) and contexts[i] is not None else None
        )

        for metric in metrics:
            try:
                if metric == "answer_correctness" and ground_truth is None:
                    logger.warning(f"Skipping {metric} for sample {i}: ground_truth not provided")
                    continue
                if metric == "faithfulness" and context is None:
                    logger.warning(f"Skipping {metric} for sample {i}: context not provided")
                    continue

                result = judge.evaluate(
                    metric=metric,
                    question=question,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    context=context,
                )
                raw_score = result["score"]
                logger.info(f"{i}: metric {metric}, score: {raw_score}")
                try:
                    score = int(raw_score)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid score '{raw_score}' for metric {metric} on sample {i}: "
                        f"{e}. Skipping."
                    )
                    continue
                if not (1 <= score <= LLMJudge.MAX_SCORE):
                    logger.warning(
                        f"Score {score} out of range [1,{LLMJudge.MAX_SCORE}] for metric {metric} "
                        f"on sample {i}. Skipping."
                    )
                    continue
                results[metric].append(score / LLMJudge.MAX_SCORE)
            except Exception as e:
                logger.warning(f"Failed to evaluate {metric} for sample {i}: {e}")
                continue

    averages = {}
    for metric, scores in results.items():
        if scores:
            averages[metric] = sum(scores) / len(scores)
        else:
            averages[metric] = 0.0

    return averages
