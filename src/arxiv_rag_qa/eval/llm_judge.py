import json
import re
from typing import Any, ClassVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)


class LLMJudge:
    """LLM Judge for evaluating RAG outputs using a local model."""

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
        "faithfulness": """You are an expert evaluator for faithfulness/grounding.
        Verify whether all claims in the answer are supported by the provided context.
        IMPORTANT: Output ONLY a valid JSON object with exactly this key:
        "score": <integer between 1 and 5>
        Do not include any additional text, reasoning, or explanations outside the JSON.

        Context: {context}
        Answer: {answer}
        """,
        "context_relevance": """You are an expert evaluator for context relevance in
        question-answering systems.
        Evaluate how relevant and complete the retrieved context is for answering the question.
        Consider both whether the context contains the information needed to answer the question
        and whether it provides sufficient detail and completeness.

        IMPORTANT: Output ONLY a valid JSON object with exactly this key:
        "score": <integer between 1 and 5>
        Do not include any additional text, reasoning, or explanations outside the JSON.

        Question: {question}
        Context: {context}
        """,
    }

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.model:
            raise ValueError("Model not configured. Set model parameter or pass model name.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model: {self.model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.model_obj = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            logger.info(f"LLM Judge initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model}: {e}")
            raise

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

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Generate response using local model."""
        try:
            formatted_prompt = (
                "System: You are a helpful assistant that evaluates responses.\n\n"
                f"Human: {prompt}\n\nAssistant:"
            )

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model_obj.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Assistant:" in response_text:
                response_text = response_text.split("Assistant:")[1].strip()

            if not response_text:
                logger.warning("LLM returned empty response")
                raise ValueError("LLM returned empty response")

            json_str = self.extract_json_from_text(response_text)
            if not json_str:
                logger.warning(f"Failed to extract JSON from content: {response_text}")
                raise ValueError("No JSON found in LLM response")

            return json.loads(json_str)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

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
            "faithfulness": self.PROMPTS["faithfulness"].format(context=context, answer=prediction),
            "context_relevance": self.PROMPTS["context_relevance"].format(
                question=question, context=context
            ),
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
    model: str | None = None,
) -> dict[str, float]:
    if metrics is None:
        metrics = ["answer_relevance", "faithfulness", "context_relevance"]

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
        judge = LLMJudge(model=model)
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
