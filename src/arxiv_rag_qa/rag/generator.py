import os
from http import HTTPStatus
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QwenGenerator:
    """Local LLM generator using Hugging Face transformers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = True,
    ):
        self.model_name = model_name
        self.device = device

        if load_in_4bit and "cuda" in device:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                trust_remote_code=True,
            )

        self.model.eval()

    def generate(self, query: str, context: str, max_new_tokens: int = 256) -> tuple[str, int]:
        """Generate answer and return (text, num_generated_tokens)."""
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful scientific assistant. "
                "Answer based ONLY on the provided context.",
            },
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        num_tokens = outputs.shape[1] - input_len
        return response.strip(), num_tokens


class OpenRouterGenerator:
    """LLM generator using OpenRouter API (compatible with OpenAI chat API)."""

    def __init__(
        self,
        model_name: str = "qwen/qwen2.5-0.5b-instruct",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY env var "
                "or pass api_key in config."
            )

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int | None = None,
    ) -> tuple[str, int]:
        """Generate answer and return (text, num_generated_tokens)."""
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful scientific assistant. "
                "Answer based ONLY on the provided context.",
            },
            {"role": "user", "content": prompt},
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_new_tokens or self.max_tokens,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"OpenRouter API error (status {response.status_code}): {response.text}"
            )

        result = response.json()
        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter API returned no choices: {result}")

        text = choices[0]["message"]["content"].strip()
        usage = result.get("usage", {})
        num_tokens = usage.get("completion_tokens", 0)
        return text, num_tokens


def create_generator(config: dict[str, Any]) -> QwenGenerator | OpenRouterGenerator:
    """Create a generator instance based on configuration.

    Config structure:
        generator:
            type: "local"            # "local" or "openrouter"
            model_name: "Qwen/Qwen2.5-0.5B-Instruct"
            max_new_tokens: 256
            local:
                device: "cuda"
                load_in_4bit: true
            openrouter:
                api_key: "${oc.env:OPENROUTER_API_KEY}"
                base_url: "https://openrouter.ai/api/v1"
                max_tokens: 256
                temperature: 0.0
                timeout: 120
    """
    gen_type = config.get("type", "local")

    if gen_type == "local":
        local_config = config.get("local", {})
        return QwenGenerator(
            model_name=config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"),
            device=local_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            load_in_4bit=local_config.get("load_in_4bit", True),
        )

    if gen_type == "openrouter":
        or_config = config.get("openrouter", {})
        return OpenRouterGenerator(
            model_name=config.get("model_name", "qwen/qwen2.5-0.5b-instruct"),
            api_key=or_config.get("api_key"),
            base_url=or_config.get("base_url", "https://openrouter.ai/api/v1"),
            max_tokens=or_config.get("max_tokens", 256),
            temperature=or_config.get("temperature", 0.0),
            timeout=or_config.get("timeout", 120),
        )

    raise ValueError(f"Unknown generator type: {gen_type}. Expected 'local' or 'openrouter'.")
