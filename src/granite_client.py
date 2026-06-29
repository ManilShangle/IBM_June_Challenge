"""Thin abstraction over IBM Granite inference so the backend is swappable
via .env without touching predictor.py. Supports watsonx.ai (primary),
Hugging Face Inference API (fallback), and local Ollama (offline fallback).
"""
import time

import requests

from src.config import (
    GRANITE_BACKEND,
    HF_API_TOKEN,
    HF_MODEL_ID,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    REPLICATE_API_TOKEN,
    REPLICATE_MODEL,
    WATSONX_API_KEY,
    WATSONX_MODEL_ID,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
)


class GraniteClientError(RuntimeError):
    pass


def generate(prompt: str, temperature: float = 0.2, max_new_tokens: int = 600) -> str:
    backend = GRANITE_BACKEND.lower()
    if backend == "replicate":
        return _generate_replicate(prompt, temperature, max_new_tokens)
    if backend == "watsonx":
        return _generate_watsonx(prompt, temperature, max_new_tokens)
    if backend == "huggingface":
        return _generate_huggingface(prompt, temperature, max_new_tokens)
    if backend == "ollama":
        return _generate_ollama(prompt, temperature, max_new_tokens)
    raise GraniteClientError(f"Unknown GRANITE_BACKEND: {backend!r}")


def _generate_replicate(prompt: str, temperature: float, max_new_tokens: int) -> str:
    if not REPLICATE_API_TOKEN:
        raise GraniteClientError(
            "REPLICATE_API_TOKEN missing. Create a free Replicate account "
            "(GitHub login), grab a token at replicate.com/account/api-tokens, "
            "and claim free credit at https://replicate.fyi/ibm. Set it in .env."
        )
    import replicate

    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    max_retries = 4
    for attempt in range(max_retries):
        try:
            output = client.run(
                REPLICATE_MODEL,
                input={
                    "prompt": prompt,
                    "temperature": max(temperature, 0.01),
                    "max_new_tokens": max_new_tokens,
                },
            )
            break
        except replicate.exceptions.ReplicateError as exc:
            is_rate_limited = getattr(exc, "status", None) == 429
            if is_rate_limited and attempt < max_retries - 1:
                time.sleep(12)
                continue
            raise GraniteClientError(f"Replicate request failed: {exc}") from exc
    if isinstance(output, list):
        return "".join(str(chunk) for chunk in output)
    return str(output)


def _generate_watsonx(prompt: str, temperature: float, max_new_tokens: int) -> str:
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        raise GraniteClientError(
            "WATSONX_API_KEY / WATSONX_PROJECT_ID missing. Set them in .env, "
            "or switch GRANITE_BACKEND to 'huggingface' or 'ollama'."
        )
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    model = ModelInference(
        model_id=WATSONX_MODEL_ID,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
    )
    params = {
        "decoding_method": "greedy" if temperature <= 0 else "sample",
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }
    result = model.generate_text(prompt=prompt, params=params)
    return result


def _generate_huggingface(prompt: str, temperature: float, max_new_tokens: int) -> str:
    if not HF_API_TOKEN:
        raise GraniteClientError(
            "HF_API_TOKEN missing. Set it in .env, or switch GRANITE_BACKEND."
        )
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}",
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={
            "inputs": prompt,
            "parameters": {
                "temperature": max(temperature, 0.01),
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    raise GraniteClientError(f"Unexpected Hugging Face response shape: {data!r}")


def _generate_ollama(prompt: str, temperature: float, max_new_tokens: int) -> str:
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_new_tokens},
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")
