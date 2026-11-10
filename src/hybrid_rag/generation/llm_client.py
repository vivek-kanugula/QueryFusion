"""Unified client for multi-provider LLM generation and verification.

Supports:
- Google Gemini (gemini-3.5-flash-lite, gemini-3.8-flash, gemini-2.5-flash, gemini-3.1-pro-preview)
- Groq (llama-3.3-70b-versatile, llama-3.1-8b-instant)
- Local Ollama (ollama/llama3.2, etc.)
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from hybrid_rag.settings import settings


def resolve_provider(model: str | None, explicit_provider: str | None = None) -> tuple[str, str]:
    """Resolve (provider, clean_model_name)."""
    model_name = (model or settings.gemini_model).strip()

    if explicit_provider:
        provider = explicit_provider.lower().strip()
    elif model_name.startswith("groq/") or model_name.startswith("llama-"):
        provider = "groq"
    elif model_name.startswith("ollama/"):
        provider = "ollama"
    else:
        provider = "gemini"

    clean_model = model_name
    if clean_model.startswith("groq/"):
        clean_model = clean_model[len("groq/") :]
    elif clean_model.startswith("ollama/"):
        clean_model = clean_model[len("ollama/") :]

    return provider, clean_model


def call_llm(
    prompt: str,
    system_instruction: str,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
) -> str:
    """Execute LLM call across configured provider."""
    resolved_provider, resolved_model = resolve_provider(model, provider)

    if resolved_provider == "groq":
        return _call_groq(resolved_model, prompt, system_instruction, temperature)
    elif resolved_provider == "ollama":
        return _call_ollama(resolved_model, prompt, system_instruction, temperature)
    else:
        return _call_gemini(resolved_model, prompt, system_instruction, temperature)


def _call_gemini(
    model_name: str,
    prompt: str,
    system_instruction: str,
    temperature: float,
) -> str:
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": temperature,
            "system_instruction": system_instruction,
        },
    )
    return response.text or ""


def _call_groq(
    model_name: str,
    prompt: str,
    system_instruction: str,
    temperature: float,
) -> str:
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured in settings or environment.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or not isinstance(choices[0], dict):
            return ""
        message = choices[0].get("message")
        if not message or not isinstance(message, dict):
            return ""
        content = message.get("content")
        return str(content) if content is not None else ""


def _call_ollama(
    model_name: str,
    prompt: str,
    system_instruction: str,
    temperature: float,
) -> str:
    host = (settings.ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    url = f"{host}/api/generate"
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system_instruction:
        payload["system"] = system_instruction
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return str(data["response"])
