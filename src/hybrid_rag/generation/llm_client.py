"""Unified client for multi-provider LLM generation and verification.

Supports:
- Google Gemini (gemini-3.5-flash-lite, gemini-3.8-flash, gemini-2.5-flash, gemini-3.1-pro-preview)
- Groq (llama-3.3-70b-versatile, llama-3.1-8b-instant)
- Local Ollama (ollama/llama3.2, etc.)
- Multi-key rotation and auxiliary offloading.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
from typing import Any

from hybrid_rag.settings import settings

logger = logging.getLogger(__name__)

_gemini_key_lock = threading.Lock()
_gemini_key_idx = 0


def get_gemini_api_keys() -> list[str]:
    """Retrieve all available Gemini API keys configured in settings or environment."""
    keys = list(settings.gemini_api_keys)
    if not keys and settings.gemini_api_key:
        keys = [settings.gemini_api_key]
    return [k.strip() for k in keys if k and k.strip()]


def get_active_gemini_key(explicit_key: str | None = None) -> str | None:
    """Get currently active Gemini API key from pool or explicit override."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    keys = get_gemini_api_keys()
    if not keys:
        return settings.gemini_api_key
    global _gemini_key_idx
    with _gemini_key_lock:
        return keys[_gemini_key_idx % len(keys)]


def rotate_to_next_gemini_key() -> str | None:
    """Advance to the next Gemini API key in the configured pool."""
    keys = get_gemini_api_keys()
    if not keys:
        return None
    global _gemini_key_idx
    with _gemini_key_lock:
        _gemini_key_idx = (_gemini_key_idx + 1) % len(keys)
        new_key = keys[_gemini_key_idx % len(keys)]
        logger.info("Rotated to Gemini API key index %d/%d", _gemini_key_idx, len(keys))
        return new_key


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
    api_key: str | None = None,
) -> str:
    """Execute LLM call across configured provider."""
    resolved_provider, resolved_model = resolve_provider(model, provider)

    if resolved_provider == "groq":
        return _call_groq(resolved_model, prompt, system_instruction, temperature, api_key=api_key)
    elif resolved_provider == "ollama":
        return _call_ollama(resolved_model, prompt, system_instruction, temperature)
    else:
        return _call_gemini(resolved_model, prompt, system_instruction, temperature, api_key=api_key)


def get_auxiliary_llm_config() -> tuple[str, str]:
    """Get recommended (provider, model) for auxiliary tasks (HyDE, Contextualize, Verify)."""
    groq_available = bool(settings.groq_api_key or os.getenv("GROQ_API_KEY"))
    if settings.auxiliary_llm_provider in ("auto", "groq") and groq_available:
        return "groq", settings.groq_fast_model
    return "gemini", settings.gemini_verifier_model or settings.gemini_model


def call_auxiliary_llm(
    prompt: str,
    system_instruction: str,
    temperature: float = 0.0,
    model: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    fallback_to_gemini: bool = True,
) -> str:
    """Execute auxiliary LLM call with preference for Groq and graceful fallback to Gemini."""
    if provider is None:
        rec_provider, rec_model = get_auxiliary_llm_config()
        chosen_provider = rec_provider
        chosen_model = model or rec_model
    else:
        chosen_provider = provider
        chosen_model = model

    try:
        return call_llm(
            prompt=prompt,
            system_instruction=system_instruction,
            model=chosen_model,
            provider=chosen_provider,
            temperature=temperature,
            api_key=api_key,
        )
    except Exception as exc:
        if fallback_to_gemini and chosen_provider != "gemini":
            logger.warning(
                "Auxiliary LLM call on provider %s failed (%s); falling back to Gemini",
                chosen_provider,
                exc,
            )
            return call_llm(
                prompt=prompt,
                system_instruction=system_instruction,
                model=settings.gemini_model,
                provider="gemini",
                temperature=temperature,
                api_key=api_key,
            )
        raise


def _call_gemini(
    model_name: str,
    prompt: str,
    system_instruction: str,
    temperature: float,
    api_key: str | None = None,
) -> str:
    from google import genai

    keys = [api_key.strip()] if api_key and api_key.strip() else get_gemini_api_keys()
    if not keys:
        keys = [settings.gemini_api_key] if settings.gemini_api_key else [None]

    last_error: Exception | None = None
    for attempt in range(len(keys)):
        active_key = keys[(attempt + _gemini_key_idx) % len(keys)]
        try:
            client = genai.Client(api_key=active_key)
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
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            is_quota_error = (
                (status_code in (429, 503))
                or ("429" in err_str)
                or ("RESOURCE_EXHAUSTED" in err_str)
            )
            if is_quota_error and len(keys) > 1 and attempt < len(keys) - 1:
                rotate_to_next_gemini_key()
                continue
            raise

    if last_error:
        raise last_error
    return ""


def _call_groq(
    model_name: str,
    prompt: str,
    system_instruction: str,
    temperature: float,
    api_key: str | None = None,
) -> str:
    resolved_api_key = api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY")
    if not resolved_api_key:
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
            "Authorization": f"Bearer {resolved_api_key}",
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
