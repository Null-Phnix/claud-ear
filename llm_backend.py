#!/usr/bin/env python3
"""
LLM Backend — configurable LLM API client for Claud-Ear.

Supports Ollama (default), OpenAI-compatible APIs, or any HTTP endpoint.
Configure via environment variables:

  AUDIO_LLM_MODEL   — model name (default: llama3.1:8b)
  AUDIO_LLM_HOST    — API host (default: http://localhost:11434)
  AUDIO_LLM_PROVIDER — "ollama" or "openai" (default: ollama)
"""

import json
import os
import sys
from typing import Optional

import requests


def get_model() -> str:
    return os.environ.get("AUDIO_LLM_MODEL", "llama3.1:8b")


def get_host() -> str:
    return os.environ.get("AUDIO_LLM_HOST", "http://localhost:11434")


def get_provider() -> str:
    return os.environ.get("AUDIO_LLM_PROVIDER", "ollama")


def call_ollama(prompt: str, system: Optional[str] = None,
                model: Optional[str] = None, timeout: int = 600) -> str:
    """Call Ollama /api/generate endpoint."""
    host = get_host()
    model = model or get_model()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    resp = requests.post(
        f"{host}/api/generate",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def call_openai(prompt: str, system: Optional[str] = None,
                model: Optional[str] = None, timeout: int = 600) -> str:
    """Call OpenAI-compatible /v1/chat/completions endpoint."""
    host = get_host()
    model = model or get_model()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{host}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.7,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_llm(prompt: str, system: Optional[str] = None,
             model: Optional[str] = None, timeout: int = 600) -> str:
    """Call the configured LLM and return response text."""
    provider = get_provider()
    if provider == "openai":
        return call_openai(prompt, system=system, model=model, timeout=timeout)
    return call_ollama(prompt, system=system, model=model, timeout=timeout)


def check_connection() -> bool:
    """Verify the LLM backend is reachable."""
    try:
        provider = get_provider()
        host = get_host()
        if provider == "openai":
            resp = requests.get(f"{host}/v1/models", timeout=5)
        else:
            resp = requests.get(f"{host}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    # Quick test
    if check_connection():
        print(f"✓ Connected to {get_provider()} at {get_host()}")
        print(f"✓ Model: {get_model()}")
        response = call_llm("Say 'Claud-Ear is online' in exactly three words.")
        print(f"✓ Response: {response}")
    else:
        print(f"✗ Cannot reach {get_provider()} at {get_host()}")
        sys.exit(1)
