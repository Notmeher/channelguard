"""Ollama wrapper for local Llama-3.1-8B (or any Ollama-served model).

Contract: `chat(system, user, role_tag)` -> (text, LLMCall).
Talks to `http://localhost:11434/api/chat` by default; override with the
`OLLAMA_URL` env var. Used by the weak-model appendix run.
"""
from __future__ import annotations

import os
import time

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from infoshield.trace import LLMCall

# Only transient network failures are retried. HTTP 4xx must surface.
_RETRYABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout)


class OllamaClient:
    """Thin wrapper around Ollama's /api/chat endpoint."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        url: str | None = None,
        timeout_s: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout_s = timeout_s

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def chat(self, system: str, user: str, role_tag: str = "agent") -> tuple[str, LLMCall]:
        # Ollama also rejects empty content; substitute a placeholder defensively.
        if not (user or "").strip():
            user = "[empty]"
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        latency_ms = (time.perf_counter() - started) * 1000.0
        text = (data.get("message") or {}).get("content", "") or ""
        call = LLMCall(
            role=role_tag,
            provider="ollama",
            model=self.model,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency_ms=latency_ms,
        )
        return text, call
