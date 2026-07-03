"""Anthropic wrapper for Claude Sonnet / Opus / Haiku.

Contract: `chat(system, user, role_tag)` -> (text, LLMCall).
Reads ANTHROPIC_API_KEY from .env. Same key works across all Claude models.
"""
from __future__ import annotations

import time

from anthropic import Anthropic, APIError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from infoshield.config import get_env
from infoshield.trace import LLMCall

_RETRYABLE = (RateLimitError, APIError)


class AnthropicClient:
    """Thin wrapper around anthropic.messages.create."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.client = Anthropic(api_key=get_env("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def chat(
        self, system: str, user: str, role_tag: str = "verifier"
    ) -> tuple[str, LLMCall]:
        # Anthropic rejects empty user content with 400. Substitute a placeholder
        # so upstream callers that pass through empty LLM outputs don't crash the
        # whole pipeline; callers should already be handling empties, this is
        # defensive.
        if not (user or "").strip():
            user = "[empty]"
        started = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        text = "".join(
            getattr(block, "text", "") for block in resp.content
        )
        call = LLMCall(
            role=role_tag,
            provider="anthropic",
            model=self.model,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
        )
        return text, call
