"""Azure OpenAI wrapper for gpt-5 (or any Azure chat deployment).

Contract: `chat(system, user, role_tag)` -> (text, LLMCall).
Reads AZURE_OPENAI_{API_KEY, API_VERSION, ENDPOINT} from .env.

Note on reasoning models (gpt-5 / o-series): completion tokens include hidden
reasoning tokens that count toward the budget but are not returned as text.
Use max_tokens >= 256 even for short answers, or the visible reply may be empty.
"""
from __future__ import annotations

import time

from openai import APIError, AzureOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from infoshield.config import get_env
from infoshield.trace import LLMCall

_RETRYABLE = (RateLimitError, APIError)


class AzureOpenAIClient:
    """Thin wrapper around AzureOpenAI.chat.completions."""

    def __init__(
        self,
        deployment: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 0,
    ):
        self.client = AzureOpenAI(
            api_key=get_env("AZURE_OPENAI_API_KEY"),
            api_version=get_env("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
        )
        self.deployment = deployment
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def chat(self, system: str, user: str, role_tag: str = "agent") -> tuple[str, LLMCall]:
        started = time.perf_counter()
        kwargs: dict = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_tokens,
        }
        # gpt-5 / reasoning models may reject temperature/seed — try but tolerate.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except APIError as e:
            msg = str(e).lower()
            # Azure content filter refuses (400 + jailbreak.detected). Treat as
            # an upstream refusal so the pipeline records the event and moves on
            # instead of exploding the whole run.
            if "content_filter" in msg or "responsibleaipolicyviolation" in msg:
                latency_ms = (time.perf_counter() - started) * 1000.0
                return "[AZURE_CONTENT_FILTER_REFUSED]", LLMCall(
                    role=role_tag,
                    provider="azure_openai",
                    model=self.deployment,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency_ms,
                )
            # Strip params the model rejects and retry once, in-place.
            if "temperature" in msg:
                kwargs.pop("temperature", None)
            if "seed" in msg:
                kwargs.pop("seed", None)
            if "max_completion_tokens" in msg:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            resp = self.client.chat.completions.create(**kwargs)

        latency_ms = (time.perf_counter() - started) * 1000.0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        call = LLMCall(
            role=role_tag,
            provider="azure_openai",
            model=self.deployment,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
        )
        return text, call
