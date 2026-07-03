"""Azure OpenAI wrapper for gpt-5 (or any Azure chat deployment).

Contract: `chat(system, user, role_tag)` -> (text, LLMCall).
Reads AZURE_OPENAI_{API_KEY, API_VERSION, ENDPOINT} from .env.

Note on reasoning models (gpt-5 / o-series): completion tokens include hidden
reasoning tokens that count toward the budget but are not returned as text.
Use max_tokens >= 256 even for short answers, or the visible reply may be empty.

Content-filter handling: Azure sometimes 400s the request with
`code: content_filter` (input jailbroken) or filters the response the same way.
We convert both into a sentinel string `[AZURE_CONTENT_FILTER_REFUSED]` and log
zero tokens, so the pipeline records the event and moves on.

Reasoning-model quirk: gpt-5 and o-series deployments reject `temperature`,
`seed`, and `max_completion_tokens` in various combinations. Rather than pay
for a failing round-trip on every call, we cache the offending params per
deployment the first time we see them rejected and strip them from subsequent
requests.
"""
from __future__ import annotations

import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    AzureOpenAI,
    BadRequestError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from infoshield.config import get_env
from infoshield.trace import LLMCall

# Only truly transient failures are retried by tenacity. Content-filter refusals
# (400) are handled explicitly inside `chat()`, never retried.
_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)

# Sentinel returned to the pipeline when Azure content-filters the request.
CONTENT_FILTER_SENTINEL = "[AZURE_CONTENT_FILTER_REFUSED]"


def _is_content_filter_error(e: BadRequestError) -> bool:
    """True iff this 400 is an Azure content-management/jailbreak refusal."""
    # Inspect the structured .body first (most reliable), then fall back to string.
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        code = str(body.get("code") or "").lower()
        if code == "content_filter":
            return True
        inner = body.get("innererror") if isinstance(body, dict) else None
        if isinstance(inner, dict):
            inner_code = str(inner.get("code") or "").lower()
            if inner_code in ("responsibleaipolicyviolation", "content_filter"):
                return True
    # Fallback: substring match on str(e).lower().
    msg = str(e).lower()
    return "content_filter" in msg or "responsibleaipolicyviolation" in msg


class AzureOpenAIClient:
    """Thin wrapper around AzureOpenAI.chat.completions.

    Handles two Azure-specific quirks:
      1. Content-filter 400s -> returns `[AZURE_CONTENT_FILTER_REFUSED]`.
      2. Reasoning-model param rejection (temperature/seed/max_completion_tokens)
         -> auto-strips the rejected param, caches the fact per deployment so
         subsequent calls skip the wasted round-trip, and retries in-place.
    """

    # Class-level cache: deployment -> set of param names that this deployment
    # has been observed to reject. Populated on the first rejection, then all
    # future calls skip that param up front.
    _unsupported_params: dict[str, set[str]] = {}

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

    def _build_kwargs(self, system: str, user: str) -> dict:
        """Build the request kwargs, pre-stripping any params known-unsupported."""
        kwargs: dict = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        # Strip params this deployment has previously rejected.
        for bad in self._unsupported_params.get(self.deployment, set()):
            if bad == "max_completion_tokens" and "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            else:
                kwargs.pop(bad, None)
        return kwargs

    def _mark_unsupported(self, param: str) -> None:
        self._unsupported_params.setdefault(self.deployment, set()).add(param)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def chat(self, system: str, user: str, role_tag: str = "agent") -> tuple[str, LLMCall]:
        started = time.perf_counter()
        kwargs = self._build_kwargs(system, user)

        # At most 4 attempts. Each attempt either succeeds, returns a
        # content-filter refusal, or strips one more unsupported param.
        last_error: BadRequestError | None = None
        for _attempt in range(4):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except BadRequestError as e:
                last_error = e
                # Content filter — refuse gracefully, do not retry.
                if _is_content_filter_error(e):
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    return CONTENT_FILTER_SENTINEL, LLMCall(
                        role=role_tag,
                        provider="azure_openai",
                        model=self.deployment,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=latency_ms,
                    )
                # Try to identify and strip a rejected param.
                msg = str(e).lower()
                stripped = False
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature", None)
                    self._mark_unsupported("temperature")
                    stripped = True
                if "seed" in msg and "seed" in kwargs:
                    kwargs.pop("seed", None)
                    self._mark_unsupported("seed")
                    stripped = True
                if "max_completion_tokens" in msg and "max_completion_tokens" in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    self._mark_unsupported("max_completion_tokens")
                    stripped = True
                if not stripped:
                    # Unknown 400 — surface it rather than loop forever.
                    raise
                # Loop and retry with reduced kwargs.
        else:
            # Ran out of attempts without success.
            assert last_error is not None
            raise last_error

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
