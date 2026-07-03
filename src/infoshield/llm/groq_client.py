"""Groq wrapper for the F1 permissive-filter experiment.

Groq serves open-model chat completions (Llama-3.x, Qwen, DeepSeek, etc.) via
an OpenAI-compatible endpoint. Unlike Azure OpenAI, Groq applies NO server-side
content filter beyond the model's own RLHF safety training. This makes it the
ideal counterfactual for the mechanism-substitution claim in §7.2 of the paper:
if `undefended` on Groq leaks attacks that Azure would have caught, we
empirically validate that provider-filter drift is a real deployment risk.

Contract: `chat(system, user, role_tag)` -> (text, LLMCall).

Env var: `groq_api_key` (matches the user's .env casing).

Notes on Llama-3.x safety training:
  * The model refuses obvious jailbreaks in its instruction-following, but
    lacks Azure's separate 400 content-filter layer.
  * When a refusal is generated, it comes back as normal completion text
    (e.g. "I can't help with that."), NOT as a 400 error. We do NOT emit
    the [AZURE_CONTENT_FILTER_REFUSED] sentinel — the attribution classifier
    will correctly count these as `answered_safely` if the marker never
    appears, or as `leaked` if the model complied with the injection.
"""
from __future__ import annotations

import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from infoshield.config import get_env
from infoshield.trace import LLMCall

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqClient:
    """Thin OpenAI-compatible client pointed at Groq.

    Default model is `llama-3.3-70b-versatile` — Meta's flagship open-model
    with weaker safety training than Azure's gpt-5 wrapping. Good for
    demonstrating provider-filter drift; still capable enough that multi-agent
    pipeline reasoning does not collapse.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        api_key = get_env("groq_api_key")
        self.client = OpenAI(api_key=api_key, base_url=_GROQ_BASE_URL)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def chat(self, system: str, user: str, role_tag: str = "agent") -> tuple[str, LLMCall]:
        started = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except BadRequestError as e:
            # Groq returns 400 for genuinely malformed requests, not content
            # filtering. Surface as-is so we can debug rather than mask it.
            latency_ms = (time.perf_counter() - started) * 1000.0
            return f"[GROQ_BAD_REQUEST: {e}]", LLMCall(
                role=role_tag, provider="groq", model=self.model,
                prompt_tokens=0, completion_tokens=0, latency_ms=latency_ms,
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, LLMCall(
            role=role_tag,
            provider="groq",
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
        )
