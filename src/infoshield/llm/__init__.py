"""LLM wrappers: retry, seed, usage capture. Return (text, LLMCall) uniformly."""
from __future__ import annotations

from typing import Any, Protocol

from infoshield.llm.anthropic import AnthropicClient
from infoshield.llm.azure_embedding import AzureEmbeddingClient
from infoshield.llm.azure_openai import AzureOpenAIClient
from infoshield.llm.groq_client import GroqClient

__all__ = [
    "AzureOpenAIClient",
    "AnthropicClient",
    "AzureEmbeddingClient",
    "GroqClient",
    "ChatClient",
    "build_chat_client",
]


class ChatClient(Protocol):
    """Uniform chat-client interface implemented by every provider wrapper."""

    def chat(self, system: str, user: str, role_tag: str = "agent") -> tuple[str, Any]:
        ...


def build_chat_client(model_cfg) -> ChatClient:
    """Dispatch to the correct provider client based on `model_cfg.provider`.

    `model_cfg` is a `ModelCfg` (or subclass) from `infoshield.config`. Used by
    `build_pipeline` so any agent role can be swapped between gpt-5 (Azure),
    Claude (Anthropic), and Llama-3.x / Qwen / DeepSeek (Groq) purely via config.
    Groq is used for the F1 permissive-filter experiment (§6.16) — it serves
    the same OpenAI-compatible chat API without Azure's server-side content
    filter, so `undefended` runs on Groq measure what happens when the
    provider filter is not there to catch attacks.
    """
    provider = model_cfg.provider
    if provider == "azure_openai":
        if not model_cfg.deployment:
            raise ValueError("azure_openai provider requires `deployment`")
        return AzureOpenAIClient(
            deployment=model_cfg.deployment,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
        )
    if provider == "anthropic":
        if not model_cfg.model:
            raise ValueError("anthropic provider requires `model`")
        return AnthropicClient(
            model=model_cfg.model,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
        )
    if provider == "groq":
        if not model_cfg.model:
            raise ValueError("groq provider requires `model` (e.g. llama-3.3-70b-versatile)")
        return GroqClient(
            model=model_cfg.model,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
        )
    raise ValueError(f"Unknown chat provider: {provider!r}")
