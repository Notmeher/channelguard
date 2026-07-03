"""LLM wrappers: retry, seed, usage capture. Return (text, LLMCall) uniformly."""
from __future__ import annotations

from typing import Any, Protocol

from infoshield.llm.anthropic import AnthropicClient
from infoshield.llm.azure_embedding import AzureEmbeddingClient
from infoshield.llm.azure_openai import AzureOpenAIClient

__all__ = [
    "AzureOpenAIClient",
    "AnthropicClient",
    "AzureEmbeddingClient",
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
    `build_pipeline` so any agent role can be swapped between gpt-5 (Azure) and
    Claude (Anthropic) purely via config.
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
    raise ValueError(f"Unknown chat provider: {provider!r}")
