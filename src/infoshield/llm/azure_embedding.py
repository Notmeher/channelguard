"""Azure OpenAI embedding client.

Fallback for the IB layer when the local sentence-transformer is unavailable.
Uses text-embedding-3-large (or any Azure embedding deployment). Batches
inputs and returns L2-normalized vectors so cosine similarity == dot product,
matching the sentence-transformers path.
"""
from __future__ import annotations

import numpy as np
from openai import APIConnectionError, APITimeoutError, AzureOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from infoshield.config import get_env

# Only truly transient failures are retried. BadRequestError (400) must surface.
_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


class AzureEmbeddingClient:
    """Batch encoder using an Azure OpenAI embedding deployment."""

    def __init__(self, deployment: str, dimensions: int | None = None, batch_size: int = 64):
        self.client = AzureOpenAI(
            api_key=get_env("AZURE_OPENAI_EMBED_API_KEY"),
            api_version=get_env("AZURE_OPENAI_EMBED_API_VERSION"),
            azure_endpoint=get_env("AZURE_OPENAI_EMBED_ENDPOINT"),
        )
        self.deployment = deployment
        self.dimensions = dimensions
        self.batch_size = batch_size

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def _one_batch(self, batch: list[str]) -> np.ndarray:
        kwargs: dict = {"model": self.deployment, "input": batch}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self.client.embeddings.create(**kwargs)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # L2-normalize so cosine == dot product downstream.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,   # accepted for signature compat with sentence-transformers
        normalize_embeddings: bool = True,  # accepted for signature compat; output is ALWAYS L2-normalized
    ) -> np.ndarray:
        # Note: `normalize_embeddings` is accepted for API compat with
        # sentence-transformers but is effectively ignored — vectors are always
        # L2-normalized so downstream cosine similarity reduces to a dot product.
        if not texts:
            return np.zeros((0, self.dimensions or 1), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            chunks.append(self._one_batch(texts[i : i + self.batch_size]))
        return np.vstack(chunks)
