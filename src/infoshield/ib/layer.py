"""IB gate layer.

Contract: `IBLayer.gate(text)` -> (risk: float, status: str, output: str).

Risk score: max cosine similarity between the sentence embedding of `text`
and any pre-encoded adversarial phrase in the bank. Scored per-sentence so
a poisoned sentence in a benign paragraph still triggers the gate.

Status:
    BLOCKED    if risk >= threshold                 -> output = ""
    COMPRESSED if compress_ratio*threshold <= risk  -> first N sentences kept
    PASSED     otherwise                            -> original text

Encoder: any object with .encode(list[str], normalize_embeddings=True) -> ndarray.
Both `sentence_transformers.SentenceTransformer` and
`infoshield.llm.AzureEmbeddingClient` satisfy this.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from infoshield.ib.phrase_bank import get_bank


@lru_cache(maxsize=4)
def _load_local_encoder(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


class IBLayer:
    """Embedding-similarity gate over a fixed adversarial phrase bank."""

    def __init__(
        self,
        threshold: float = 0.60,
        compress_ratio: float = 0.70,
        compress_sentences: int = 2,
        phrase_bank_size: int = 20,
        encoder: Any = None,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        # Encoder resolution:
        #   1. explicit `encoder` (used by build_ib_gates when config picks azure)
        #   2. otherwise, local sentence-transformer at `embed_model`
        if encoder is None:
            encoder = _load_local_encoder(embed_model, device)
        self.encoder = encoder
        self.threshold = threshold
        self.compress_ratio = compress_ratio
        self.compress_sentences = compress_sentences
        phrases = get_bank(phrase_bank_size)
        self.phrases = phrases
        # Normalize embeddings so cosine == dot product (faster + numerically stable).
        self.bank_emb = self.encoder.encode(
            phrases, convert_to_numpy=True, normalize_embeddings=True
        )

    # --- Core methods ---------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip())]
        return [s for s in parts if s]

    def score(self, text: str) -> float:
        """Max cosine similarity between any SENTENCE in text and any bank phrase.

        Per-sentence scoring prevents dilution: a single poisoned sentence
        embedded in a benign paragraph still triggers the gate.
        """
        text = (text or "").strip()
        if not text:
            return 0.0
        sentences = self._split_sentences(text) or [text]
        v = self.encoder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(v, self.bank_emb)  # shape (n_sent, |bank|)
        return float(sims.max())

    def compress(self, text: str) -> str:
        # Simple extractive: keep first N sentences. Adversarial suffixes are
        # typically appended, so truncation strips them.
        parts = self._split_sentences(text)
        return " ".join(parts[: max(1, self.compress_sentences)]).strip()

    def gate(self, text: str) -> tuple[float, str, str]:
        risk = self.score(text)
        if risk >= self.threshold:
            return risk, "BLOCKED", ""
        if risk >= self.compress_ratio * self.threshold:
            return risk, "COMPRESSED", self.compress(text)
        return risk, "PASSED", text
