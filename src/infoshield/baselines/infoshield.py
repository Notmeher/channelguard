"""INFOSHIELD system variants.

Two encoders are registered as separate systems so runners can compare them
side-by-side on identical inputs:

  infoshield         : local sentence-transformer (MiniLM by default). Free,
                       offline, fast.
  infoshield_azure   : Azure OpenAI text-embedding-3-large (or whichever
                       AZURE_OPENAI_EMBED_DEPLOYMENT is configured). Higher
                       quality, network dependency, per-call cost.
"""
from __future__ import annotations

import copy

from infoshield.baselines.registry import register
from infoshield.config import Config
from infoshield.ib.gates import build_ib_gates
from infoshield.pipeline.langgraph_pipeline import PipelineFn, build_pipeline


@register("infoshield")
def build_local(cfg: Config) -> PipelineFn:
    return build_pipeline(cfg, ib_gates=build_ib_gates(cfg))


@register("infoshield_azure")
def build_azure(cfg: Config) -> PipelineFn:
    """Same pipeline; only the encoder differs."""
    cfg_azure = copy.deepcopy(cfg)
    cfg_azure.embedding.provider = "azure"
    if not cfg_azure.embedding.deployment:
        # Sensible default from user's .env (AZURE_OPENAI_EMBED_DEPLOYMENT).
        cfg_azure.embedding.deployment = "text-embedding-3-large"
    if not cfg_azure.embedding.dimensions:
        cfg_azure.embedding.dimensions = 1536
    return build_pipeline(cfg, ib_gates=build_ib_gates(cfg_azure))
