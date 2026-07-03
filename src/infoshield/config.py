"""Load and validate config.yaml + .env. Single source of truth for every module."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


# --- Model configs ------------------------------------------------------------


class ModelCfg(BaseModel):
    provider: Literal["azure_openai", "anthropic", "groq"]
    deployment: str | None = None   # Azure only
    model: str | None = None        # Anthropic + Groq
    temperature: float = 0.0
    max_tokens: int = 1024


class VerifierModelCfg(ModelCfg):
    samples: int = 5


class ModelsCfg(BaseModel):
    agent: ModelCfg
    verifier: VerifierModelCfg
    judge_hallucination: ModelCfg
    judge_consistency: ModelCfg


# --- Component configs --------------------------------------------------------


class EmbeddingCfg(BaseModel):
    """Local sentence-transformer OR Azure text-embedding-3-large.

    provider="local": use sentence-transformers with `model` (default MiniLM).
    provider="azure": use Azure OpenAI embeddings with `deployment` +
        optional `dimensions`.
    """

    provider: Literal["local", "azure"] = "local"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    deployment: str | None = None       # e.g. "text-embedding-3-large"
    dimensions: int | None = None       # e.g. 1536 for text-embedding-3-large
    device: str = "cpu"
    batch_size: int = 32


class IBCfg(BaseModel):
    beta: float
    threshold: float = Field(gt=0.0, lt=1.0)
    compress_ratio: float = Field(gt=0.0, le=1.0)
    compress_sentences: int = Field(gt=0)
    phrase_bank_size: int = Field(gt=0)
    active_layers: list[int]

    @field_validator("active_layers")
    @classmethod
    def _validate_layers(cls, v: list[int]) -> list[int]:
        if not all(0 <= i <= 5 for i in v):
            raise ValueError("active_layers must be a subset of {0..5}")
        return sorted(set(v))


class VerifierCfg(BaseModel):
    unsafe_threshold: float
    uncertainty_threshold: float
    consensus_collapse_std: float


class PipelineCfg(BaseModel):
    n_workers: int = Field(gt=0)
    request_timeout_s: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    parallel_samples: int = Field(default=1, ge=1)


class ExperimentCfg(BaseModel):
    dataset: str
    n_samples: int = Field(gt=0)
    seeds: list[int]
    betas: list[float] | None = None
    frameworks: list[str] | None = None
    sizes: list[int] | None = None


class LoggingCfg(BaseModel):
    trace_write_every: int = 1
    console_level: str = "INFO"


class Config(BaseModel):
    seed: int
    output_root: str
    dataset_cache: str
    models: ModelsCfg
    embedding: EmbeddingCfg
    ib: IBCfg
    verifier: VerifierCfg
    pipeline: PipelineCfg
    experiments: dict[str, ExperimentCfg]
    logging: LoggingCfg


# --- Loader -------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Config:
    """Load config.yaml + .env once per process."""
    if DEFAULT_ENV_PATH.exists():
        load_dotenv(DEFAULT_ENV_PATH)
    _normalize_env()
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def _normalize_env() -> None:
    """Map user's .env variable names to the ones downstream libraries expect."""
    # HuggingFace hub reads HF_TOKEN / HUGGING_FACE_HUB_TOKEN
    for src in ("Hf_api_key", "HF_API_KEY", "HUGGINGFACE_TOKEN"):
        val = os.getenv(src)
        if val:
            os.environ.setdefault("HF_TOKEN", val)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", val)
            break


def get_env(key: str, required: bool = True) -> str:
    val = os.getenv(key)
    if required and not val:
        raise RuntimeError(
            f"Missing required env var: {key}. "
            f"Check .env at {DEFAULT_ENV_PATH}."
        )
    return val or ""
