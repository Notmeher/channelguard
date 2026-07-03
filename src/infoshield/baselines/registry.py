"""Baseline registry: `SYSTEMS[name](cfg)` returns a pipeline callable.

Every baseline shares the SAME pipeline shell as INFOSHIELD but with a
different gate configuration or pre-filter:

  undefended   : no gates at all
  ibprotector  : single IB-0 gate on the raw input only
  smoothllm    : perturb input 10x, run pipeline on each copy, majority vote
  llm_guard    : LLM-classifier gate on IB-0 (cheap Sonnet call)
  perplexity   : GPT-2 perplexity threshold on IB-0
  infoshield   : all six IB gates active (from config)
"""
from __future__ import annotations

from typing import Callable

from infoshield.config import Config
from infoshield.pipeline.langgraph_pipeline import PipelineFn

SystemBuilder = Callable[[Config], PipelineFn]

SYSTEMS: dict[str, SystemBuilder] = {}


def register(name: str):
    def deco(fn: SystemBuilder) -> SystemBuilder:
        SYSTEMS[name] = fn
        return fn
    return deco
