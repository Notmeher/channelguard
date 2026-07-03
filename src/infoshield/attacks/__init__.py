"""Attack modules — importing registers each with the ATTACKS dict."""

from infoshield.attacks import (  # noqa: F401  (import for side effects)
    adaptive_paraphrase,
    consensus_collapse,
    memory_poisoning,
    prompt_injection,
    reasoning_state,
    tool_poisoning,
)
from infoshield.attacks.registry import ATTACKS, AttackFn, register

__all__ = ["ATTACKS", "AttackFn", "register"]
