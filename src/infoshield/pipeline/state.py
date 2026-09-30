"""LangGraph pipeline state.

Uses TypedDict (LangGraph's native format) so channel-based reducers work.
The Trace object is threaded through as a first-class field — every node
appends to `state["trace"]` via the wrappers in `infoshield.trace`.
"""
from __future__ import annotations

from typing import Any, TypedDict

from infoshield.trace import Trace


class Subtask(TypedDict):
    id: str
    description: str
    passed_ib1: bool          # False if IB-1 blocked this subtask


class WorkerOutput(TypedDict):
    worker_id: str
    subtask_id: str
    subtask: str
    tool_output: str          # possibly blocked/compressed by IB-3
    tool_output_status: str   # "PASSED" | "COMPRESSED" | "BLOCKED"
    output: str               # possibly blocked/compressed by IB-2
    output_status: str
    memory_context: str


class PipelineState(TypedDict, total=False):
    # --- Inputs (set once at start) --------------------------------------
    raw_input: str
    ground_truth: Any
    sanitized_input: str      # after IB-0 (unchanged in undefended)
    input_status: str         # gate status for IB-0
    # --- Attack config (set by runner before execution) ------------------
    attack: str               # attack name applied to this run
    tool_poisoned: bool       # true if tool should return poisoned outputs
    tool_marker: str          # payload token inside poisoned tool output ("" = ATTACK_SUCCESS)
    poison_memory: bool       # true if memory was pre-populated with adversarial entries
    injected_subtask_suffix: str   # non-empty triggers reasoning-state injection
    consensus_prefix: str     # non-empty forces consensus collapse attempt
    # --- Populated by nodes ----------------------------------------------
    subtasks: list[Subtask]
    worker_outputs: list[WorkerOutput]
    verifier_summary: str
    final_output: str
    refused: bool
    # --- Trace (accumulator) ---------------------------------------------
    trace: Trace


def new_state(
    raw_input: str,
    ground_truth: Any,
    trace: Trace,
    attack: str = "none",
    tool_poisoned: bool = False,
    tool_marker: str = "",
    poison_memory: bool = False,
    injected_subtask_suffix: str = "",
    consensus_prefix: str = "",
) -> PipelineState:
    """Fresh state for one pipeline run."""
    return {
        "raw_input": raw_input,
        "ground_truth": ground_truth,
        "sanitized_input": raw_input,
        "input_status": "PASSED",
        "attack": attack,
        "tool_poisoned": tool_poisoned,
        "tool_marker": tool_marker,
        "poison_memory": poison_memory,
        "injected_subtask_suffix": injected_subtask_suffix,
        "consensus_prefix": consensus_prefix,
        "subtasks": [],
        "worker_outputs": [],
        "verifier_summary": "",
        "final_output": "",
        "refused": False,
        "trace": trace,
    }
