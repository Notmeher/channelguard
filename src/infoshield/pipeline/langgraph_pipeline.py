"""Assemble the multi-agent pipeline on LangGraph.

`build_pipeline(cfg, ib_layer=None)` returns a callable
`pipeline(sample_id, raw_input, ground_truth, experiment, system, attack, seed, **attack_kwargs) -> Trace`
that runs one full end-to-end sample and returns the completed Trace.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.graph import END, StateGraph

from infoshield.agents.planner import make_planner
from infoshield.agents.synthesizer import make_synthesizer
from infoshield.agents.verifier import make_verifier
from infoshield.agents.worker import make_workers
from infoshield.config import Config
from infoshield.llm import AnthropicClient, AzureOpenAIClient
from infoshield.memory.controller import MemoryController
from infoshield.pipeline.state import PipelineState, new_state
from infoshield.trace import Trace

# Placeholder for the IBLayer (wired in Phase 2). Kept as an opaque callable so
# Phase 1 does not import from ib/.
IBGateFn = Callable[[str], tuple[float, str, str]]  # -> (risk, status, output)


PipelineFn = Callable[..., Trace]


def build_pipeline(
    cfg: Config,
    ib_gates: Optional[dict[str, IBGateFn]] = None,
) -> PipelineFn:
    """Return a callable pipeline.

    Args:
        cfg: loaded Config.
        ib_gates: dict mapping gate id ("IB-0" ... "IB-5") to a gate function.
            When None, all boundaries pass through (undefended baseline).
    """
    ib_gates = ib_gates or {}

    agent_client = AzureOpenAIClient(
        deployment=cfg.models.agent.deployment,
        temperature=cfg.models.agent.temperature,
        max_tokens=cfg.models.agent.max_tokens,
    )
    verifier_client = AnthropicClient(
        model=cfg.models.verifier.model,
        temperature=cfg.models.verifier.temperature,
        max_tokens=cfg.models.verifier.max_tokens,
    )

    # IB-3 (tool output) rescore fn — used by MemoryController if provided.
    tool_gate = ib_gates.get("IB-3")

    planner = make_planner(cfg, agent_client, ib1=ib_gates.get("IB-1"))
    verifier = make_verifier(cfg, verifier_client, ib4=ib_gates.get("IB-4"))
    synthesizer = make_synthesizer(cfg, agent_client, ib5=ib_gates.get("IB-5"))
    # workers is built per-run because it closes over the fresh MemoryController.

    def pipeline(
        *,
        sample_id: str,
        raw_input: str,
        ground_truth: Any,
        experiment: str,
        system: str,
        attack: str,
        seed: int,
        dataset: str,
        tool_poisoned: bool = False,
        poison_memory: bool = False,
        injected_subtask_suffix: str = "",
        consensus_prefix: str = "",
        preloaded_memory: Optional[list[str]] = None,
    ) -> Trace:
        # Fresh memory + fresh workers per run — LangGraph nodes are closures
        # over the memory instance, so we build fresh nodes each call.
        mem = MemoryController(rescore=tool_gate)
        if preloaded_memory:
            mem.preload("shared_reasoning_trace", preloaded_memory)
        workers_bound = make_workers(
            cfg,
            agent_client,
            memory=mem,
            ib2=ib_gates.get("IB-2"),
            ib3=ib_gates.get("IB-3"),
        )

        # Re-compile a per-run graph so the worker node closes over this run's memory.
        g = StateGraph(PipelineState)
        g.add_node("planner", planner)
        g.add_node("workers", workers_bound)
        g.add_node("verifier", verifier)
        g.add_node("synthesizer", synthesizer)
        g.set_entry_point("planner")
        g.add_edge("planner", "workers")
        g.add_edge("workers", "verifier")
        g.add_edge("verifier", "synthesizer")
        g.add_edge("synthesizer", END)
        graph_run = g.compile()

        trace = Trace(
            sample_id=sample_id,
            experiment=experiment,
            system=system,
            attack=attack,
            seed=seed,
            dataset=dataset,
            raw_input=raw_input,
            ground_truth=ground_truth,
        )
        # IB-0 wraps raw_input in Phase 2 (via a hook set by build_pipeline consumer).
        initial = new_state(
            raw_input=raw_input,
            ground_truth=ground_truth,
            trace=trace,
            attack=attack,
            tool_poisoned=tool_poisoned,
            poison_memory=poison_memory,
            injected_subtask_suffix=injected_subtask_suffix,
            consensus_prefix=consensus_prefix,
        )

        # Apply IB-0 to the raw input if a gate is registered.
        ib0 = ib_gates.get("IB-0")
        if ib0 is not None:
            from infoshield.trace import GateDecision  # local import avoids Phase 1 cycles
            import time
            t0 = time.perf_counter()
            risk, status, output = ib0(raw_input)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            trace.add_gate(
                GateDecision(
                    gate_id="IB-0",
                    agent_id="input",
                    text_len_chars=len(raw_input),
                    risk=risk,
                    status=status,
                    latency_ms=latency_ms,
                )
            )
            initial["input_status"] = status
            if status == "BLOCKED":
                trace.final_output = (
                    "I cannot process this request; the input triggered a "
                    "safety gate and the pipeline was aborted."
                )
                trace.refused = True
                return trace
            if status == "COMPRESSED":
                initial["sanitized_input"] = output

        final_state = graph_run.invoke(initial)
        trace.final_output = final_state["final_output"]
        trace.refused = final_state.get("refused", False)
        return trace

    return pipeline
