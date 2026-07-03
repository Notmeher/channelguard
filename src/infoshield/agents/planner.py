"""Planner agent: decomposes raw_input into N subtasks.

Also applies IB-1 to each generated subtask description. IB-0 (raw input) is
applied by the pipeline factory before the graph runs.
"""
from __future__ import annotations

import json
import re
import uuid

from infoshield.config import Config
from infoshield.ib.gates import GateFn, apply_gate
from infoshield.llm import AzureOpenAIClient
from infoshield.pipeline.state import PipelineState, Subtask

_SYSTEM_PROMPT = (
    "You are a planning agent. Decompose the user's task into exactly {n} "
    "focused subtasks. Respond ONLY with a JSON array of {n} strings — no prose, "
    "no markdown, no explanation. Each string is one subtask description."
)


def _parse_subtasks_json(text: str, n: int) -> list[str]:
    """Extract a JSON array of strings from the LLM response robustly."""
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return [f"Address aspect {i + 1} of the query." for i in range(n)]
    try:
        arr = json.loads(match.group(0))
        if not isinstance(arr, list):
            raise ValueError
        # Coerce to strings, pad or truncate to length n
        arr = [str(x) for x in arr][:n]
        while len(arr) < n:
            arr.append(f"Address aspect {len(arr) + 1} of the query.")
        return arr
    except Exception:
        return [f"Address aspect {i + 1} of the query." for i in range(n)]


def make_planner(
    cfg: Config,
    client: AzureOpenAIClient,
    ib1: GateFn | None = None,
):
    """Return a LangGraph node function for the planner."""
    n_workers = cfg.pipeline.n_workers

    def planner_node(state: PipelineState) -> PipelineState:
        user_prompt = state["sanitized_input"]
        text, call = client.chat(
            system=_SYSTEM_PROMPT.format(n=n_workers),
            user=user_prompt,
            role_tag="planner",
        )
        state["trace"].add_llm(call)

        descs = _parse_subtasks_json(text, n_workers)
        subtasks: list[Subtask] = []
        for d in descs:
            gated_desc, status = apply_gate(
                ib1, d, gate_id="IB-1", agent_id="planner", trace=state["trace"]
            )
            passed = status != "BLOCKED"
            subtasks.append(
                Subtask(
                    id=f"st_{uuid.uuid4().hex[:6]}",
                    # When IB-1 blocks a subtask, drop the (adversarial) content
                    # from state entirely so downstream nodes / persisted traces
                    # never see it. Worker still skips via passed_ib1=False.
                    description=gated_desc if passed else "[BLOCKED by IB-1]",
                    passed_ib1=passed,
                )
            )
        state["subtasks"] = subtasks
        return state

    return planner_node
