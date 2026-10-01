"""Run the ChannelGuard experiment suite with local Llama-3.1-8B-Instruct MLX 4-bit.

Uses the Qwen suite's experiment matrix and resume logic, unchanged Sonnet verifier,
and separate results_llama/ and paper/llama/ outputs.
"""
from __future__ import annotations

if __package__:
    from .run_qwen_suite import SuiteProfile, main
else:
    from run_qwen_suite import SuiteProfile, main


LLAMA_SUITE = SuiteProfile("llama", "Llama", "Llama-3.1-8B-Instruct", "8B", 4)


if __name__ == "__main__":
    main(LLAMA_SUITE)