"""Local MLX wrapper (Apple Silicon) for on-device open models, e.g. Qwen3-4B.

Contract: `chat(system, user, role_tag)` -> (text, LLMCall).

MLX GPU streams are bound to the thread that created them, and one model on one
GPU cannot usefully run concurrent generations. So every model path gets ONE
dedicated engine thread that owns the weights; pipeline threads (the runner's
ThreadPoolExecutor) submit jobs to its queue and block on a Future. Verifier
calls to Anthropic still overlap freely across pipeline threads.

`LLMCall.latency_ms` records pure generation time on the engine thread, NOT
queue wait, so per-role latency stays comparable across parallelism levels.

Like Groq/Anthropic, a local model has no server-side content filter: refusals
come back as ordinary completion text and never emit a provider sentinel.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from pathlib import Path

from infoshield.trace import LLMCall

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _MLXEngine:
    """Owns one loaded MLX model on a dedicated thread; serializes generation."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._jobs: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._load_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._loop, name=f"mlx-engine:{Path(model_path).name}", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._load_error is not None:
            raise RuntimeError(f"Failed to load MLX model at {model_path}") from self._load_error

    def _loop(self) -> None:
        try:
            from mlx_lm import load

            self.model, self.tokenizer = load(self.model_path)
        except BaseException as e:  # surface to the constructing thread
            self._load_error = e
            self._ready.set()
            return
        self._ready.set()
        while True:
            fut, messages, max_tokens, temperature = self._jobs.get()
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(self._generate(messages, max_tokens, temperature))
            except BaseException as e:
                fut.set_exception(e)

    def _generate(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> tuple[str, int, int, float]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        sampler = make_sampler(temp=temperature)
        pieces: list[str] = []
        last = None
        started = time.perf_counter()
        for resp in stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            pieces.append(resp.text)
            last = resp
        gen_ms = (time.perf_counter() - started) * 1000.0
        prompt_tokens = last.prompt_tokens if last else 0
        completion_tokens = last.generation_tokens if last else 0
        return "".join(pieces), prompt_tokens, completion_tokens, gen_ms

    def submit(self, messages: list[dict], max_tokens: int, temperature: float) -> Future:
        fut: Future = Future()
        self._jobs.put((fut, messages, max_tokens, temperature))
        return fut


_ENGINES: dict[str, _MLXEngine] = {}
_ENGINES_LOCK = threading.Lock()


def _resolve_path(model: str) -> str:
    p = Path(model).expanduser()
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"MLX model directory not found: {p}")
    return str(p)


def _get_engine(model: str) -> _MLXEngine:
    path = _resolve_path(model)
    with _ENGINES_LOCK:
        if path not in _ENGINES:
            _ENGINES[path] = _MLXEngine(path)
        return _ENGINES[path]


class MLXClient:
    """Chat client backed by a shared local MLX engine (one per model path)."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.engine = _get_engine(model)
        self.model = Path(self.engine.model_path).name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system: str, user: str, role_tag: str = "agent") -> tuple[str, LLMCall]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text, prompt_tokens, completion_tokens, gen_ms = self.engine.submit(
            messages, self.max_tokens, self.temperature
        ).result()
        return text, LLMCall(
            role=role_tag,
            provider="mlx",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=gen_ms,
        )
