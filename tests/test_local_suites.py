"""Offline checks for paired local-model suites and isolated outputs."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from infoshield.config import load_config
from scripts import run_qwen_suite as suite_runner
from scripts.run_llama_suite import LLAMA_SUITE


def test_llama_config_changes_only_model_and_output():
    qwen = load_config(str(suite_runner.QWEN_SUITE.config_path)).model_dump()
    llama = load_config(str(LLAMA_SUITE.config_path)).model_dump()
    assert qwen.pop("output_root") == "results_qwen"
    assert llama.pop("output_root") == "results_llama"
    assert qwen["models"]["agent"].pop("model") == "../Qwen3-4B-Instruct-2507-mlx-8bit"
    assert llama["models"]["agent"].pop("model") == "../Llama-3.1-8B-Instruct-mlx-4bit"
    assert llama == qwen


def test_local_suite_matrix():
    steps = suite_runner.build_steps()
    counts = {
        step.key: 30 * len(step.systems) * len(step.attacks)
        for step in steps
    }
    assert counts == {
        "main": 480, "gsm8k": 60, "sentinel": 180, "ablation": 210,
        "beta": 180, "bank": 90, "extra": 120, "baselines": 360,
        "judges": 0, "smoothllm": 120,
    }
    assert next(step for step in steps if step.key == "smoothllm").inner_runs == 10


def test_resume_does_not_read_other_model_results(tmp_path, monkeypatch):
    monkeypatch.setattr(suite_runner, "load_samples", lambda *args: [{"sample_id": "sample_0"}])
    step = next(step for step in suite_runner.build_steps() if step.key == "gsm8k")
    qwen_cfg = load_config(str(suite_runner.QWEN_SUITE.config_path)).model_copy(deep=True)
    llama_cfg = load_config(str(LLAMA_SUITE.config_path)).model_copy(deep=True)
    qwen_root = tmp_path / "results_qwen"
    llama_root = tmp_path / "results_llama"
    qwen_cfg.output_root = str(qwen_root)
    llama_cfg.output_root = str(llama_root)
    trace = qwen_root / "gsm8k" / "undefended" / "none" / "seed_0" / "sample_0.json"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}", encoding="utf-8")
    assert suite_runner.pending_runs(qwen_cfg, step, 1) == (2, 1)
    assert suite_runner.pending_runs(llama_cfg, step, 1) == (2, 0)
    own_trace = llama_root / trace.relative_to(qwen_root)
    own_trace.parent.mkdir(parents=True)
    own_trace.write_text("{}", encoding="utf-8")
    assert suite_runner.pending_runs(llama_cfg, step, 1) == (2, 1)


def test_reports_use_model_specific_paths_and_labels(tmp_path, monkeypatch):
    configs = {
        profile.key: load_config(str(profile.config_path)).model_copy(deep=True)
        for profile in (suite_runner.QWEN_SUITE, LLAMA_SUITE)
    }
    monkeypatch.setattr(suite_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(suite_runner, "load_samples", lambda *args: [{"sample_id": "sample_0"}])
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(__version__="test"))
    for profile in (suite_runner.QWEN_SUITE, LLAMA_SUITE):
        cfg = configs[profile.key]
        cfg.output_root = str(tmp_path / f"results_{profile.key}")
        path = suite_runner.write_report(cfg, suite_runner.build_steps(), 1, [], profile)
        assert path == tmp_path / "paper" / profile.key / "README.md"
        report = path.read_text(encoding="utf-8")
        other_label = "Qwen" if profile.key == "llama" else "Llama"
        assert other_label not in report
        assert profile.model_name in report
        assert f"MLX {profile.quantization_bits}-bit" in report
        assert profile.script_path in report
        assert "_Section failed to render" not in report
        numbers = json.loads(path.with_name("numbers.json").read_text(encoding="utf-8"))
        assert numbers["agent"]["quantization_bits"] == profile.quantization_bits
        assert numbers["agent"]["model"] == cfg.models.agent.model
    assert (tmp_path / "paper" / "qwen" / "README.md").exists()