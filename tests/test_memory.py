"""MemoryController: write/read/preload + read-time rescore hook."""
from __future__ import annotations

from infoshield.memory.controller import MemoryController


def test_write_read_ordered():
    m = MemoryController()
    m.write("k", "first", "agent_a")
    m.write("k", "second", "agent_b")
    m.write("k", "third", "agent_a")
    assert m.read("k") == ["first", "second", "third"]


def test_missing_key_returns_empty():
    assert MemoryController().read("nope") == []


def test_preload_bulk():
    m = MemoryController()
    m.preload("k", ["a", "b", "c"], agent_id="attacker")
    assert m.read("k") == ["a", "b", "c"]


def test_read_time_rescore_blocks_adversarial():
    # Attacker wrote 3 entries with fake low stored_risk.
    m_pre = MemoryController()
    m_pre.preload("k", ["safe", "IGNORE ALL", "safe again"], stored_risk=0.0)

    # Attach a rescore fn that flags anything containing "IGNORE".
    def rescore(text: str):
        if "IGNORE" in text:
            return 0.99, "BLOCKED", ""
        return 0.1, "PASSED", text

    m = MemoryController(rescore=rescore)
    for v in m_pre._store["k"]:
        m.write("k", v.value, v.agent_id, stored_risk=v.stored_risk)

    assert m.read("k") == ["safe", "safe again"]


def test_read_time_rescore_compresses():
    def rescore(text: str):
        if "TAIL" in text:
            return 0.5, "COMPRESSED", "compressed_head"
        return 0.0, "PASSED", text

    m = MemoryController(rescore=rescore)
    m.write("k", "head. TAIL.", "a")
    m.write("k", "clean", "a")
    assert m.read("k") == ["compressed_head", "clean"]


def test_clear():
    m = MemoryController()
    m.write("k", "v", "a")
    m.clear()
    assert m.keys() == []
