from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "tau3" / "build_sdpo_memory_teacher_probe.py"
SPEC = importlib.util.spec_from_file_location("build_sdpo_memory_teacher_probe_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_teacher_probe_emits_relevant_random_irrelevant_and_peer_arms(tmp_path: Path, monkeypatch):
    rollout = tmp_path / "rollout.jsonl"
    memory_path = tmp_path / "memory.jsonl"
    output = tmp_path / "probe.jsonl"
    _write_jsonl(
        rollout,
        [
            {
                "uid": "u1",
                "score": 1.0,
                "input": "user asks to cancel",
                "output": "<think>private</think>cancel_reservation",
            },
            {
                "uid": "u1",
                "score": 0.0,
                "input": "user asks to cancel",
                "output": "I will keep searching.",
                "feedback": "The agent should cancel after eligibility is known.",
            },
        ],
    )
    cards = []
    for idx, text in enumerate(("cancel after eligibility", "book after search", "transfer when impossible")):
        cards.append(
            {
                "card_id": f"c{idx}",
                "split": "train",
                "status": "active",
                "unit_type": "teacher_written_decision_note",
                "leakage_audit": {"passed": True},
                "task_cluster": f"cluster_{idx}",
                "decision_state": "sufficient_to_act",
                "display_text": text,
                "embedding_text": text,
            }
        )
    _write_jsonl(memory_path, cards)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_sdpo_memory_teacher_probe.py",
            "--rollout",
            str(rollout),
            "--memory-path",
            str(memory_path),
            "--output-jsonl",
            str(output),
            "--max-samples",
            "1",
        ],
    )

    rc = probe.main()

    assert rc == 0
    rows = _read_jsonl(output)
    arms = {row["arm"] for row in rows}
    assert {
        "T0_original",
        "T1_relevant_memory",
        "T2_random_memory",
        "T3_no_context",
        "T4_matched_irrelevant_memory",
        "T5_peer_success",
    } <= arms
    no_context = next(row for row in rows if row["arm"] == "T3_no_context")
    assert "feedback from your unsuccessful earlier attempt" not in no_context["prompt"]
    peer = next(row for row in rows if row["arm"] == "T5_peer_success")
    assert "Correct solution" in peer["prompt"]
    assert "private" not in peer["prompt"]
    assert "feedback from your unsuccessful earlier attempt" not in peer["prompt"]
