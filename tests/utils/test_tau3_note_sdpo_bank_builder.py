from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "tau3" / "build_note_sdpo_bank.py"
SPEC = importlib.util.spec_from_file_location("build_note_sdpo_bank_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)

MEMORY_SCRIPT = ROOT / "verl" / "utils" / "tau3_sdpo_memory.py"
MEMORY_SPEC = importlib.util.spec_from_file_location("tau3_sdpo_memory_test", MEMORY_SCRIPT)
assert MEMORY_SPEC is not None and MEMORY_SPEC.loader is not None
memory = importlib.util.module_from_spec(MEMORY_SPEC)
sys.modules[MEMORY_SPEC.name] = memory
MEMORY_SPEC.loader.exec_module(memory)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_emit_prompts_redacts_sensitive_terms_and_thinking(tmp_path: Path):
    rollout = tmp_path / "10.jsonl"
    _write_jsonl(
        rollout,
        [
            {
                "score": 1.0,
                "input": "User Mei Brown asks to cancel reservation ABC123 on 2026-05-07.",
                "output": (
                    "<think>private deliberation</think>"
                    "<tool_call><function=get_user_details><parameter=user_id>mei_brown_7075</parameter></function></tool_call>"
                    "<tool_call><function=cancel_reservation><parameter>ABC123</parameter></function></tool_call>"
                    "I cancelled reservation ABC123 for Mei."
                ),
                "gts": json.dumps({"id": "task_42"}),
            }
        ],
    )
    prompts = tmp_path / "prompts.jsonl"
    rc = builder.main(
        [
            "emit-prompts",
            "--rollout",
            str(rollout),
            "--output-prompts",
            str(prompts),
            "--score-threshold",
            "1",
            "--source-step-max",
            "10",
        ]
    )

    assert rc == 0
    rows = _read_jsonl(prompts)
    assert len(rows) == 1
    packet_text = json.dumps(rows[0]["packet"], sort_keys=True)
    assert "private deliberation" not in packet_text
    assert "mei_brown_7075" not in packet_text
    assert "ABC123" not in packet_text
    assert "Mei" not in packet_text
    assert rows[0]["packet"]["tool_sequence"] == ["get_user_details", "cancel_reservation"]
    assert rows[0]["source"]["task_id"] == "task_42"


def test_compile_rejects_leaky_note_and_accepts_clean_note(tmp_path: Path):
    prompt = {
        "prompt_id": "p1",
        "source": {
            "source_file": "10.jsonl",
            "source_index": 0,
            "source_step": 10,
            "source_hash": "abc",
            "score": 1.0,
            "task_id": "task_42",
            "uid": "u1",
            "split": "train",
        },
        "safety": {
            "blocked_task_ids": ["task_42"],
            "blocked_entity_ids": ["Mei", "ABC123", "mei_brown_7075"],
        },
        "packet": {"domain": "airline"},
    }
    prompts = tmp_path / "prompts.jsonl"
    writer = tmp_path / "writer.jsonl"
    bank = tmp_path / "bank.jsonl"
    rejections = tmp_path / "rejections.jsonl"
    _write_jsonl(prompts, [prompt])
    _write_jsonl(
        writer,
        [
            {
                "prompt_id": "p1",
                "note": {
                    "note_type": "decision_lesson",
                    "situation": "Mei asks about reservation ABC123.",
                    "applicability": ["The user has requested cancellation."],
                    "known_evidence": ["The reservation exists."],
                    "missing_state_required_before_action": ["Cancellation eligibility must be checked."],
                    "decision_boundary": "sufficient_to_act",
                    "correct_action_pattern": ["cancel_reservation when policy permits"],
                    "avoid": ["repeat lookup"],
                    "unsafe_if": ["Cancellation eligibility is unknown."],
                    "stop_condition": "Stop once cancellation eligibility is known.",
                    "why_reusable": "It teaches when a verified cancellation can proceed.",
                    "task_cluster": "cancel_after_verified_policy",
                    "tool_family": ["cancel_reservation"],
                    "confidence": 0.8,
                },
            },
            {
                "prompt_id": "p1",
                "note": {
                    "note_type": "decision_lesson",
                    "situation": "A user asks to cancel an existing reservation.",
                    "applicability": ["The requested reservation has been located."],
                    "known_evidence": ["The relevant reservation and cancellation eligibility have been verified."],
                    "missing_state_required_before_action": ["No additional state is required after eligibility is known."],
                    "decision_boundary": "sufficient_to_act",
                    "correct_action_pattern": ["Use cancel_reservation if cancellation policy permits."],
                    "avoid": ["Do not repeat the same lookup after eligibility is known."],
                    "unsafe_if": ["The reservation has not been identified."],
                    "stop_condition": "Stop searching once the eligibility check determines the safe next action.",
                    "why_reusable": "The lesson transfers to cancellation tasks where evidence is already sufficient.",
                    "task_cluster": "cancel_after_verified_policy",
                    "tool_family": ["cancel_reservation"],
                    "confidence": 0.9,
                },
            },
        ],
    )

    rc = builder.main(
        [
            "compile",
            "--prompt-jsonl",
            str(prompts),
            "--writer-output-jsonl",
            str(writer),
            "--output-bank",
            str(bank),
            "--rejections-jsonl",
            str(rejections),
            "--writer-model",
            "ema_teacher_step10",
        ]
    )

    assert rc == 0
    rejected = _read_jsonl(rejections)
    accepted = _read_jsonl(bank)
    assert [row["reason"] for row in rejected] == ["source_leakage_in_note"]
    assert len(accepted) == 1
    assert accepted[0]["unit_type"] == "teacher_written_decision_note"
    assert accepted[0]["status"] == "active"
    assert accepted[0]["source"]["source_step"] == 10
    assert accepted[0]["leakage_audit"]["passed"] is True
    assert accepted[0]["utility_stats"]["used_n"] == 0
    assert accepted[0]["decision_state"] == "sufficient_to_act"
    assert accepted[0]["blocked_task_ids"] == ["task_42"]
    assert "Mei" not in accepted[0]["display_text"]
    assert "ABC123" not in accepted[0]["display_text"]

    memory_bank = memory.load_memory_bank(str(bank))
    card = memory_bank.retrieve("The reservation eligibility is known; should cancel now?", mode="relevant")
    assert card is not None
    assert card.payload["note_writer"]["model"] == "ema_teacher_step10"


def test_compile_merges_similar_notes_as_memory_momentum(tmp_path: Path):
    prompts = tmp_path / "prompts.jsonl"
    writer = tmp_path / "writer.jsonl"
    bank = tmp_path / "bank.jsonl"
    prompt_template = {
        "source": {
            "source_file": "10.jsonl",
            "source_index": 0,
            "source_step": 10,
            "score": 1.0,
            "split": "train",
        },
        "safety": {"blocked_task_ids": [], "blocked_entity_ids": []},
        "packet": {"domain": "airline"},
    }
    _write_jsonl(
        prompts,
        [
            {"prompt_id": "p1", **prompt_template, "source": {**prompt_template["source"], "source_hash": "a"}},
            {"prompt_id": "p2", **prompt_template, "source": {**prompt_template["source"], "source_hash": "b"}},
        ],
    )
    note = {
        "note_type": "decision_lesson",
        "situation": "A user requests a cancellation after eligibility is known.",
        "applicability": ["The reservation has been found."],
        "known_evidence": ["Cancellation eligibility is already verified."],
        "missing_state_required_before_action": ["No additional state is needed once eligibility is verified."],
        "decision_boundary": "sufficient_to_act",
        "correct_action_pattern": ["Cancel if allowed by policy."],
        "avoid": ["Do not keep searching after eligibility is known."],
        "unsafe_if": ["Eligibility was not verified."],
        "stop_condition": "Stop when the eligibility check fixes the safe action.",
        "why_reusable": "This prevents read-and-stall loops.",
        "task_cluster": "cancel_after_verified_policy",
        "tool_family": ["cancel_reservation"],
        "confidence": 0.9,
    }
    _write_jsonl(writer, [{"prompt_id": "p1", "note": note}, {"prompt_id": "p2", "note": note}])

    rc = builder.main(
        [
            "compile",
            "--prompt-jsonl",
            str(prompts),
            "--writer-output-jsonl",
            str(writer),
            "--output-bank",
            str(bank),
        ]
    )

    assert rc == 0
    rows = _read_jsonl(bank)
    assert len(rows) == 1
    assert rows[0]["memory_momentum"]["support_count"] == 2
    assert sorted(rows[0]["memory_momentum"]["source_hashes"]) == ["a", "b"]


def test_compile_rejects_invalid_boundary_and_confidence(tmp_path: Path):
    prompt = {
        "prompt_id": "p1",
        "source": {"source_hash": "abc", "score": 1.0, "split": "train"},
        "safety": {"blocked_task_ids": [], "blocked_entity_ids": []},
        "packet": {"domain": "airline"},
    }
    base_note = {
        "note_type": "decision_lesson",
        "situation": "A user asks to change a booking after evidence is available.",
        "applicability": ["The booking has been inspected."],
        "known_evidence": ["The allowed workflow is known."],
        "missing_state_required_before_action": ["No missing state remains."],
        "decision_boundary": "sufficient_to_act",
        "correct_action_pattern": ["Commit the allowed tool action."],
        "avoid": ["Do not keep searching."],
        "unsafe_if": ["The policy state is unknown."],
        "stop_condition": "Stop when the allowed next action is determined.",
        "why_reusable": "It teaches decision sufficiency.",
        "task_cluster": "commit_after_evidence",
        "tool_family": ["update_reservation"],
        "confidence": 0.8,
    }
    prompts = tmp_path / "prompts.jsonl"
    writer = tmp_path / "writer.jsonl"
    bank = tmp_path / "bank.jsonl"
    rejections = tmp_path / "rejections.jsonl"
    _write_jsonl(prompts, [prompt])
    _write_jsonl(
        writer,
        [
            {"prompt_id": "p1", "note": {**base_note, "decision_boundary": "creative_guess"}},
            {"prompt_id": "p1", "note": {**base_note, "confidence": 1.5}},
            {"prompt_id": "p1", "note": {**base_note, "extra_hint": "do this exact task"}},
        ],
    )

    rc = builder.main(
        [
            "compile",
            "--prompt-jsonl",
            str(prompts),
            "--writer-output-jsonl",
            str(writer),
            "--output-bank",
            str(bank),
            "--rejections-jsonl",
            str(rejections),
        ]
    )

    assert rc == 0
    assert _read_jsonl(bank) == []
    assert [row["reason"] for row in _read_jsonl(rejections)] == [
        "invalid_decision_boundary",
        "invalid_confidence",
        "unknown_note_fields",
    ]


def test_live_memory_loader_requires_active_audited_train_cards(tmp_path: Path):
    rows = [
        {
            "card_id": "active",
            "split": "train",
            "status": "active",
            "unit_type": "teacher_written_decision_note",
            "leakage_audit": {"passed": True},
            "task_cluster": "cancel_after_verified_policy",
            "decision_state": "sufficient_to_act",
            "display_text": "Use cancel_reservation after eligibility is verified.",
        },
        {
            "card_id": "candidate",
            "split": "train",
            "status": "candidate",
            "unit_type": "teacher_written_decision_note",
            "leakage_audit": {"passed": True},
            "display_text": "candidate card",
        },
        {
            "card_id": "failed_audit",
            "split": "train",
            "status": "active",
            "unit_type": "teacher_written_decision_note",
            "leakage_audit": {"passed": False},
            "display_text": "failed audit card",
        },
        {
            "card_id": "eval",
            "split": "eval",
            "status": "active",
            "unit_type": "teacher_written_decision_note",
            "leakage_audit": {"passed": True},
            "display_text": "eval card",
        },
        {
            "card_id": "raw",
            "split": "train",
            "status": "active",
            "leakage_audit": {"passed": True},
            "display_text": "raw trajectory shaped card",
        },
        {
            "card_id": "same_uid",
            "split": "train",
            "status": "active",
            "unit_type": "teacher_written_decision_note",
            "leakage_audit": {"passed": True},
            "blocked_uids": ["u1"],
            "display_text": "same uid card",
        },
    ]
    bank_path = tmp_path / "bank.jsonl"
    _write_jsonl(bank_path, rows)

    memory.load_memory_bank.cache_clear()
    bank = memory.load_memory_bank(str(bank_path))
    card = bank.retrieve("eligibility verified cancellation", mode="relevant")
    blocked_card = bank.retrieve("same uid", mode="relevant", blocked_ids=["u1"])

    assert card is not None
    assert card.card_id == "active"
    assert blocked_card is not None
    assert blocked_card.card_id == "active"
