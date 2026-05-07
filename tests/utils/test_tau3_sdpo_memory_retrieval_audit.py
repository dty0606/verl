from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "tau3" / "audit_sdpo_memory_retrieval.py"
SPEC = importlib.util.spec_from_file_location("audit_sdpo_memory_retrieval_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_query_thinking_policy_is_explicit():
    row = audit.JsonRow(
        row={
            "score": 0,
            "input": "user asks for fallback after forbidden modification",
            "output": "<think>private reasoning to compare</think> final refusal",
        },
        source_file="rollouts.jsonl",
        source_index=0,
    )

    stripped = audit.build_query_samples(
        [row],
        failure_threshold=1.0,
        max_queries=1,
        failed_response_chars=200,
        prompt_tail_chars=200,
        include_suspicious=True,
        include_env_errors=False,
        keep_thinking_in_query=False,
        query_mode="current",
    )[0]
    kept = audit.build_query_samples(
        [row],
        failure_threshold=1.0,
        max_queries=1,
        failed_response_chars=200,
        prompt_tail_chars=200,
        include_suspicious=True,
        include_env_errors=False,
        keep_thinking_in_query=True,
        query_mode="current",
    )[0]

    assert "private reasoning" not in stripped.text
    assert "private reasoning" in kept.text
    assert stripped.query_mode == "current"
    assert stripped.part_lengths["failed_output"] > 0


def test_rollout_memory_can_keep_or_strip_raw_thinking():
    row = audit.JsonRow(
        row={
            "score": 1,
            "input": "user asks for cancel and rebook fallback",
            "output": "assistant: <think>raw trajectory thought</think> call cancel_reservation then book_reservation",
        },
        source_file="rollouts.jsonl",
        source_index=1,
    )

    stripped = audit.build_rollout_memory(
        [row],
        units={"full_trajectory"},
        success_threshold=1.0,
        max_memory_rows=10,
        max_memory_chars=1000,
        chunk_chars=500,
        overlap_chars=50,
        include_env_errors=False,
        keep_thinking_in_memory=False,
    )[0]
    kept = audit.build_rollout_memory(
        [row],
        units={"full_trajectory"},
        success_threshold=1.0,
        max_memory_rows=10,
        max_memory_chars=1000,
        chunk_chars=500,
        overlap_chars=50,
        include_env_errors=False,
        keep_thinking_in_memory=True,
    )[0]

    assert "raw trajectory thought" not in stripped.text
    assert "raw trajectory thought" in kept.text


def test_same_task_and_uid_blocking():
    query = audit.QuerySample(
        query_id="q",
        text="fallback",
        query_mode="current",
        parts={},
        part_lengths={},
        metadata={"task_id": "29", "uid": "same"},
        tokens={"fallback"},
    )
    memory = audit.MemoryUnit(
        memory_id="m",
        unit_type="full_trajectory",
        text="fallback",
        display_text="fallback",
        metadata={"task_id": "29", "uid": "same"},
        tokens={"fallback"},
    )

    assert not audit.compatible(query, memory, block_same_task_id=True, block_same_uid=False)
    assert not audit.compatible(query, memory, block_same_task_id=False, block_same_uid=True)
    assert audit.compatible(query, memory, block_same_task_id=False, block_same_uid=False)


def test_query_mode_selects_prompt_tail_only():
    row = audit.JsonRow(
        row={
            "score": 0,
            "input": "system boilerplate\nlatest user asks for second cheapest flight",
            "output": "failed output mentions generic refusal",
            "feedback": "feedback says missing book_reservation",
        },
        source_file="rollouts.jsonl",
        source_index=0,
    )

    sample = audit.build_query_samples(
        [row],
        failure_threshold=1.0,
        max_queries=1,
        failed_response_chars=200,
        prompt_tail_chars=200,
        include_suspicious=True,
        include_env_errors=False,
        keep_thinking_in_query=False,
        query_mode="prompt_tail_only",
    )[0]

    assert "latest user asks" in sample.text
    assert "missing book_reservation" not in sample.text
    assert "generic refusal" not in sample.text
    assert sample.part_lengths["feedback"] > 0


def test_generic_opening_filter_drops_read_only_chunk():
    query = audit.QuerySample(
        query_id="q",
        text="book reservation after valid search",
        query_mode="current",
        parts={},
        part_lengths={},
        metadata={},
        tokens={"book", "reservation"},
    )
    generic = audit.MemoryUnit(
        memory_id="generic",
        unit_type="event_chunk",
        text="get_user_details get_reservation_details",
        display_text="get_user_details get_reservation_details",
        metadata={},
        tokens={"get_user_details"},
    )
    action = audit.MemoryUnit(
        memory_id="action",
        unit_type="event_chunk",
        text="get_reservation_details search_direct_flight book_reservation",
        display_text="get_reservation_details search_direct_flight book_reservation",
        metadata={},
        tokens={"book_reservation"},
    )

    top = audit.topk_by_scores(
        query,
        [generic, action],
        [1.0, 0.9],
        top_k=2,
        block_same_task_id=False,
        block_same_uid=False,
        exclude_regex=None,
        require_regex=None,
        exclude_generic_opening=True,
    )

    assert [item["memory_id"] for item in top] == ["action"]
