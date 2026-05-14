from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "tau3_sdpo_mask_debug_test",
    ROOT / "verl" / "utils" / "tau3_sdpo_mask_debug.py",
)
mask_debug = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mask_debug)
build_tau3_sdpo_mask_debug_rows = mask_debug.build_tau3_sdpo_mask_debug_rows


class TensorLike:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def test_builds_compact_json_serializable_debug_row_with_metadata():
    rows = build_tau3_sdpo_mask_debug_rows(
        prompt_texts=["prompt"],
        response_texts=["alpha beta gamma delta"],
        response_mask=TensorLike([[1, 1, 1, 0]]),
        self_distillation_loss_mask=TensorLike([[1.0, 0.0, 1.0, 1.0]]),
        reward_tensor=TensorLike([[0.25, 0.75, 0.0, 0.0]]),
        reward_extra_infos_dict={
            "reward_source": ["tau3_live"],
            "official_score": [0.5],
            "strict_score": [1.0],
        },
        uids=["uid-1"],
        task_ids=["task-9"],
        response_token_texts=[["alpha ", "beta ", "gamma ", "delta"]],
        preview_chars=12,
    )

    json.dumps(rows)
    row = rows[0]
    assert row["uid"] == "uid-1"
    assert row["task_id"] == "task-9"
    assert row["reward"] == 1.0
    assert row["reward_source"] == "tau3_live"
    assert row["official_score"] == 0.5
    assert row["strict_score"] == 1.0
    assert row["response_token_count"] == 3
    assert row["selected_token_count"] == 2
    assert row["selected_token_fraction"] == pytest.approx(2 / 3)
    assert row["selected_text_preview"] == "alpha gamma"
    assert row["response_text_preview"].endswith("...")
    assert row["selected_spans"] == [
        {
            "response_token_start": 0,
            "response_token_end": 1,
            "prediction_position_start": 0,
            "prediction_position_end": 1,
            "text_preview": "alpha",
        },
        {
            "response_token_start": 2,
            "response_token_end": 3,
            "prediction_position_start": 2,
            "prediction_position_end": 3,
            "text_preview": "gamma",
        },
    ]


def test_prediction_position_alignment_starts_at_prompt_length_minus_one():
    rows = build_tau3_sdpo_mask_debug_rows(
        prompt_texts=["p0 p1 p2"],
        response_texts=["first second"],
        response_mask=[[1, 1]],
        self_distillation_target_token_mask=[[1, 0]],
        response_token_texts=[["first ", "second"]],
        prompt_token_counts=[3],
    )

    row = rows[0]
    assert row["prediction_position_basis"] == "full_sequence_shifted"
    assert row["prompt_token_count"] == 3
    assert row["selected_spans"][0]["response_token_start"] == 0
    assert row["selected_spans"][0]["prediction_position_start"] == 2
    assert row["selected_spans"][0]["prediction_position_end"] == 3
    assert all("prompt" not in span for span in row["selected_spans"])


def test_does_not_emit_prompt_token_rows_when_prompt_is_longer_than_response():
    rows = build_tau3_sdpo_mask_debug_rows(
        prompt_texts=["prompt token " * 20],
        response_texts=["ok"],
        response_mask=[[1]],
        self_distillation_target_token_mask=[[1]],
        response_token_texts=[["ok"]],
        prompt_token_counts=[20],
    )

    row = rows[0]
    assert len(row["selected_spans"]) == 1
    assert row["selected_spans"][0]["response_token_start"] == 0
    assert row["selected_spans"][0]["response_token_end"] == 1
    assert row["selected_spans"][0]["prediction_position_start"] == 19
    assert row["response_text_preview"] == "ok"


def test_empty_masks_produce_empty_spans_safely():
    rows = build_tau3_sdpo_mask_debug_rows(
        prompt_texts=["prompt"],
        response_texts=[""],
        response_mask=[[0, 0, 0]],
        self_distillation_target_token_mask=[[0, 0, 0]],
        response_token_texts=[["a", "b", "c"]],
    )

    row = rows[0]
    assert row["response_token_count"] == 0
    assert row["selected_token_count"] == 0
    assert row["selected_token_fraction"] == 0.0
    assert row["selected_text_preview"] == ""
    assert row["selected_spans"] == []
