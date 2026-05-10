from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verl.utils.tau3_sdpo_decision_spans import build_sdpo_decision_weight_mask


def test_decision_spans_shadow_detects_tool_and_write_actions():
    response_mask = torch.ones((2, 64), dtype=torch.float32)
    active = torch.tensor([1.0, 1.0])
    weights, masks, metrics = build_sdpo_decision_weight_mask(
        response_mask=response_mask,
        response_texts=[
            '<think>ready</think><tool_call>{"name":"cancel_reservation","arguments":{}}</tool_call>',
            "I am done.",
        ],
        active_mask=active,
        cfg={
            "base_weight": 0.5,
            "decision_span_weight": 1.0,
            "tool_action_weight": 2.0,
            "write_action_weight": 3.0,
            "final_answer_weight": 1.5,
        },
    )

    assert weights.shape == response_mask.shape
    assert masks["tool_action_span"][0].any()
    assert masks["write_action_span"][0].any()
    assert masks["final_answer_span"][1].any()
    assert metrics["self_distillation/action_span_token_fraction"] > 0
    assert metrics["self_distillation/max_decision_weight"] == 3.0


def test_short_unsolved_answer_is_not_upweighted():
    response_mask = torch.ones((1, 16), dtype=torch.float32)
    weights, masks, metrics = build_sdpo_decision_weight_mask(
        response_mask=response_mask,
        response_texts=["Done."],
        active_mask=torch.tensor([1.0]),
        cfg={"base_weight": 0.25, "final_answer_weight": 4.0, "short_unsolved_chars": 160},
    )

    assert masks["short_unsolved_guard"][0]
    assert torch.all(weights == 0.25)
    assert metrics["self_distillation/short_unsolved_selected_fraction"] == 1.0
