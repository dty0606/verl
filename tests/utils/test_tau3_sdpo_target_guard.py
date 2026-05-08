import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "tau3_sdpo_target_guard_test",
    ROOT / "verl" / "utils" / "tau3_sdpo_target_guard.py",
)
target_guard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(target_guard)
build_sdpo_target_guard_mask = target_guard.build_sdpo_target_guard_mask
has_unclosed_think = target_guard.has_unclosed_think


def test_has_unclosed_think_detects_open_target_block():
    assert has_unclosed_think("<think>still reasoning")
    assert has_unclosed_think("<think>done</think>\n<think>again")
    assert not has_unclosed_think("<think>done</think>\nfinal")


def test_target_guard_masks_corrupted_rows_without_touching_clean_rows():
    response_mask = torch.ones(5, 6)
    response_texts = [
        "clean final answer",
        "<think>runaway hidden chain",
        "loop loop loop loop loop loop",
        "tool call sequence",
        "malformed target",
    ]
    extras = {
        "tau3_live/nonterminal_fraction": [0, 0, 0, 1, 0],
        "tau3_live/budget_exhausted_fraction": [0, 0, 0, 0, 0],
        "tau3_live/tool_count": [0, 0, 0, 50, 0],
        "incorrect_format": [0, 0, 0, 0, 1],
    }

    loss_mask, masks = build_sdpo_target_guard_mask(
        response_mask=response_mask,
        response_texts=response_texts,
        reward_extra_infos_dict=extras,
        guard_cfg={
            "enabled": True,
            "corrupted_row_weight": 0.0,
            "max_response_tokens": 99,
            "repetition_ngram_size": 1,
            "repetition_max_count": 6,
            "max_tool_count": 32,
        },
    )

    assert torch.all(loss_mask[0] == 1)
    assert torch.all(loss_mask[1:] == 0)
    assert masks["open_think"].tolist() == [False, True, False, False, False]
    assert masks["repetition"].tolist() == [False, False, True, False, False]
    assert masks["nonterminal"].tolist() == [False, False, False, True, False]
    assert masks["tool_loop"].tolist() == [False, False, False, True, False]
    assert masks["parse_error"].tolist() == [False, False, False, False, True]


def test_target_guard_can_downweight_instead_of_zeroing():
    response_mask = torch.ones(1, 4)
    loss_mask, masks = build_sdpo_target_guard_mask(
        response_mask=response_mask,
        response_texts=["<think>unfinished"],
        reward_extra_infos_dict={},
        guard_cfg={"enabled": True, "corrupted_row_weight": 0.25},
    )

    assert masks["guarded"].tolist() == [True]
    assert torch.allclose(loss_mask, torch.full_like(response_mask, 0.25))
