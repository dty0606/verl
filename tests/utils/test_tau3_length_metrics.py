from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_tau3_length_metrics():
    module_path = Path(__file__).resolve().parents[2] / "verl" / "utils" / "tau3_length_metrics.py"
    spec = importlib.util.spec_from_file_location("tau3_length_metrics_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tau3_length_metrics = _load_tau3_length_metrics()
assistant_response_part_lengths = tau3_length_metrics.assistant_response_part_lengths
count_text_tokens = tau3_length_metrics.count_text_tokens


class WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()


def test_count_text_tokens_uses_tokenizer_without_special_tokens():
    tokenizer = WhitespaceTokenizer()

    assert count_text_tokens(tokenizer, "alpha beta gamma") == 3
    assert count_text_tokens(tokenizer, "") == 0


def test_assistant_response_part_lengths_split_think_tool_and_final_text():
    tokenizer = WhitespaceTokenizer()
    response = (
        "<think> check reservation </think>\n"
        "<tool_call> get_reservation_details </tool_call>\n"
        "I cannot cancel this reservation."
    )

    lengths = assistant_response_part_lengths(tokenizer, response)

    assert lengths["thinking"] == 4
    assert lengths["tool_call"] == 3
    assert lengths["final_text"] == 5


def test_unclosed_think_counts_as_thinking_not_final_text():
    tokenizer = WhitespaceTokenizer()

    lengths = assistant_response_part_lengths(tokenizer, "<think> loop loop loop")

    assert lengths["thinking"] == 4
    assert lengths["tool_call"] == 0
    assert lengths["final_text"] == 0


def test_tool_tag_inside_think_is_not_double_counted_as_tool_call():
    tokenizer = WhitespaceTokenizer()

    lengths = assistant_response_part_lengths(
        tokenizer,
        "<think> maybe <tool_call> search </tool_call> later </think> final",
    )

    assert lengths["thinking"] == 7
    assert lengths["tool_call"] == 0
    assert lengths["final_text"] == 1
