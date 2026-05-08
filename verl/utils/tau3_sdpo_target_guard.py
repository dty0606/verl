from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

import torch


def reward_extra_float_list(
    reward_extra_infos_dict: Optional[dict[str, list]],
    key: str,
    *,
    batch_size: int,
    default: float = 0.0,
) -> list[float]:
    values = (reward_extra_infos_dict or {}).get(key)
    if values is None:
        return [default] * batch_size
    result: list[float] = []
    for item in list(values)[:batch_size]:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            result.append(default)
    if len(result) < batch_size:
        result.extend([default] * (batch_size - len(result)))
    return result


def has_unclosed_think(text: str) -> bool:
    """Detect target-side generations that leave a Qwen thinking block open."""
    text = text or ""
    opens = [m.start() for m in re.finditer(r"<think\b[^>]*>", text, flags=re.IGNORECASE)]
    closes = [m.start() for m in re.finditer(r"</think>", text, flags=re.IGNORECASE)]
    return len(opens) > len(closes) or (bool(opens) and (not closes or opens[-1] > closes[-1]))


def has_repetition_loop(text: str, *, ngram_size: int, max_count: int) -> bool:
    if ngram_size <= 0 or max_count <= 0:
        return False
    words = re.findall(r"\S+", text or "")
    if len(words) < ngram_size * max_count:
        return False
    counts: Counter[tuple[str, ...]] = Counter(
        tuple(words[i : i + ngram_size]) for i in range(0, len(words) - ngram_size + 1)
    )
    return any(count >= max_count for count in counts.values())


def build_sdpo_target_guard_mask(
    *,
    response_mask: torch.Tensor,
    response_texts: list[str],
    reward_extra_infos_dict: Optional[dict[str, list]],
    guard_cfg: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build an optional target-token mask without changing SDPO row routing."""
    guard_cfg = guard_cfg or {}
    loss_mask = response_mask.to(dtype=torch.float32).clone()
    batch_size = response_mask.size(0)
    device = response_mask.device

    def _zeros() -> torch.Tensor:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    if not bool(guard_cfg.get("enabled", True)):
        return loss_mask, {
            "nonterminal": _zeros(),
            "budget_exhausted": _zeros(),
            "response_saturated": _zeros(),
            "parse_error": _zeros(),
            "open_think": _zeros(),
            "repetition": _zeros(),
            "tool_loop": _zeros(),
            "guarded": _zeros(),
        }

    nonterminal = torch.tensor(
        [
            value >= 0.5
            for value in reward_extra_float_list(
                reward_extra_infos_dict,
                "tau3_live/nonterminal_fraction",
                batch_size=batch_size,
            )
        ],
        dtype=torch.bool,
        device=device,
    )
    budget_exhausted = torch.tensor(
        [
            value >= 0.5
            for value in reward_extra_float_list(
                reward_extra_infos_dict,
                "tau3_live/budget_exhausted_fraction",
                batch_size=batch_size,
            )
        ],
        dtype=torch.bool,
        device=device,
    )
    max_response_tokens = int(guard_cfg.get("max_response_tokens", response_mask.size(1)))
    response_saturated = response_mask.to(torch.float32).sum(dim=-1) >= max_response_tokens
    parse_error = torch.tensor(
        [
            value >= 0.5
            for value in reward_extra_float_list(
                reward_extra_infos_dict,
                "incorrect_format",
                batch_size=batch_size,
            )
        ],
        dtype=torch.bool,
        device=device,
    )

    open_think_values = [has_unclosed_think(text) for text in response_texts[:batch_size]]
    open_think_values.extend([False] * max(0, batch_size - len(open_think_values)))
    open_think = torch.tensor(open_think_values, dtype=torch.bool, device=device)

    repetition_values = [
        has_repetition_loop(
            text,
            ngram_size=int(guard_cfg.get("repetition_ngram_size", 8)),
            max_count=int(guard_cfg.get("repetition_max_count", 4)),
        )
        for text in response_texts[:batch_size]
    ]
    repetition_values.extend([False] * max(0, batch_size - len(repetition_values)))
    repetition = torch.tensor(repetition_values, dtype=torch.bool, device=device)

    tool_counts = reward_extra_float_list(
        reward_extra_infos_dict,
        "tau3_live/tool_count",
        batch_size=batch_size,
    )
    max_tool_count = float(guard_cfg.get("max_tool_count", 32))
    tool_loop = torch.tensor([count >= max_tool_count for count in tool_counts], dtype=torch.bool, device=device)

    guarded = _zeros()
    if bool(guard_cfg.get("mask_nonterminal", True)):
        guarded = guarded | nonterminal
    if bool(guard_cfg.get("mask_budget_exhausted", True)):
        guarded = guarded | budget_exhausted
    if bool(guard_cfg.get("mask_response_saturated", True)):
        guarded = guarded | response_saturated
    if bool(guard_cfg.get("mask_parse_error", True)):
        guarded = guarded | parse_error
    if bool(guard_cfg.get("mask_open_think", True)):
        guarded = guarded | open_think
    if bool(guard_cfg.get("mask_repetition", True)):
        guarded = guarded | repetition
    if bool(guard_cfg.get("mask_tool_loop", True)):
        guarded = guarded | tool_loop

    downweight = float(guard_cfg.get("corrupted_row_weight", 0.0))
    downweight = max(0.0, min(1.0, downweight))
    if bool(guarded.any().item()):
        loss_mask[guarded] = loss_mask[guarded] * downweight

    return loss_mask, {
        "nonterminal": nonterminal,
        "budget_exhausted": budget_exhausted,
        "response_saturated": response_saturated,
        "parse_error": parse_error,
        "open_think": open_think,
        "repetition": repetition,
        "tool_loop": tool_loop,
        "guarded": guarded,
    }
