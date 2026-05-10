from __future__ import annotations

import re
from typing import Any

import torch


_TOOL_OR_ACTION_RE = re.compile(
    r"(?:<tool_call>|<function=|\"name\"\s*:\s*\"|"
    r"\b(?:get_user_details|get_reservation_details|search_direct_flight|search_onestop_flight|"
    r"book_reservation|update_reservation|cancel_reservation|transfer_to_human_agents)\b)",
    flags=re.IGNORECASE,
)
_WRITE_ACTION_RE = re.compile(
    r"\b(?:book_reservation|update_reservation|cancel_reservation|transfer_to_human_agents)\b",
    flags=re.IGNORECASE,
)
_FINAL_OR_BOUNDARY_RE = re.compile(
    r"(?:</think>|human agent|transfer|not allowed|cannot|can't|unable|I have|I've|done|completed)",
    flags=re.IGNORECASE,
)


def _regex_spans(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in pattern.finditer(text or "")]


def _final_spans(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    spans = _regex_spans(_FINAL_OR_BOUNDARY_RE, text)
    think_close = text.lower().rfind("</think>")
    if think_close >= 0 and think_close + len("</think>") < len(text):
        spans.append((think_close + len("</think>"), len(text)))
    elif not spans and len(text) < 480:
        spans.append((0, len(text)))
    return spans


def _span_to_token_mask(
    *,
    text: str,
    spans: list[tuple[int, int]],
    valid_tokens: torch.Tensor,
    pad_tokens: int = 8,
) -> torch.Tensor:
    mask = torch.zeros_like(valid_tokens, dtype=torch.bool)
    token_positions = torch.nonzero(valid_tokens.to(torch.bool), as_tuple=False).flatten()
    if token_positions.numel() == 0 or not text:
        return mask
    n_tokens = int(token_positions.numel())
    text_len = max(len(text), 1)
    for start_char, end_char in spans:
        start = max(0, min(text_len, start_char))
        end = max(start, min(text_len, end_char))
        start_idx = max(0, int(start / text_len * n_tokens) - pad_tokens)
        end_idx = min(n_tokens, int((end / text_len) * n_tokens) + pad_tokens + 1)
        if end_idx > start_idx:
            mask[token_positions[start_idx:end_idx]] = True
    return mask


def build_sdpo_decision_weight_mask(
    *,
    response_mask: torch.Tensor,
    response_texts: list[str],
    active_mask: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    """Build optional decision/action-span weights without changing row routing."""

    cfg = cfg or {}
    weights = response_mask.to(dtype=torch.float32).clone()
    batch_size, response_len = response_mask.shape
    device = response_mask.device

    decision_span = torch.zeros((batch_size, response_len), dtype=torch.bool, device=device)
    action_span = torch.zeros_like(decision_span)
    write_action_span = torch.zeros_like(decision_span)
    final_answer_span = torch.zeros_like(decision_span)
    short_unsolved_guard = torch.zeros(batch_size, dtype=torch.bool, device=device)

    active_bool = active_mask.to(device=device).to(torch.bool)
    for idx in range(batch_size):
        text = str(response_texts[idx] if idx < len(response_texts) else "")
        valid = response_mask[idx].to(torch.bool)
        tool_spans = _regex_spans(_TOOL_OR_ACTION_RE, text)
        write_spans = _regex_spans(_WRITE_ACTION_RE, text)
        final_spans = _final_spans(text)

        action_span[idx] = _span_to_token_mask(text=text, spans=tool_spans, valid_tokens=valid)
        write_action_span[idx] = _span_to_token_mask(text=text, spans=write_spans, valid_tokens=valid)
        final_answer_span[idx] = _span_to_token_mask(text=text, spans=final_spans, valid_tokens=valid)
        decision_span[idx] = action_span[idx] | write_action_span[idx] | final_answer_span[idx]

        stripped = " ".join(text.split())
        has_action = bool(tool_spans or write_spans)
        short_unsolved_guard[idx] = (
            active_bool[idx]
            and len(stripped) < int(cfg.get("short_unsolved_chars", 160))
            and not has_action
        )

    base_weight = float(cfg.get("base_weight", 1.0))
    decision_weight = float(cfg.get("decision_span_weight", 1.0))
    action_weight = float(cfg.get("tool_action_weight", decision_weight))
    write_weight = float(cfg.get("write_action_weight", action_weight))
    final_weight = float(cfg.get("final_answer_weight", decision_weight))

    weights = torch.where(response_mask.to(torch.bool), torch.full_like(weights, base_weight), torch.zeros_like(weights))
    weights = torch.where(decision_span, torch.full_like(weights, decision_weight), weights)
    weights = torch.where(action_span, torch.full_like(weights, action_weight), weights)
    weights = torch.where(write_action_span, torch.full_like(weights, write_weight), weights)
    weights = torch.where(final_answer_span, torch.full_like(weights, final_weight), weights)
    weights = torch.where(short_unsolved_guard.unsqueeze(1), response_mask.to(dtype=torch.float32) * base_weight, weights)

    selected_tokens = response_mask.to(torch.bool) & active_bool.unsqueeze(1)
    selected_count = selected_tokens.float().sum().clamp(min=1.0)
    active_count = active_bool.float().sum().clamp(min=1.0)
    metrics = {
        "self_distillation/decision_weighting_enabled": 1.0,
        "self_distillation/decision_weighting_shadow_mode": float(bool(cfg.get("shadow_mode", True))),
        "self_distillation/decision_weighted_row_fraction": float(
            ((decision_span.any(dim=1)) & active_bool).float().sum().item() / active_count.item()
        ),
        "self_distillation/decision_span_token_fraction": float(
            (decision_span & selected_tokens).float().sum().item() / selected_count.item()
        ),
        "self_distillation/action_span_token_fraction": float(
            (action_span & selected_tokens).float().sum().item() / selected_count.item()
        ),
        "self_distillation/write_action_span_token_fraction": float(
            (write_action_span & selected_tokens).float().sum().item() / selected_count.item()
        ),
        "self_distillation/final_answer_span_token_fraction": float(
            (final_answer_span & selected_tokens).float().sum().item() / selected_count.item()
        ),
        "self_distillation/weighted_token_fraction": float(
            (weights * active_bool.unsqueeze(1).float()).sum().item() / selected_count.item()
        ),
        "self_distillation/mean_decision_weight": float(weights[selected_tokens].mean().item())
        if bool(selected_tokens.any().item())
        else 0.0,
        "self_distillation/max_decision_weight": float(weights.max().item()) if weights.numel() else 0.0,
        "self_distillation/short_unsolved_selected_fraction": float(
            (short_unsolved_guard & active_bool).float().sum().item() / active_count.item()
        ),
    }
    masks = {
        "decision_span": decision_span,
        "tool_action_span": action_span,
        "write_action_span": write_action_span,
        "final_answer_span": final_answer_span,
        "short_unsolved_guard": short_unsolved_guard,
    }
    return weights, masks, metrics
