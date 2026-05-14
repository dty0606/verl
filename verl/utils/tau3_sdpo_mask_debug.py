from __future__ import annotations

from typing import Any, Optional


def compact_preview(text: Any, *, limit: int = 160) -> str:
    """Return a bounded, whitespace-normalized preview for debug logs."""

    limit = max(0, int(limit))
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: limit - 3] + "..."


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_batch(value: Any, *, batch_size: int, default: Any = None) -> list[Any]:
    if value is None:
        return [default] * batch_size
    value = _to_python(value)
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [value] + [default] * max(0, batch_size - 1)
    out = value[:batch_size]
    if len(out) < batch_size:
        out.extend([default] * (batch_size - len(out)))
    return out


def _as_bool_row(value: Any) -> list[bool]:
    value = _to_python(value)
    if value is None:
        return []
    if isinstance(value, (int, float, bool)):
        return [bool(value)]
    return [bool(item) for item in list(value)]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extra_value(reward_extra_infos_dict: Optional[dict[str, list]], key: str, row_index: int) -> Any:
    values = (reward_extra_infos_dict or {}).get(key)
    values = _to_python(values)
    if isinstance(values, list) and row_index < len(values):
        return values[row_index]
    return None


def _first_extra_value(
    reward_extra_infos_dict: Optional[dict[str, list]],
    keys: tuple[str, ...],
    row_index: int,
) -> Any:
    for key in keys:
        value = _extra_value(reward_extra_infos_dict, key, row_index)
        if value is not None:
            return value
    return None


def _selected_mask_for_row(
    *,
    response_row: list[bool],
    loss_row: list[bool],
    target_row: list[bool],
) -> list[bool]:
    response_len = len(response_row)
    if target_row:
        selected = target_row
    elif loss_row:
        selected = [loss and response for loss, response in zip(loss_row, response_row, strict=False)]
    else:
        selected = response_row
    return list(selected[:response_len])


def _contiguous_spans(mask: list[bool]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, selected in enumerate(mask):
        if selected and start is None:
            start = idx
        elif not selected and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _token_text_span(response_token_texts: list[str], start: int, end: int) -> str:
    if not response_token_texts:
        return ""
    return "".join(str(token) for token in response_token_texts[start:end])


def build_tau3_sdpo_mask_debug_rows(
    *,
    prompt_texts: list[str],
    response_texts: list[str],
    response_mask: Any,
    self_distillation_loss_mask: Any = None,
    self_distillation_target_token_mask: Any = None,
    reward_tensor: Any = None,
    reward_extra_infos_dict: Optional[dict[str, list]] = None,
    uids: Any = None,
    task_ids: Any = None,
    response_token_texts: Optional[list[list[str]]] = None,
    prompt_token_counts: Any = None,
    preview_chars: int = 160,
    max_spans: int = 8,
) -> list[dict[str, Any]]:
    """Build compact JSON-serializable SDPO mask debug rows.

    Masks are interpreted as response-token masks. When ``prompt_token_counts`` is
    supplied, span prediction positions use the shifted full-sequence log-prob
    convention: first response token is predicted at ``prompt_token_count - 1``.
    No prompt-token rows are emitted.
    """

    batch_size = max(len(prompt_texts), len(response_texts))
    prompt_batch = _as_batch(prompt_texts, batch_size=batch_size, default="")
    response_batch = _as_batch(response_texts, batch_size=batch_size, default="")
    response_mask_batch = _as_batch(response_mask, batch_size=batch_size, default=[])
    loss_mask_batch = _as_batch(self_distillation_loss_mask, batch_size=batch_size, default=[])
    target_mask_batch = _as_batch(self_distillation_target_token_mask, batch_size=batch_size, default=[])
    reward_batch = _as_batch(reward_tensor, batch_size=batch_size, default=None)
    uid_batch = _as_batch(uids, batch_size=batch_size, default=None)
    task_id_batch = _as_batch(task_ids, batch_size=batch_size, default=None)
    prompt_token_count_batch = _as_batch(prompt_token_counts, batch_size=batch_size, default=None)
    token_text_batch = _as_batch(response_token_texts, batch_size=batch_size, default=[])

    rows: list[dict[str, Any]] = []
    for row_index in range(batch_size):
        response_row = _as_bool_row(response_mask_batch[row_index])
        loss_row = _as_bool_row(loss_mask_batch[row_index])
        target_row = _as_bool_row(target_mask_batch[row_index])
        selected_mask = _selected_mask_for_row(
            response_row=response_row,
            loss_row=loss_row,
            target_row=target_row,
        )
        response_token_count = int(sum(response_row))
        selected_token_count = int(sum(1 for selected in selected_mask if selected))
        selected_fraction = (
            float(selected_token_count / response_token_count) if response_token_count > 0 else 0.0
        )

        prompt_token_count = prompt_token_count_batch[row_index]
        prompt_token_count_float = _safe_float(prompt_token_count)
        if prompt_token_count_float is None:
            prediction_offset = 0
            prediction_basis = "response_relative"
            prompt_token_count_json = None
        else:
            prediction_offset = max(0, int(prompt_token_count_float) - 1)
            prediction_basis = "full_sequence_shifted"
            prompt_token_count_json = int(prompt_token_count_float)

        row_token_texts = token_text_batch[row_index] or []
        spans = []
        selected_text_parts: list[str] = []
        for start, end in _contiguous_spans(selected_mask)[: max(0, int(max_spans))]:
            span_text = _token_text_span(row_token_texts, start, end)
            if span_text:
                selected_text_parts.append(span_text)
            spans.append(
                {
                    "response_token_start": start,
                    "response_token_end": end,
                    "prediction_position_start": prediction_offset + start,
                    "prediction_position_end": prediction_offset + end,
                    "text_preview": compact_preview(span_text, limit=preview_chars),
                }
            )

        selected_text = "".join(selected_text_parts)
        if not selected_text and selected_token_count == response_token_count and selected_token_count > 0:
            selected_text = str(response_batch[row_index] or "")

        reward_value = reward_batch[row_index]
        if isinstance(reward_value, list):
            reward_value = sum(_safe_float(item) or 0.0 for item in reward_value)
        reward_value = _safe_float(reward_value)

        official_score = _safe_float(
            _first_extra_value(
                reward_extra_infos_dict,
                ("official_score", "official/score", "score_official", "tau3_live/official_score"),
                row_index,
            )
        )
        strict_score = _safe_float(
            _first_extra_value(
                reward_extra_infos_dict,
                ("strict_score", "strict/score", "score_strict", "tau3_live/strict_score"),
                row_index,
            )
        )
        uid = uid_batch[row_index]
        if uid is None:
            uid = _extra_value(reward_extra_infos_dict, "uid", row_index)
        task_id = task_id_batch[row_index]
        if task_id is None:
            task_id = _extra_value(reward_extra_infos_dict, "task_id", row_index)

        rows.append(
            {
                "row_index": row_index,
                "uid": uid,
                "task_id": task_id,
                "reward": reward_value,
                "reward_source": _extra_value(reward_extra_infos_dict, "reward_source", row_index),
                "official_score": official_score,
                "strict_score": strict_score,
                "prompt_token_count": prompt_token_count_json,
                "prediction_position_basis": prediction_basis,
                "response_token_count": response_token_count,
                "selected_token_count": selected_token_count,
                "selected_token_fraction": selected_fraction,
                "response_text_preview": compact_preview(response_batch[row_index], limit=preview_chars),
                "selected_text_preview": compact_preview(selected_text, limit=preview_chars),
                "selected_spans": spans,
            }
        )
    return rows
