"""Pure helpers for SDPO same-UID rollout group diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(values: Iterable[Any] | None, length: int) -> list[Any]:
    if values is None:
        return [None] * length
    out = list(values)
    if len(out) < length:
        out.extend([None] * (length - len(out)))
    return out[:length]


def _summarize_groups(
    grouped_scores: dict[Any, list[float]],
    *,
    threshold: float,
) -> dict[str, float]:
    group_count = len(grouped_scores)
    all_fail = 0
    all_success = 0
    mixed = 0
    for scores in grouped_scores.values():
        if not scores:
            continue
        success_count = sum(score >= threshold for score in scores)
        if success_count == 0:
            all_fail += 1
        elif success_count == len(scores):
            all_success += 1
        else:
            mixed += 1

    denom = float(group_count) if group_count else 1.0
    return {
        "group_count": float(group_count),
        "all_fail_group_count": float(all_fail),
        "all_success_group_count": float(all_success),
        "mixed_group_count": float(mixed),
        "all_fail_fraction": float(all_fail / denom),
        "all_success_fraction": float(all_success / denom),
        "mixed_fraction": float(mixed / denom),
    }


def sdpo_same_uid_group_metrics(
    *,
    uids: Iterable[Any] | None,
    strict_scores: Iterable[Any],
    official_scores: Iterable[Any] | None = None,
    eligible_mask: Iterable[Any] | None = None,
    success_threshold: float = 1.0,
    prefix: str = "self_distillation",
) -> dict[str, float]:
    """Return all-fail/all-success/mixed same-UID metrics.

    The strict-score view is the training view. The official-score view is kept
    beside it to audit reward-overlay effects without changing training.
    """

    strict_values = list(strict_scores)
    length = len(strict_values)
    uid_values = _as_list(uids, length)
    official_values = _as_list(official_scores, length) if official_scores is not None else None
    eligible_values = _as_list(eligible_mask, length)

    strict_groups: dict[Any, list[float]] = defaultdict(list)
    official_groups: dict[Any, list[float]] = defaultdict(list)
    missing_uid = 0
    ineligible = 0

    for idx in range(length):
        if eligible_mask is not None and not bool(eligible_values[idx]):
            ineligible += 1
            continue

        uid = uid_values[idx]
        if uid is None or str(uid) == "":
            missing_uid += 1
            continue

        strict_score = _coerce_float(strict_values[idx])
        if strict_score is not None:
            strict_groups[uid].append(strict_score)

        if official_values is not None:
            official_score = _coerce_float(official_values[idx])
            if official_score is not None:
                official_groups[uid].append(official_score)

    strict_summary = _summarize_groups(strict_groups, threshold=success_threshold)
    metrics = {
        f"{prefix}/same_uid_group_count": strict_summary["group_count"],
        f"{prefix}/same_uid_missing_uid_fraction": float(missing_uid / length) if length else 0.0,
        f"{prefix}/same_uid_ineligible_fraction": float(ineligible / length) if length else 0.0,
    }
    for key, value in strict_summary.items():
        metrics[f"{prefix}/same_uid_strict_{key}"] = value

    # Backward-readable aliases: unqualified same_uid_* means strict/training view.
    for key in (
        "all_fail_group_count",
        "all_success_group_count",
        "mixed_group_count",
        "all_fail_fraction",
        "all_success_fraction",
        "mixed_fraction",
    ):
        metrics[f"{prefix}/same_uid_{key}"] = strict_summary[key]

    if official_values is not None:
        official_summary = _summarize_groups(official_groups, threshold=success_threshold)
        for key, value in official_summary.items():
            metrics[f"{prefix}/same_uid_official_{key}"] = value

    return metrics


def format_sdpo_same_uid_warning(
    *,
    step: int,
    metrics: dict[str, float],
    prefix: str = "self_distillation",
) -> str | None:
    group_count = int(metrics.get(f"{prefix}/same_uid_group_count", 0.0))
    mixed_count = int(metrics.get(f"{prefix}/same_uid_strict_mixed_group_count", 0.0))
    if group_count <= 0 or mixed_count > 0:
        return None

    all_success = int(metrics.get(f"{prefix}/same_uid_strict_all_success_group_count", 0.0))
    all_fail = int(metrics.get(f"{prefix}/same_uid_strict_all_fail_group_count", 0.0))
    missing_uid = metrics.get(f"{prefix}/same_uid_missing_uid_fraction", 0.0)
    return (
        "[sdpo_same_uid_warning] no_mixed_strict_groups "
        f"step={step} groups={group_count} all_success={all_success} "
        f"all_fail={all_fail} missing_uid_fraction={missing_uid:.6g}"
    )
