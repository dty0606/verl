from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any

from verl.utils.tau3_diagnostics import build_safe_diagnostic, compact_text, diagnostic_scalar_metrics
from verl.utils.tau3_feedback_renderers import normalize_feedback_mode, normalize_feedback_renderer, render_feedback
from verl.utils.tau3_live_runtime import compute_live_task_reward


GENERIC_REFUSAL_PATTERNS = [
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bunable\b",
    r"\bnot allowed\b",
    r"\bnot eligible\b",
    r"\bnot covered\b",
    r"\bcan not\b",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_any_regex(text: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern and re.search(pattern, text):
            return True
    return False


def _parse_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            return {"label": ground_truth}
    if isinstance(ground_truth, dict):
        return ground_truth
    return {}


def _extract_live_result(extra_info: dict[str, Any] | None) -> dict[str, Any]:
    extra_info = extra_info or {}
    if isinstance(extra_info.get("tau3_live_result"), dict):
        return dict(extra_info["tau3_live_result"])

    tool_extra_fields = extra_info.get("tool_extra_fields") or {}
    if isinstance(tool_extra_fields, dict) and isinstance(tool_extra_fields.get("tau3_live_result"), dict):
        return dict(tool_extra_fields["tau3_live_result"])

    return {}


def _is_terminal_live_result(live_result: dict[str, Any]) -> bool:
    status = str(live_result.get("status", "") or "").lower()
    terminal_reason = str(live_result.get("terminal_reason", "") or "").lower()
    return status == "terminated" or terminal_reason not in {"", "running"}


def _best_available_reward(
    *,
    parsed_ground_truth: dict[str, Any],
    live_result: dict[str, Any],
    extra_info: dict[str, Any] | None,
    final_assistant: str,
) -> tuple[float, str]:
    """Return official tau reward when present; fall back only for legacy proxy runs."""

    extra_info = extra_info or {}

    if _is_terminal_live_result(live_result):
        for key in ("final_reward", "reward", "score"):
            if live_result.get(key) is not None:
                return float(live_result[key]), "official_or_runtime"

    runtime = str(live_result.get("runtime") or extra_info.get("tau3_runtime") or "").lower()
    if runtime == "official_gym":
        if _is_terminal_live_result(live_result):
            return 0.0, "official_gym_missing_final_reward"
        return 0.0, "official_gym_nonterminal_snapshot"

    for key in ("final_reward", "reward", "score"):
        if live_result.get(key) is not None:
            return float(live_result[key]), "official_or_runtime"

    turn_scores = extra_info.get("turn_scores") or []
    tool_rewards = extra_info.get("tool_rewards") or []
    numeric_scores = []
    for value in list(turn_scores) + list(tool_rewards):
        try:
            numeric_scores.append(float(value))
        except Exception:
            pass
    if numeric_scores:
        return max(numeric_scores), "propagated_turn_or_tool_reward"

    executed_tools = list(live_result.get("executed_tools") or [])
    return (
        compute_live_task_reward(
            task_payload=parsed_ground_truth,
            executed_tools=executed_tools,
            final_assistant_text=final_assistant,
        ),
        "legacy_proxy_heuristic",
    )


def _build_pred(parsed_ground_truth: dict[str, Any], live_result: dict[str, Any], final_assistant: str) -> str:
    executed_tools = list(live_result.get("executed_tools") or [])
    final_text = _normalize(final_assistant)
    label = str(parsed_ground_truth.get("label", "") or "").strip().lower()
    if label == "refuse":
        return "refuse" if _contains_any_regex(final_text, GENERIC_REFUSAL_PATTERNS) else "other"
    if executed_tools:
        return executed_tools[-1]
    return "text_only"


def _feedback_mode(extra_info: dict[str, Any] | None) -> str:
    extra_info = extra_info or {}
    return normalize_feedback_mode(
        os.environ.get("TAU3_LIVE_FEEDBACK_FORMAT")
        or extra_info.get("teacher_feedback_format")
        or extra_info.get("feedback_mode")
        or "json"
    )


def _feedback_renderer(extra_info: dict[str, Any] | None) -> str:
    extra_info = extra_info or {}
    return normalize_feedback_renderer(
        extra_info.get("teacher_feedback_renderer")
        or extra_info.get("feedback_renderer")
        or os.environ.get("TAU3_FEEDBACK_RENDERER", "diagnostic")
    )


def compute_score(solution_str: str | None = None, ground_truth: Any = None, extra_info: dict | None = None, **kwargs) -> dict[str, Any]:
    """Reward-score entry point for live tau SDPO.

    The main score should be official tau AgentGymEnv/evaluator final reward
    when available. The old heuristic is retained only as a legacy fallback for
    proxy_legacy smoke runs.
    """

    if solution_str is None:
        solution_str = kwargs.get("solution", "")

    parsed_ground_truth = _parse_ground_truth(ground_truth)
    live_result = _extract_live_result(extra_info)

    final_assistant = str(
        (live_result.get("latest_assistant_message") if _is_terminal_live_result(live_result) else solution_str)
        or solution_str
        or ""
    )
    reward, reward_source = _best_available_reward(
        parsed_ground_truth=parsed_ground_truth,
        live_result=live_result,
        extra_info=extra_info,
        final_assistant=final_assistant,
    )

    live_result.setdefault("final_reward", float(reward))
    live_result.setdefault("latest_assistant_message", final_assistant)

    pred = _build_pred(parsed_ground_truth, live_result, final_assistant)
    diagnostic = build_safe_diagnostic(live_result)
    diagnostic_metrics = diagnostic_scalar_metrics(diagnostic)
    mode = _feedback_mode(extra_info)
    renderer = _feedback_renderer(extra_info)
    feedback = "" if reward >= 1.0 else render_feedback(diagnostic, mode, renderer)
    # Ensure feedback is always a string for downstream aggregators. The JSON
    # renderer returns a dict; serialize it before emitting.
    if isinstance(feedback, dict):
        feedback = json.dumps(feedback, ensure_ascii=False, sort_keys=True)

    runtime = str(live_result.get("runtime") or extra_info.get("tau3_runtime") or "").lower()
    terminal = float(_is_terminal_live_result(live_result))
    terminal_reason = str(live_result.get("terminal_reason") or "").lower()
    budget_exhausted = float(
        terminal_reason in {"truncated", "max_steps", "max_user_turns"}
        or reward_source == "official_gym_nonterminal_snapshot"
    )
    turn_count = float(live_result.get("turn_count") or 0.0)
    tool_count = float(len(live_result.get("executed_tools") or []))
    official_reward_path_violation = float(
        runtime == "official_gym" and _is_terminal_live_result(live_result) and reward_source != "official_or_runtime"
    )

    result = {
        "score": float(reward),
        "acc": float(reward),
        "pred": pred,
        "incorrect_format": int(bool(live_result.get("last_parse_error"))),
        "feedback": feedback,
        "feedback_mode": mode,
        "feedback_renderer": renderer,
        "reward_source": reward_source,
        "tau3_live/reward_source_official_fraction": 1.0 if reward_source == "official_or_runtime" else 0.0,
        "tau3_live/reward_source_nonterminal_fraction": (
            1.0 if reward_source == "official_gym_nonterminal_snapshot" else 0.0
        ),
        "tau3_live/reward_source_proxy_fallback_fraction": 1.0 if reward_source == "legacy_proxy_heuristic" else 0.0,
        "tau3_live/reward_source_official_violation_fraction": official_reward_path_violation,
        "tau3_live/terminal_fraction": terminal,
        "tau3_live/nonterminal_fraction": 1.0 - terminal,
        "tau3_live/budget_exhausted_fraction": budget_exhausted,
        "tau3_live/turn_count": turn_count,
        "tau3_live/tool_count": tool_count,
    }
    result.update(diagnostic_metrics)

    # Attach non-scalar debug fields only if raw artifacts are enabled.
    # The verl validation aggregator cannot np.mean dicts, so we keep these
    # off the main result by default.
    if os.environ.get("TAU3_LIVE_INCLUDE_RAW_EVAL_ARTIFACTS", "0") == "1":
        result["diagnostic"] = asdict(diagnostic)
        result["proxy_reward_debug"] = {
            "legacy_label": str(parsed_ground_truth.get("label", "") or ""),
            "executed_tools": list(live_result.get("executed_tools") or []),
            "final_assistant_preview": compact_text(final_assistant),
        }
        result["reward_info_json"] = live_result.get("reward_info_json") or live_result.get("reward_info")
        result["simulation_run_json"] = live_result.get("simulation_run_json") or live_result.get("simulation_run")

    return result
