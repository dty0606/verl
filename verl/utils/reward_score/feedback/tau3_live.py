from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any

from verl.utils.tau3_diagnostics import build_safe_diagnostic, compact_text, diagnostic_scalar_metrics
from verl.utils.tau3_feedback_renderers import normalize_feedback_mode, normalize_feedback_renderer
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

TRANSFER_TOOL_NAME = "transfer_to_human_agents"
TRANSFER_TEXT_MARKER_PATTERNS = [
    r"\btransfer_to_human_agents\b",
    r"\btransfer(?:red|ring)?\s+(?:you\s+)?to\s+(?:a\s+)?human\s+agent[s]?\b",
    r"\bconnect(?:ed|ing)?\s+(?:you\s+)?(?:with|to)\s+(?:a\s+)?human\s+agent[s]?\b",
    r"\bhuman\s+agent[s]?\s+(?:will|can)\s+(?:assist|help|take over)\b",
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _extract_live_result(extra_info: dict[str, Any] | None) -> dict[str, Any]:
    extra_info = extra_info or {}
    if isinstance(extra_info.get("tau3_live_result"), dict):
        return dict(extra_info["tau3_live_result"])

    tool_extra_fields = extra_info.get("tool_extra_fields") or {}
    if isinstance(tool_extra_fields, dict) and isinstance(tool_extra_fields.get("tau3_live_result"), dict):
        return dict(tool_extra_fields["tau3_live_result"])

    return {}


def _tool_name(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name")
    return str(value or "").strip()


def _executed_tool_names(live_result: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in live_result.get("executed_tools") or []:
        name = _tool_name(item)
        if name:
            names.append(name)
    if names:
        return names

    # Some summaries include richer tool events instead of executed_tools. Only
    # count successful events for strict reward evidence.
    for event in live_result.get("tool_events") or []:
        if not isinstance(event, dict) or event.get("success") is False:
            continue
        name = _tool_name(event)
        if name:
            names.append(name)
    return names


def _action_names_from_items(actions: Any) -> list[str]:
    if isinstance(actions, dict):
        actions = [actions]
    names: list[str] = []
    if not isinstance(actions, list):
        return names
    for action in actions:
        if isinstance(action, dict):
            requestor = str(action.get("requestor") or "assistant").strip().lower()
            if requestor and requestor != "assistant":
                continue
        name = _tool_name(action)
        if name:
            names.append(name)
    return names


def _expected_action_names(parsed_ground_truth: dict[str, Any], live_result: dict[str, Any]) -> list[str]:
    # Prefer canonical task annotations over runtime metadata. Runtime
    # expected_actions can be partial or malformed and should not suppress
    # evaluation_criteria.actions from the ground truth.
    for actions in (
        parsed_ground_truth.get("expected_actions"),
        (parsed_ground_truth.get("evaluation_criteria") or {}).get("actions")
        if isinstance(parsed_ground_truth.get("evaluation_criteria"), dict)
        else None,
        live_result.get("expected_actions"),
    ):
        names = _action_names_from_items(actions)
        if names:
            return names
    return []


def _has_transfer_text_marker(text: str) -> bool:
    return _contains_any_regex(_normalize(text), TRANSFER_TEXT_MARKER_PATTERNS)


def _strict_action_overlay_enabled(*, runtime: str) -> bool:
    return _env_flag("TAU3_STRICT_ACTION_REWARD", default=(runtime == "official_gym"))


def _strict_action_reward_qc(
    *,
    official_score: float,
    parsed_ground_truth: dict[str, Any],
    live_result: dict[str, Any],
    final_assistant: str,
    runtime: str,
) -> dict[str, float | bool | str]:
    enabled = _strict_action_overlay_enabled(runtime=runtime)
    executed_tools = _executed_tool_names(live_result)
    expected_actions = _expected_action_names(parsed_ground_truth, live_result)
    executed_counts: dict[str, int] = {}
    for name in executed_tools:
        executed_counts[name] = executed_counts.get(name, 0) + 1

    missing_expected: list[str] = []
    required_counts: dict[str, int] = {}
    for name in expected_actions:
        required_counts[name] = required_counts.get(name, 0) + 1
    for name, required_count in required_counts.items():
        executed_count = executed_counts.get(name, 0)
        if executed_count < required_count:
            missing_expected.extend([name] * (required_count - executed_count))

    transfer_marker_without_tool = _has_transfer_text_marker(final_assistant) and executed_counts.get(TRANSFER_TOOL_NAME, 0) <= 0
    official_success = float(official_score) > 0.0
    zero_tool_official_success = official_success and not executed_tools
    zero_tool_official_success_with_actions = zero_tool_official_success and bool(expected_actions)
    zero_tool_official_success_without_actions = zero_tool_official_success and not bool(expected_actions)

    strict_required = bool(expected_actions or transfer_marker_without_tool)
    strict_pass = (not missing_expected) and not transfer_marker_without_tool
    violation = enabled and strict_required and not strict_pass
    strict_score = 0.0 if violation else float(official_score)

    return {
        "enabled": enabled,
        "strict_score": strict_score,
        "strict_required": strict_required,
        "strict_pass": strict_pass,
        "violation": violation,
        "executed_action_count": float(len(executed_tools)),
        "expected_action_count": float(len(expected_actions)),
        "missing_expected_action_count": float(len(missing_expected)),
        "check_count": float(len(expected_actions) + (1 if transfer_marker_without_tool else 0)),
        "transfer_marker_without_tool": transfer_marker_without_tool,
        "official_success_zero_tool": zero_tool_official_success,
        "official_success_zero_tool_with_actions": zero_tool_official_success_with_actions,
        "official_success_zero_tool_without_actions": zero_tool_official_success_without_actions,
        "official_success_strict_override": official_success and violation,
        "high_trust_success": official_success and enabled and strict_required and strict_pass,
        "missing_expected_actions": ",".join(missing_expected),
        "executed_action_names_json": json.dumps(executed_tools, ensure_ascii=True, separators=(",", ":")),
        "expected_action_names_json": json.dumps(expected_actions, ensure_ascii=True, separators=(",", ":")),
        "missing_expected_action_names_json": json.dumps(missing_expected, ensure_ascii=True, separators=(",", ":")),
    }


def _is_terminal_live_result(live_result: dict[str, Any]) -> bool:
    status = str(live_result.get("status", "") or "").lower()
    terminal_reason = str(live_result.get("terminal_reason", "") or "").lower()
    return status == "terminated" or terminal_reason not in {"", "running"}


def _is_env_error_live_result(live_result: dict[str, Any]) -> bool:
    terminal_reason = str(live_result.get("terminal_reason", "") or "").lower()
    return bool(live_result.get("env_error")) or terminal_reason in {"env_error", "environment_error"}


def _best_available_reward(
    *,
    parsed_ground_truth: dict[str, Any],
    live_result: dict[str, Any],
    extra_info: dict[str, Any] | None,
    final_assistant: str,
) -> tuple[float, str]:
    """Return official tau reward when present; fall back only for legacy proxy runs."""

    extra_info = extra_info or {}

    if _is_env_error_live_result(live_result):
        return 0.0, "official_gym_env_error"

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
        or "none"
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
    extra_info = extra_info or {}

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
    # The faithful Tau3 SDPO baseline must not inject our deterministic
    # diagnostic interpretation into the teacher prompt. Keep diagnostics as
    # scalar audit metrics only; any future real environment-feedback path
    # should use a separate explicit reward key rather than this heuristic field.
    feedback = ""

    runtime = str(live_result.get("runtime") or extra_info.get("tau3_runtime") or "").lower()
    strict_qc = _strict_action_reward_qc(
        official_score=float(reward),
        parsed_ground_truth=parsed_ground_truth,
        live_result=live_result,
        final_assistant=final_assistant,
        runtime=runtime,
    )
    training_reward = float(strict_qc["strict_score"])
    terminal = float(_is_terminal_live_result(live_result))
    terminal_reason = str(live_result.get("terminal_reason") or "").lower()
    env_error = float(_is_env_error_live_result(live_result))
    bedrock_error = float(bool(live_result.get("bedrock_error")))
    budget_exhausted = float(
        terminal_reason in {"truncated", "max_steps", "max_user_turns"}
        or reward_source == "official_gym_nonterminal_snapshot"
        or diagnostic.primary_failure_type == "length_or_step_budget_failure"
    )
    turn_count = float(live_result.get("turn_count") or 0.0)
    tool_count = float(len(live_result.get("executed_tools") or []))
    official_reward_path_violation = float(
        runtime == "official_gym"
        and _is_terminal_live_result(live_result)
        and reward_source not in {"official_or_runtime", "official_gym_env_error"}
    )
    if env_error:
        feedback = ""

    result = {
        "score": training_reward,
        "acc": training_reward,
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
        "tau3_live/reward_source_env_error_fraction": 1.0 if reward_source == "official_gym_env_error" else 0.0,
        "tau3_live/reward_source_proxy_fallback_fraction": 1.0 if reward_source == "legacy_proxy_heuristic" else 0.0,
        "tau3_live/reward_source_official_violation_fraction": official_reward_path_violation,
        "tau3_live/terminal_fraction": terminal,
        "tau3_live/nonterminal_fraction": 1.0 - terminal,
        "tau3_live/budget_exhausted_fraction": budget_exhausted,
        "tau3_live/env_error_fraction": env_error,
        "tau3_live/bedrock_error_fraction": bedrock_error,
        "tau3_live/bedrock_retry_count": float(live_result.get("bedrock_retry_count") or 0.0),
        "tau3_live/bedrock_fallback_fraction": float(bool(live_result.get("bedrock_fallback_model"))),
        "tau3_live/turn_count": turn_count,
        "tau3_live/tool_count": tool_count,
        "official_score": float(reward),
        "strict_score": training_reward,
        "strict_action_overlay_enabled_fraction": float(bool(strict_qc["enabled"])),
        "strict_action_required_fraction": float(bool(strict_qc["strict_required"])),
        "strict_action_pass_fraction": float(bool(strict_qc["strict_pass"])),
        "strict_action_violation_fraction": float(bool(strict_qc["violation"])),
        "strict_action_check_count": float(strict_qc["check_count"]),
        "strict_executed_action_count": float(strict_qc["executed_action_count"]),
        "strict_expected_action_count": float(strict_qc["expected_action_count"]),
        "strict_missing_expected_action_count": float(strict_qc["missing_expected_action_count"]),
        "transfer_marker_without_tool_fraction": float(bool(strict_qc["transfer_marker_without_tool"])),
        "official_success_zero_tool_fraction": float(bool(strict_qc["official_success_zero_tool"])),
        "official_success_zero_tool_with_actions_fraction": float(
            bool(strict_qc["official_success_zero_tool_with_actions"])
        ),
        "official_success_zero_tool_without_actions_fraction": float(
            bool(strict_qc["official_success_zero_tool_without_actions"])
        ),
        "official_success_strict_override_fraction": float(bool(strict_qc["official_success_strict_override"])),
        "tau3_live/official_score": float(reward),
        "tau3_live/strict_score": training_reward,
        "tau3_live/strict_action_overlay_enabled_fraction": float(bool(strict_qc["enabled"])),
        "tau3_live/strict_action_required_fraction": float(bool(strict_qc["strict_required"])),
        "tau3_live/strict_action_pass_fraction": float(bool(strict_qc["strict_pass"])),
        "tau3_live/strict_action_violation_fraction": float(bool(strict_qc["violation"])),
        "tau3_live/strict_action_check_count": float(strict_qc["check_count"]),
        "tau3_live/strict_executed_action_count": float(strict_qc["executed_action_count"]),
        "tau3_live/strict_expected_action_count": float(strict_qc["expected_action_count"]),
        "tau3_live/strict_missing_expected_action_count": float(strict_qc["missing_expected_action_count"]),
        "tau3_live/transfer_marker_without_tool_fraction": float(bool(strict_qc["transfer_marker_without_tool"])),
        "tau3_live/official_success_zero_tool_fraction": float(bool(strict_qc["official_success_zero_tool"])),
        "tau3_live/official_success_zero_tool_with_actions_fraction": float(
            bool(strict_qc["official_success_zero_tool_with_actions"])
        ),
        "tau3_live/official_success_zero_tool_without_actions_fraction": float(
            bool(strict_qc["official_success_zero_tool_without_actions"])
        ),
        "tau3_live/official_success_strict_override_fraction": float(
            bool(strict_qc["official_success_strict_override"])
        ),
        "tau3_live/strict_action_high_trust_success_fraction": float(bool(strict_qc["high_trust_success"])),
        "tau3_action_executed_names_json": str(strict_qc["executed_action_names_json"]),
        "tau3_action_expected_names_json": str(strict_qc["expected_action_names_json"]),
        "tau3_action_missing_names_json": str(strict_qc["missing_expected_action_names_json"]),
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
