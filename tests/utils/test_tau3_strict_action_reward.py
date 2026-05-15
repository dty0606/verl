from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_tau3_live_reward():
    root = Path(__file__).resolve().parents[2]
    sys.modules.setdefault("verl", types.ModuleType("verl"))
    sys.modules.setdefault("verl.utils", types.ModuleType("verl.utils"))
    sys.modules.setdefault("verl.utils.reward_score", types.ModuleType("verl.utils.reward_score"))
    sys.modules.setdefault("verl.utils.reward_score.feedback", types.ModuleType("verl.utils.reward_score.feedback"))

    _load_module("verl.utils.tau3_diagnostics", root / "verl" / "utils" / "tau3_diagnostics.py")
    _load_module("verl.utils.tau3_feedback_renderers", root / "verl" / "utils" / "tau3_feedback_renderers.py")
    _load_module("verl.utils.tau3_live_runtime", root / "verl" / "utils" / "tau3_live_runtime.py")
    return _load_module(
        "verl.utils.reward_score.feedback.tau3_live",
        root / "verl" / "utils" / "reward_score" / "feedback" / "tau3_live.py",
    )


tau3_live = _load_tau3_live_reward()


def _score(*, final_reward: float = 1.0, executed_tools: list[str] | None = None, final_text: str = "Done"):
    return tau3_live.compute_score(
        solution_str=final_text,
        ground_truth={"label": "success", "expected_actions": [{"name": "transfer_to_human_agents"}]},
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": final_reward,
                "executed_tools": executed_tools or [],
                "latest_assistant_message": final_text,
            }
        },
    )


def test_strict_action_overlay_zeroes_official_success_when_expected_action_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = _score(executed_tools=[], final_text="I will connect you with a human agent.")

    assert result["score"] == 0.0
    assert result["acc"] == 0.0
    assert result["tau3_live/official_score"] == 1.0
    assert result["tau3_live/strict_score"] == 0.0
    assert result["tau3_live/strict_action_overlay_enabled_fraction"] == 1.0
    assert result["tau3_live/strict_action_required_fraction"] == 1.0
    assert result["tau3_live/strict_action_violation_fraction"] == 1.0
    assert result["tau3_live/official_success_strict_override_fraction"] == 1.0
    assert result["feedback"] == ""


def test_strict_action_overlay_preserves_official_success_when_expected_action_executed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = _score(executed_tools=["transfer_to_human_agents"], final_text="Transferred.")

    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["tau3_live/strict_action_pass_fraction"] == 1.0
    assert result["tau3_live/strict_action_violation_fraction"] == 0.0
    assert result["tau3_live/strict_expected_action_count"] == 1.0
    assert result["tau3_live/strict_action_high_trust_success_fraction"] == 1.0


def test_fake_transfer_marker_without_transfer_tool_is_strict_violation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="Calling transfer_to_human_agents now.",
        ground_truth={"label": "success"},
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "executed_tools": [],
                "latest_assistant_message": "Calling transfer_to_human_agents now.",
            }
        },
    )

    assert result["score"] == 0.0
    assert result["tau3_live/transfer_marker_without_tool_fraction"] == 1.0
    assert result["tau3_live/strict_action_check_count"] == 1.0
    assert result["tau3_live/official_success_zero_tool_fraction"] == 1.0
    assert result["tau3_live/official_success_zero_tool_without_actions_fraction"] == 1.0


def test_fake_connect_human_agent_marker_is_strict_violation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="I will connect you with a human agent now.",
        ground_truth={"label": "success"},
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "executed_tools": [],
                "latest_assistant_message": "I will connect you with a human agent now.",
            }
        },
    )

    assert result["score"] == 0.0
    assert result["feedback"] == ""
    assert result["tau3_live/transfer_marker_without_tool_fraction"] == 1.0


def test_official_success_without_action_requirement_or_tools_is_preserved(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="Done.",
        ground_truth={"label": "success"},
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "executed_tools": [],
                "latest_assistant_message": "Done.",
            }
        },
    )

    assert result["score"] == 1.0
    assert result["tau3_live/strict_action_required_fraction"] == 0.0
    assert result["tau3_live/strict_action_pass_fraction"] == 1.0
    assert result["tau3_live/strict_action_violation_fraction"] == 0.0
    assert result["tau3_live/official_success_zero_tool_fraction"] == 1.0
    assert result["tau3_live/official_success_zero_tool_with_actions_fraction"] == 0.0
    assert result["tau3_live/official_success_zero_tool_without_actions_fraction"] == 1.0
    assert result["tau3_live/strict_action_high_trust_success_fraction"] == 0.0


def test_evaluation_criteria_actions_are_required_for_strict_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="I checked this for you.",
        ground_truth={
            "label": "success",
            "evaluation_criteria": {
                "actions": [
                    {
                        "action_id": "47_0",
                        "requestor": "assistant",
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": "H8Q05L"},
                    }
                ]
            },
        },
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "executed_tools": [],
                "latest_assistant_message": "I checked this for you.",
            }
        },
    )

    assert result["score"] == 0.0
    assert result["tau3_live/strict_expected_action_count"] == 1.0
    assert result["tau3_live/strict_action_required_fraction"] == 1.0
    assert result["tau3_live/strict_action_violation_fraction"] == 1.0
    assert result["tau3_live/official_success_zero_tool_with_actions_fraction"] == 1.0
    assert result["tau3_live/official_success_zero_tool_without_actions_fraction"] == 0.0


def test_evaluation_criteria_actions_pass_when_required_tool_executed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="I checked this for you.",
        ground_truth={
            "label": "success",
            "evaluation_criteria": {
                "actions": [
                    {
                        "action_id": "47_0",
                        "requestor": "assistant",
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": "H8Q05L"},
                    }
                ]
            },
        },
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "executed_tools": ["get_reservation_details"],
                "latest_assistant_message": "I checked this for you.",
            }
        },
    )

    assert result["score"] == 1.0
    assert result["tau3_live/strict_expected_action_count"] == 1.0
    assert result["tau3_live/strict_action_violation_fraction"] == 0.0


def test_malformed_expected_actions_do_not_suppress_evaluation_criteria_actions(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="I checked this for you.",
        ground_truth={
            "label": "success",
            "expected_actions": [{"bad": "metadata-without-name"}],
            "evaluation_criteria": {
                "actions": [{"requestor": "assistant", "name": "get_reservation_details"}]
            },
        },
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "executed_tools": [],
                "latest_assistant_message": "I checked this for you.",
            }
        },
    )

    assert result["score"] == 0.0
    assert result["tau3_live/strict_expected_action_count"] == 1.0
    assert result["tau3_live/strict_action_violation_fraction"] == 1.0


def test_live_result_expected_actions_used_when_ground_truth_has_no_actions(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="I checked this for you.",
        ground_truth={"label": "success"},
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "expected_actions": [{"name": "get_user_details"}],
                "executed_tools": [],
                "latest_assistant_message": "I checked this for you.",
            }
        },
    )

    assert result["score"] == 0.0
    assert result["tau3_live/strict_expected_action_count"] == 1.0
    assert result["tau3_live/strict_action_violation_fraction"] == 1.0


def test_non_assistant_evaluation_actions_are_ignored(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="Done.",
        ground_truth={
            "label": "success",
            "evaluation_criteria": {
                "actions": [{"requestor": "user", "name": "some_user_action"}]
            },
        },
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "executed_tools": [],
                "latest_assistant_message": "Done.",
            }
        },
    )

    assert result["score"] == 1.0
    assert result["tau3_live/strict_expected_action_count"] == 0.0
    assert result["tau3_live/strict_action_required_fraction"] == 0.0


def test_transfer_marker_with_actual_transfer_tool_passes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="I will connect you with a human agent now.",
        ground_truth={"label": "success"},
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "executed_tools": ["transfer_to_human_agents"],
                "latest_assistant_message": "I will connect you with a human agent now.",
            }
        },
    )

    assert result["score"] == 1.0
    assert result["tau3_live/transfer_marker_without_tool_fraction"] == 0.0
    assert result["tau3_live/strict_action_violation_fraction"] == 0.0


def test_duplicate_expected_actions_require_duplicate_tool_executions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="Done.",
        ground_truth={
            "label": "success",
            "expected_actions": [{"name": "get_reservation_details"}, {"name": "get_reservation_details"}],
        },
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "executed_tools": ["get_reservation_details"],
                "latest_assistant_message": "Done.",
            }
        },
    )

    assert result["score"] == 0.0
    assert result["tau3_live/strict_expected_action_count"] == 2.0
    assert result["tau3_live/strict_action_violation_fraction"] == 1.0


def test_duplicate_expected_actions_pass_with_duplicate_tool_executions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)

    result = tau3_live.compute_score(
        solution_str="Done.",
        ground_truth={
            "label": "success",
            "expected_actions": [{"name": "get_reservation_details"}, {"name": "get_reservation_details"}],
        },
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
                "executed_tools": ["get_reservation_details", "get_reservation_details"],
                "latest_assistant_message": "Done.",
            }
        },
    )

    assert result["score"] == 1.0
    assert result["tau3_live/strict_expected_action_count"] == 2.0
    assert result["tau3_live/strict_action_violation_fraction"] == 0.0


def test_strict_action_reward_env_toggle_controls_official_gym_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_STRICT_ACTION_REWARD", raising=False)
    assert _score(executed_tools=[])["score"] == 0.0

    monkeypatch.setenv("TAU3_STRICT_ACTION_REWARD", "0")
    disabled = _score(executed_tools=[])

    assert disabled["score"] == 1.0
    assert disabled["tau3_live/official_score"] == 1.0
    assert disabled["tau3_live/strict_score"] == 1.0
    assert disabled["tau3_live/strict_action_overlay_enabled_fraction"] == 0.0
    assert disabled["tau3_live/strict_action_violation_fraction"] == 0.0
