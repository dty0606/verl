from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_tau3_live_runtime():
    module_path = Path(__file__).resolve().parents[2] / "verl" / "utils" / "tau3_live_runtime.py"
    spec = importlib.util.spec_from_file_location("tau3_live_runtime_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tau3_live_runtime = _load_tau3_live_runtime()
classify_tau3_env_error_payload = tau3_live_runtime.classify_tau3_env_error_payload
tau3_bedrock_retry_delays = tau3_live_runtime.tau3_bedrock_retry_delays


def test_classifies_bedrock_service_unavailable_from_simulation_payload():
    payload = {
        "error": (
            "Simulation loop exited with an exception: litellm.ServiceUnavailableError: "
            "Bedrock is unable to process your request."
        )
    }

    result = classify_tau3_env_error_payload(payload)

    assert result["env_error"] is True
    assert result["bedrock_error"] is True
    assert result["env_error_type"] == "bedrock_transient"


def test_traceback_key_without_error_text_is_not_env_error():
    payload = {"traceback": None, "termination_reason": "user_stop", "reward": 1.0}

    result = classify_tau3_env_error_payload(payload)

    assert result["env_error"] is False
    assert result["bedrock_error"] is False


def test_env_error_reward_is_zero_and_feedback_is_suppressed():
    pytest.importorskip("ray")
    from verl.utils.reward_score.feedback.tau3_live import compute_score

    result = compute_score(
        solution_str="<think>tool failed</think>",
        ground_truth={"label": "success"},
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "env_error",
                "env_error": True,
                "bedrock_error": True,
                "bedrock_retry_count": 3,
                "final_reward": 0.0,
            }
        },
    )

    assert result["score"] == 0.0
    assert result["reward_source"] == "official_gym_env_error"
    assert result["feedback"] == ""
    assert result["tau3_live/env_error_fraction"] == 1.0
    assert result["tau3_live/bedrock_error_fraction"] == 1.0
    assert result["tau3_live/bedrock_retry_count"] == 3.0


def test_normal_official_terminal_reward_is_not_env_error():
    pytest.importorskip("ray")
    from verl.utils.reward_score.feedback.tau3_live import compute_score

    result = compute_score(
        solution_str="Done",
        ground_truth={"label": "success"},
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "user_stop",
                "final_reward": 1.0,
            }
        },
    )

    assert result["score"] == 1.0
    assert result["reward_source"] == "official_or_runtime"
    assert result["tau3_live/env_error_fraction"] == 0.0


def test_retry_delays_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAU3_BEDROCK_RETRY_DELAYS", "0.1, 0.2, bad, 0.3")

    assert tau3_bedrock_retry_delays() == [0.1, 0.2, 0.3]
