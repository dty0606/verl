from __future__ import annotations

import importlib.util
import sys
import types
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


def _load_tau3_feedback_renderers():
    # Avoid importing verl.__init__ in lightweight local test environments that
    # do not install Ray.
    if importlib.util.find_spec("ray") is None:
        sys.modules.setdefault("verl", types.ModuleType("verl"))
        sys.modules.setdefault("verl.utils", types.ModuleType("verl.utils"))
        diagnostics_path = Path(__file__).resolve().parents[2] / "verl" / "utils" / "tau3_diagnostics.py"
        diagnostics_spec = importlib.util.spec_from_file_location("verl.utils.tau3_diagnostics", diagnostics_path)
        assert diagnostics_spec is not None and diagnostics_spec.loader is not None
        diagnostics_module = importlib.util.module_from_spec(diagnostics_spec)
        sys.modules[diagnostics_spec.name] = diagnostics_module
        diagnostics_spec.loader.exec_module(diagnostics_module)

    module_path = Path(__file__).resolve().parents[2] / "verl" / "utils" / "tau3_feedback_renderers.py"
    spec = importlib.util.spec_from_file_location("verl.utils.tau3_feedback_renderers", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tau3_live_runtime = _load_tau3_live_runtime()
classify_tau3_env_error_payload = tau3_live_runtime.classify_tau3_env_error_payload
tau3_bedrock_retry_delays = tau3_live_runtime.tau3_bedrock_retry_delays
tau3_runtime_mode = tau3_live_runtime.tau3_runtime_mode
normalize_feedback_mode = _load_tau3_feedback_renderers().normalize_feedback_mode


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


def test_tau3_runtime_defaults_to_official_gym(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_LIVE_RUNTIME", raising=False)

    assert tau3_runtime_mode() == "official_gym"
    assert tau3_runtime_mode("proxy_legacy") == "proxy_legacy"


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


def test_tau3_diagnostic_feedback_is_audit_only(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("ray")
    from verl.utils.reward_score.feedback.tau3_live import compute_score

    monkeypatch.setenv("TAU3_LIVE_FEEDBACK_FORMAT", "json")
    result = compute_score(
        solution_str="<think>loop</think> I cannot complete this.",
        ground_truth={"label": "success"},
        extra_info={
            "tau3_live_result": {
                "runtime": "official_gym",
                "status": "terminated",
                "terminal_reason": "max_steps",
                "final_reward": 0.0,
                "reward_info": {"info": {"note": "Simulation terminated prematurely. Termination reason: max_steps"}},
            }
        },
    )

    assert result["feedback_mode"] == "json"
    assert result["feedback"] == ""
    assert result["tau3_live/budget_exhausted_fraction"] == 1.0


def test_tau3_feedback_renderer_defaults_to_none():
    assert normalize_feedback_mode(None) == "none"
    assert normalize_feedback_mode("") == "none"


def test_retry_delays_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAU3_BEDROCK_RETRY_DELAYS", "0.1, 0.2, bad, 0.3")

    assert tau3_bedrock_retry_delays() == [0.1, 0.2, 0.3]


def test_official_gym_defaults_to_compact_observation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION", raising=False)

    created_envs = []

    class FakeAgentGymEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_envs.append(self)

        def reset(self, seed=None):
            return "initial observation", {"seed": seed}

    tau2_module = types.ModuleType("tau2")
    gym_module = types.ModuleType("tau2.gym")
    gym_agent_module = types.ModuleType("tau2.gym.gym_agent")
    gym_agent_module.AgentGymEnv = FakeAgentGymEnv

    monkeypatch.setitem(sys.modules, "tau2", tau2_module)
    monkeypatch.setitem(sys.modules, "tau2.gym", gym_module)
    monkeypatch.setitem(sys.modules, "tau2.gym.gym_agent", gym_agent_module)

    manager = tau3_live_runtime.Tau3GymLiveSessionManager
    manager._sessions.clear()
    try:
        manager.start_session(
            request_id="compact-default-test",
            domain="airline",
            task_id="1",
            max_steps=16,
            user_model="mock-user-model",
        )
        assert created_envs[-1].kwargs["all_messages_as_observation"] is False

        monkeypatch.setenv("TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION", "1")
        manager.start_session(
            request_id="full-transcript-opt-in-test",
            domain="airline",
            task_id="1",
            max_steps=16,
            user_model="mock-user-model",
        )
        assert created_envs[-1].kwargs["all_messages_as_observation"] is True
    finally:
        manager._sessions.clear()
