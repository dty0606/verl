import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "tau3" / "sdpo_p1_stabilization_qc.py"
SPEC = importlib.util.spec_from_file_location("sdpo_p1_stabilization_qc", SCRIPT)
qc = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = qc
SPEC.loader.exec_module(qc)


def test_evaluate_gates_passes_with_required_evidence():
    text = """
    [sdpo_cuda_memory] {"phase": "ref_compute_log_prob", "event": "after"}
    [sdpo_cuda_memory] {"phase": "actor_compute_log_prob", "event": "after"}
    [sdpo_cuda_memory] {"phase": "actor_update", "event": "after"}
    skip_actor_update_due_empty_sdpo cause=empty_target
    sdpo_mask_debug selected target mask sample row=0
    saving checkpoint checkpoints/foo/global_step_2
    teacher scheduling equivalence passed
    """
    metric_keys = {
        "val/tau3/pass^1",
        "val/tau3/reward/mean_at_1",
        "self_distillation/teacher_backend_actor_snapshot",
    }

    results = qc.evaluate_gates(
        text,
        metric_keys,
        mask_debug_enabled=True,
        update_counter_changed=True,
        teacher_scheduling_enabled=True,
    )

    assert {result.name: result.status for result in results} == {
        "strict_reward_metrics_present": "PASS",
        "mask_debug_sample_present": "PASS",
        "checkpoint_saved_when_update_counter_changed": "PASS",
        "early_skip_cause_logged": "PASS",
        "cuda_memory_phase_metrics_present": "PASS",
        "teacher_scheduling_equivalence": "PASS",
    }


def test_optional_gates_skip_when_not_enabled():
    text = """
    [sdpo_cuda_memory] {"phase": "ref_compute_log_prob", "event": "after"}
    [sdpo_cuda_memory] {"phase": "actor_compute_log_prob", "event": "after"}
    [sdpo_cuda_memory] {"phase": "actor_update", "event": "after"}
    actor/update_skipped_empty_sdpo_target=1
    reward/mean_at_1=0.25 pass^1=0.25
    """

    results = qc.evaluate_gates(text, set())
    by_name = {result.name: result.status for result in results}

    assert by_name["mask_debug_sample_present"] == "SKIP"
    assert by_name["checkpoint_saved_when_update_counter_changed"] == "SKIP"
    assert by_name["teacher_scheduling_equivalence"] == "SKIP"
    assert by_name["strict_reward_metrics_present"] == "PASS"


def test_missing_required_gate_fails_when_enabled():
    results = qc.evaluate_gates(
        "pass^1=0.1 reward/mean_at_1=0.1",
        set(),
        mask_debug_enabled=True,
        update_counter_changed=True,
        teacher_scheduling_enabled=True,
    )

    by_name = {result.name: result.status for result in results}
    assert by_name["mask_debug_sample_present"] == "FAIL"
    assert by_name["checkpoint_saved_when_update_counter_changed"] == "FAIL"
    assert by_name["early_skip_cause_logged"] == "FAIL"
    assert by_name["cuda_memory_phase_metrics_present"] == "FAIL"
    assert by_name["teacher_scheduling_equivalence"] == "FAIL"


def test_collect_text_loads_nested_metric_keys(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "val": {"tau3": {"pass^1": 0.5, "reward": {"mean_at_1": 0.5}}},
                "self_distillation": {"teacher_backend_ema_ref": 1.0},
            }
        ),
        encoding="utf-8",
    )

    text, metric_keys = qc.collect_text([], [metrics_path])

    assert "val/tau3/pass^1" in metric_keys
    assert "val/tau3/reward/mean_at_1" in metric_keys
    assert "teacher_backend_ema_ref" in text


def test_report_template_contains_global_safety_rules():
    template = qc.report_template()

    assert "No dynamic sampling" in template
    assert "No new live hindsight context" in template
    assert "No NL assertion" in template
    assert "No synthetic feedback" in template
    assert "test_tau3_strict_action_reward.py" in template
    assert "test_tau3_sdpo_mask_debug.py" in template
