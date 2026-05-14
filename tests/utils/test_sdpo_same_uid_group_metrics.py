import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "verl" / "trainer" / "ppo" / "sdpo_group_metrics.py"
SPEC = importlib.util.spec_from_file_location("sdpo_group_metrics", HELPER)
sdpo_group_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sdpo_group_metrics
SPEC.loader.exec_module(sdpo_group_metrics)

format_sdpo_same_uid_warning = sdpo_group_metrics.format_sdpo_same_uid_warning
sdpo_same_uid_group_metrics = sdpo_group_metrics.sdpo_same_uid_group_metrics


def test_same_uid_metrics_detect_homogeneous_skip_pattern():
    uids = ["a"] * 8 + ["b"] * 8 + ["c"] * 8 + ["d"] * 8
    scores = [1.0] * 8 + [0.0] * 8 + [1.0] * 8 + [0.0] * 8

    metrics = sdpo_same_uid_group_metrics(uids=uids, strict_scores=scores, official_scores=scores)

    assert metrics["self_distillation/same_uid_group_count"] == 4.0
    assert metrics["self_distillation/same_uid_strict_all_success_group_count"] == 2.0
    assert metrics["self_distillation/same_uid_strict_all_fail_group_count"] == 2.0
    assert metrics["self_distillation/same_uid_strict_mixed_group_count"] == 0.0
    assert metrics["self_distillation/same_uid_mixed_fraction"] == 0.0

    warning = format_sdpo_same_uid_warning(step=2, metrics=metrics)
    assert warning is not None
    assert "no_mixed_strict_groups" in warning
    assert "all_success=2" in warning
    assert "all_fail=2" in warning


def test_same_uid_metrics_report_mixed_groups_without_warning():
    uids = ["a"] * 4 + ["b"] * 4
    strict_scores = [1.0, 0.0, 0.0, 1.0] + [1.0] * 4
    official_scores = [1.0] * 8

    metrics = sdpo_same_uid_group_metrics(
        uids=uids,
        strict_scores=strict_scores,
        official_scores=official_scores,
    )

    assert metrics["self_distillation/same_uid_strict_mixed_group_count"] == 1.0
    assert metrics["self_distillation/same_uid_strict_all_success_group_count"] == 1.0
    assert metrics["self_distillation/same_uid_official_mixed_group_count"] == 0.0
    assert metrics["self_distillation/same_uid_official_all_success_group_count"] == 2.0
    assert format_sdpo_same_uid_warning(step=1, metrics=metrics) is None


def test_same_uid_metrics_exclude_ineligible_rows_and_missing_uids():
    metrics = sdpo_same_uid_group_metrics(
        uids=["a", "a", "", None],
        strict_scores=[1.0, 0.0, 1.0, 0.0],
        eligible_mask=[True, True, True, False],
    )

    assert metrics["self_distillation/same_uid_group_count"] == 1.0
    assert metrics["self_distillation/same_uid_strict_mixed_group_count"] == 1.0
    assert metrics["self_distillation/same_uid_missing_uid_fraction"] == 0.25
    assert metrics["self_distillation/same_uid_ineligible_fraction"] == 0.25
