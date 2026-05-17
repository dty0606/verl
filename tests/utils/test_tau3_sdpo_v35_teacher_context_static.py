from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v35_teacher_context_flags_default_off_and_are_wired():
    config = (ROOT / "verl" / "trainer" / "config" / "tau3_sdpo_live.yaml").read_text(encoding="utf-8")
    launcher = (ROOT / "run_local_tau3_sdpo_live_p5.sh").read_text(encoding="utf-8")

    assert "gt_metadata_enabled: ${oc.env:SDPO_GT_METADATA_ENABLED,false}" in config
    assert "failed_peer_enabled: ${oc.env:SDPO_FAILED_PEER_ENABLED,false}" in config
    assert "failed_peer_max_chars: ${oc.env:SDPO_FAILED_PEER_MAX_CHARS,4096}" in config
    assert "tau3.sdpo.gt_metadata_enabled=${SDPO_GT_METADATA_ENABLED:-false}" in launcher
    assert "tau3.sdpo.failed_peer_enabled=${SDPO_FAILED_PEER_ENABLED:-false}" in launcher
    assert "tau3.sdpo.failed_peer_max_chars=${SDPO_FAILED_PEER_MAX_CHARS:-4096}" in launcher


def test_v35_gt_metadata_renderer_is_allowlist_redacted():
    trainer = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")

    assert "def _render_sdpo_gt_metadata_redacted" in trainer
    assert "Ground-truth evaluation metadata, values redacted" in trainer
    assert '"gt_metadata_redacted"' in trainer
    assert '"reward_basis"' in trainer
    assert '"actions"' in trainer
    assert '"argument_keys"' in trainer
    assert '"communicate_info_count"' in trainer
    assert '"nl_assertions_count"' in trainer
    assert 'action.get("arguments")' in trainer
    assert 'action.get("action_id")' not in trainer
    assert 'criteria.get("env_assertions")' not in trainer
    assert 'gt.get("user_scenario")' not in trainer
    assert 'gt.get("description")' not in trainer


def test_v35_failed_assistant_evidence_is_assistant_only_and_deterministic():
    trainer = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")

    assert "def _assistant_only_response" in trainer
    assert "def _get_sdpo_failed_assistant_evidence" in trainer
    assert "hashlib.sha256" in trainer
    assert "{idx}:failed_peer" in trainer
    assert "failed_peer_max_chars" in trainer
    assert "Observed failed assistant attempt, not a solution" in trainer
    assert "self._remove_thinking_trace(evidence)" in trainer
    assert "role == \"assistant\"" in trainer
    assert "role == \"user\"" not in trainer
    assert "role == \"tool\"" not in trainer


def test_v35_all_fail_activation_and_metrics_are_auditable():
    trainer = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")

    assert "row_selected = on_failure_path or not only_failed_rollouts" in trainer
    assert "has_teacher_context = has_solution or has_gt_metadata or has_failed_assistant" in trainer
    assert "active = row_selected and has_teacher_context" in trainer
    assert '"self_distillation/gt_metadata_used_fraction"' in trainer
    assert '"self_distillation/failed_peer_used_fraction"' in trainer
    assert '"self_distillation/failed_peer_self_fallback_fraction"' in trainer
    assert '"self_distillation/active_without_success_peer_fraction"' in trainer
    assert '"self_distillation/all_fail_gt_active_fraction"' in trainer
    assert '"tau3_length/teacher_gt_metadata_tokens_mean"' in trainer
    assert '"tau3_length/teacher_failed_peer_tokens_mean"' in trainer
