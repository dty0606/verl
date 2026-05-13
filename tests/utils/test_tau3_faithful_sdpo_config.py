from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_guarded_launcher_defaults_to_peer_only_and_rejects_feedback_memory():
    launcher = (ROOT / "run_local_tau3_sdpo_live_p5.sh").read_text(encoding="utf-8")

    assert 'SDPO_ARM="${SDPO_ARM:-peer_only}"' in launcher
    assert "Use SDPO_ARM=peer_only here" in launcher
    assert "feedback_hybrid" in launcher
    assert "guarded faithful Tau3 SDPO does not allow heuristic teacher feedback" in launcher
    assert "guarded faithful Tau3 SDPO does not allow SDPO memory/note cards" in launcher
    assert "tau3.sdpo.include_environment_feedback=false" in launcher
    assert "tau3.sdpo.use_successful_peer_solution=true" in launcher
    assert "tau3.sdpo.environment_feedback_only_without_solution=false" in launcher
    assert "tau3.sdpo.memory" not in launcher


def test_guarded_tau3_sdpo_yaml_is_peer_only_by_default():
    config = (ROOT / "verl" / "trainer" / "config" / "tau3_sdpo_live.yaml").read_text(encoding="utf-8")

    assert "feedback_mode: ${oc.env:TAU3_LIVE_FEEDBACK_FORMAT,none}" in config
    assert "include_environment_feedback: False" in config
    assert "use_successful_peer_solution: True" in config
    assert "environment_feedback_only_without_solution: False" in config
    assert "serialize_nonstring_feedback: False" in config
    assert "{memory}" not in config
    assert "\n    memory:" not in config


def test_p5_sdpo_helpers_use_faithful_peer_only_route():
    east = (ROOT / "scripts" / "p5_run_east_original_sdpo_full.sh").read_text(encoding="utf-8")
    matrix = (ROOT / "scripts" / "p5_run_vllm_v1_capacity_matrix.sh").read_text(encoding="utf-8")
    image_smoke = (ROOT / "scripts" / "p5_run_image_smoke_vllm_v1.sh").read_text(encoding="utf-8")

    assert "TAU3_LIVE_FEEDBACK_FORMAT:-none" in east
    assert "export SDPO_ARM=peer_only" in east
    assert "TAU3_LIVE_FEEDBACK_FORMAT:-none" in matrix
    assert 'SDPO_ARM="${SDPO_ARM:-peer_only}"' in matrix
    assert '"${RUN_NAME_PREFIX}_${name}" none' in matrix
    assert 'export SDPO_ARM="${SDPO_ARM:-peer_only}"' in image_smoke
    assert '"$TASK_PATH" vllm_v1_image_smoke none' in image_smoke


def test_trainer_metrics_keep_selected_mask_and_no_target_teacher_lengths():
    trainer = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")

    assert "(target_loss_mask * target_mask.unsqueeze(1)).sum()" in trainer
    assert '"tau3_length/teacher_prompt_tokens_mean": 0.0' in trainer
    assert '"tau3_length/teacher_base_prompt_tokens_mean": float(np.mean(teacher_base_prompt_token_lengths))' in trainer
    assert '"tau3_length/teacher_component_decomposition_approx": 1.0' in trainer


def test_trainer_ema_skip_uses_raw_any_rank_signals():
    trainer = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")

    assert "raw_optimizer_step_skipped = actor_raw_metrics.get" in trainer
    assert "np.nanmax(np.asarray(raw_optimizer_step_skipped, dtype=float)) >= 0.5" in trainer
    assert "np.isfinite(np.asarray(raw_grad_norm, dtype=float)).all()" in trainer
