from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_guarded_launcher_defaults_to_peer_only_and_rejects_feedback_memory():
    launcher = (ROOT / "run_local_tau3_sdpo_live_p5.sh").read_text(encoding="utf-8")

    assert 'SDPO_ARM="${SDPO_ARM:-peer_only}"' in launcher
    assert "requires TAU3_LIVE_RUNTIME=official_gym" in launcher
    assert "requires TAU3_MARK_ENV_EXCEPTIONS=false" in launcher
    assert "Use SDPO_ARM=peer_only here" in launcher
    assert "feedback_hybrid" in launcher
    assert "guarded faithful Tau3 SDPO does not allow heuristic teacher feedback" in launcher
    assert "guarded faithful Tau3 SDPO does not allow SDPO memory/note cards" in launcher
    assert "keeps tau3.sdpo.target_guard disabled" in launcher
    assert "tau3.sdpo.use_successful_peer_solution=true" in launcher
    assert "tau3.sdpo.only_failed_rollouts=true" in launcher
    assert "faithful Tau3 SDPO launcher forbids override" in launcher
    assert "tau3.sdpo.reprompt_template=*" in launcher
    assert "tau3.sdpo.target_guard.enabled=*" in launcher
    assert "+tau3.sdpo.target_guard.enabled=*" in launcher
    assert 'ARGS+=("tau3.sdpo.memory' not in launcher
    assert 'ARGS+=("+tau3.sdpo.memory' not in launcher


def test_guarded_tau3_sdpo_yaml_is_peer_only_by_default():
    config = (ROOT / "verl" / "trainer" / "config" / "tau3_sdpo_live.yaml").read_text(encoding="utf-8")

    assert "train_batch_size: 4" in config
    assert "val_batch_size: 4" in config
    assert "max_model_len: 14336" in config
    assert "max_prompt_length: 8192" in config
    assert "max_response_length: 6144" in config
    assert "ppo_mini_batch_size: 4" in config
    assert "total_epochs: 30" in config
    assert "total_training_steps: 240" in config
    assert "max_reprompt_len: 6144" in config
    assert "enabled: False" in config
    assert "feedback_mode: ${oc.env:TAU3_LIVE_FEEDBACK_FORMAT,none}" in config
    assert "use_successful_peer_solution: True" in config
    assert "only_failed_rollouts: True" in config
    assert "{feedback}" not in config
    assert "{memory}" not in config
    assert "\n    memory:" not in config
    assert "feedback_template" not in config


def test_guarded_trainer_has_no_teacher_feedback_or_memory_sections():
    trainer = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(encoding="utf-8")

    assert "def _collect_feedback(" not in trainer
    assert '"feedback": feedback_section' not in trainer
    assert "feedback_template" not in trainer
    assert "memory_cfg" not in trainer
    assert '"{prompt}{solution}\\n\\nCorrectly solve the original question."' in trainer
    assert "eligible_mask=demo_safe_mask" not in trainer
    assert "teacher prompt saturation would silently truncate" not in trainer
    assert '"self_distillation/demo_target_guard_success_fraction"' in trainer
    assert '"self_distillation/teacher_prompt_saturation_active_fraction"' in trainer
    assert "skip_actor_update_due_empty_sdpo" in trainer
    assert '"actor/update_skipped_empty_sdpo_target"' in trainer
    assert '"self_distillation/ema_teacher_skipped_empty_target"' in trainer


def test_p5_sdpo_helpers_use_faithful_peer_only_route():
    east = (ROOT / "scripts" / "p5_run_east_original_sdpo_full.sh").read_text(encoding="utf-8")
    matrix = (ROOT / "scripts" / "p5_run_vllm_v1_capacity_matrix.sh").read_text(encoding="utf-8")
    image_smoke = (ROOT / "scripts" / "p5_run_image_smoke_vllm_v1.sh").read_text(encoding="utf-8")

    assert "TAU3_LIVE_FEEDBACK_FORMAT:-none" in east
    assert "export SDPO_ARM=peer_only" in east
    assert 'MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-6144}"' in east
    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-14336}"' in east
    assert 'SDPO_TARGET_GUARD_ENABLED=false' in east
    assert 'TAU3_MARK_ENV_EXCEPTIONS:-false' in east
    assert "p5_run_vllm_v1_capacity_matrix.sh" not in east
    assert 'run_local_tau3_sdpo_live_p5.sh" "$TASK_PATH" "$RUN_STEM" none' in east
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
