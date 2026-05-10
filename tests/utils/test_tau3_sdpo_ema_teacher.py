from types import SimpleNamespace

import pytest

omegaconf = pytest.importorskip("omegaconf")
torch = pytest.importorskip("torch")

OmegaConf = omegaconf.OmegaConf

from verl.trainer.ppo.utils import need_sdpo_ema_teacher
from verl.workers.config.actor import FSDPActorConfig
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.engine_workers import _force_model_only_checkpoint_contents, ema_update_module_params


def test_need_sdpo_ema_teacher_only_for_sdpo_ema_ref():
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": False,
                    "policy_loss": {"loss_mode": "sdpo"},
                }
            },
            "algorithm": {"use_kl_in_reward": False},
            "tau3": {"sdpo": {"enabled": True, "teacher_backend": "ema_ref"}},
        }
    )
    assert need_sdpo_ema_teacher(cfg)

    cfg.tau3.sdpo.teacher_backend = "actor_snapshot"
    assert not need_sdpo_ema_teacher(cfg)

    cfg.actor_rollout_ref.actor.policy_loss.loss_mode = "vanilla"
    cfg.tau3.sdpo.teacher_backend = "ema_ref"
    assert not need_sdpo_ema_teacher(cfg)


def test_ema_update_module_params_moves_teacher_toward_actor():
    teacher = torch.nn.Linear(2, 1, bias=False)
    actor = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        teacher.weight.fill_(0.0)
        actor.weight.fill_(1.0)

    metrics = ema_update_module_params(teacher, actor, update_rate=0.25)

    assert metrics["updated"] is True
    assert metrics["param_tensors"] == 1
    assert metrics["device_transfer_tensors"] == 0
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.25))


def test_ema_update_finite_check_rejects_real_nonfinite_actor_param(monkeypatch):
    monkeypatch.setenv("SDPO_EMA_FINITE_CHECK", "1")
    teacher = torch.nn.Linear(2, 1, bias=False)
    actor = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        teacher.weight.fill_(0.0)
        actor.weight.fill_(1.0)
        actor.weight[0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="bad_count=1"):
        ema_update_module_params(teacher, actor, update_rate=0.25)


def test_ema_update_skips_vision_tower_params_even_if_nonfinite(monkeypatch):
    monkeypatch.setenv("SDPO_EMA_FINITE_CHECK", "1")

    class ToyVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language = torch.nn.Linear(2, 1, bias=False)
            self.visual = torch.nn.Linear(2, 1, bias=False)

    teacher = ToyVLM()
    actor = ToyVLM()
    with torch.no_grad():
        teacher.language.weight.fill_(0.0)
        teacher.visual.weight.fill_(0.0)
        actor.language.weight.fill_(1.0)
        actor.visual.weight.fill_(1.0)
        actor.visual.weight[0, 0] = float("nan")

    metrics = ema_update_module_params(
        teacher,
        actor,
        update_rate=0.25,
        skip_frozen_params=True,
        skip_vision_tower=True,
    )

    assert metrics["param_tensors"] == 1
    assert metrics["skipped_vision_param_tensors"] == 1
    assert torch.allclose(teacher.language.weight, torch.full_like(teacher.language.weight, 0.25))
    assert torch.allclose(teacher.visual.weight, torch.zeros_like(teacher.visual.weight))


def test_ema_update_still_rejects_nonfinite_language_param_when_skipping_vision(monkeypatch):
    monkeypatch.setenv("SDPO_EMA_FINITE_CHECK", "1")

    class ToyVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language = torch.nn.Linear(2, 1, bias=False)
            self.visual = torch.nn.Linear(2, 1, bias=False)

    teacher = ToyVLM()
    actor = ToyVLM()
    with torch.no_grad():
        actor.language.weight[0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="language.weight.*bad_count=1"):
        ema_update_module_params(
            teacher,
            actor,
            update_rate=0.25,
            skip_frozen_params=True,
            skip_vision_tower=True,
        )


def test_qwen35_text_only_freeze_skips_dummy_visual_forward():
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from verl.models.transformers.qwen3_5 import _get_input_embeds

    class VisualShouldNotRun(torch.nn.Module):
        dtype = torch.float32

        def forward(self, *args, **kwargs):
            raise AssertionError("dummy visual forward should be skipped when freeze_vision_tower=True")

    class FakeQwen35(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(freeze_vision_tower=True)
            self.embed = torch.nn.Embedding(8, 4)
            self.visual = VisualShouldNotRun()

        def get_input_embeddings(self):
            return self.embed

    model = FakeQwen35()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    output = _get_input_embeds(model, input_ids)

    assert output["inputs_embeds"].shape == (1, 3, 4)
    assert output["attention_mask"] is None


def test_ema_update_module_params_handles_cpu_actor_cuda_teacher():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to reproduce actor/teacher device mismatch")

    teacher = torch.nn.Linear(2, 1, bias=False, device="cuda")
    actor = torch.nn.Linear(2, 1, bias=False, device="cpu")
    with torch.no_grad():
        teacher.weight.fill_(0.0)
        actor.weight.fill_(1.0)

    metrics = ema_update_module_params(teacher, actor, update_rate=0.25)

    assert metrics["updated"] is True
    assert metrics["device_transfer_tensors"] == 1
    assert teacher.weight.device.type == "cuda"
    assert torch.allclose(teacher.weight.cpu(), torch.full_like(actor.weight, 0.25))


def test_ema_update_rejects_mismatched_modules():
    teacher = torch.nn.Linear(2, 1, bias=False)
    actor = torch.nn.Linear(3, 1, bias=False)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        ema_update_module_params(teacher, actor, update_rate=0.25)


def test_sdpo_ema_teacher_checkpoint_manager_forced_model_only():
    manager = SimpleNamespace(
        checkpoint_save_contents=["model", "optimizer", "extra"],
        checkpoint_load_contents=["model", "optimizer", "extra"],
        checkpoint_config={"save_contents": ["model", "optimizer", "extra"], "load_contents": ["model", "optimizer"]},
    )
    worker = SimpleNamespace(engine=SimpleNamespace(checkpoint_manager=manager))

    assert _force_model_only_checkpoint_contents(worker)

    assert manager.checkpoint_save_contents == ["model"]
    assert manager.checkpoint_load_contents == ["model"]
    assert manager.checkpoint_config["save_contents"] == ["model"]
    assert manager.checkpoint_config["load_contents"] == ["model"]


def test_forward_only_ref_config_skips_ppo_micro_batch_assertion():
    cfg = FSDPActorConfig(
        rollout_n=1,
        ppo_micro_batch_size=None,
        ppo_micro_batch_size_per_gpu=None,
        fsdp_config=FSDPEngineConfig(forward_only=True),
    )

    assert cfg.fsdp_config.forward_only is True


def test_training_actor_config_still_requires_ppo_micro_batch():
    with pytest.raises(AssertionError, match="Please set at least one"):
        FSDPActorConfig(
            rollout_n=1,
            ppo_micro_batch_size=None,
            ppo_micro_batch_size_per_gpu=None,
            fsdp_config=FSDPEngineConfig(forward_only=False),
        )
