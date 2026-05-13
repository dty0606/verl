import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
tensordict = pytest.importorskip("tensordict")

ROOT = Path(__file__).resolve().parents[2]
TensorDict = tensordict.TensorDict

LOSSES_SPEC = importlib.util.spec_from_file_location(
    "sdpo_losses_test",
    ROOT / "verl" / "workers" / "utils" / "losses.py",
)
sdpo_losses = importlib.util.module_from_spec(LOSSES_SPEC)
assert LOSSES_SPEC.loader is not None
LOSSES_SPEC.loader.exec_module(sdpo_losses)

PADDING_SPEC = importlib.util.spec_from_file_location(
    "sdpo_padding_test",
    ROOT / "verl" / "workers" / "utils" / "padding.py",
)
sdpo_padding = importlib.util.module_from_spec(PADDING_SPEC)
assert PADDING_SPEC.loader is not None
PADDING_SPEC.loader.exec_module(sdpo_padding)


def test_sdpo_topk_jsd_loss_is_zero_for_matching_distributions():
    log_probs = torch.log_softmax(torch.tensor([[[2.0, 0.5, -1.0], [1.0, 0.0, -2.0]]]), dim=-1)

    loss = sdpo_losses._sdpo_jensen_shannon_topk_loss(
        student_topk_log_probs=log_probs,
        teacher_topk_log_probs=log_probs,
        alpha=0.5,
        add_tail=True,
    )

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_sdpo_topk_reverse_kl_matches_torch_kl_with_tail():
    student = torch.log_softmax(torch.tensor([[[1.0, 0.0]]]), dim=-1)
    teacher = torch.log_softmax(torch.tensor([[[0.0, 1.0]]]), dim=-1)

    loss = sdpo_losses._sdpo_jensen_shannon_topk_loss(
        student_topk_log_probs=student,
        teacher_topk_log_probs=teacher,
        alpha=1.0,
        add_tail=False,
    )
    expected = torch.nn.functional.kl_div(teacher, student, reduction="none", log_target=True).sum(dim=-1)

    assert torch.allclose(loss, expected, atol=1e-6)


def test_sdpo_topk_interior_alpha_matches_upstream_generalized_jsd():
    student = torch.log_softmax(torch.tensor([[[1.0, 0.0, -1.0]]]), dim=-1)
    teacher = torch.log_softmax(torch.tensor([[[0.0, 1.0, -0.5]]]), dim=-1)
    alpha = torch.tensor(0.5)

    loss = sdpo_losses._sdpo_jensen_shannon_topk_loss(
        student_topk_log_probs=student,
        teacher_topk_log_probs=teacher,
        alpha=float(alpha),
        add_tail=False,
    )

    mixture = torch.logsumexp(
        torch.stack([student + torch.log1p(-alpha), teacher + torch.log(alpha)]),
        dim=0,
    )
    kl_teacher = torch.nn.functional.kl_div(mixture, teacher, reduction="none", log_target=True)
    kl_student = torch.nn.functional.kl_div(mixture, student, reduction="none", log_target=True)
    expected = torch.lerp(kl_student, kl_teacher, alpha).sum(dim=-1)

    assert torch.allclose(loss, expected, atol=1e-6)


def test_sdpo_topk_mass_metrics_exist_without_full_logit_tensors():
    metrics = {}

    sdpo_losses._set_sdpo_topk_mass_metrics(metrics, torch.ones(1, 2, dtype=torch.bool))

    assert metrics["self_distillation/student_topk_mass"] == 0.0
    assert metrics["self_distillation/teacher_topk_mass"] == 0.0


def test_sdpo_topk_mass_metrics_average_selected_tokens():
    metrics = {}
    mask = torch.tensor([[True, False, True]])
    student_mass = torch.tensor([[0.8, 0.1, 0.6]])
    teacher_mass = torch.tensor([[0.9, 0.2, 0.7]])

    sdpo_losses._set_sdpo_topk_mass_metrics(metrics, mask, student_mass, teacher_mass)

    assert metrics["self_distillation/student_topk_mass"] == pytest.approx(0.7)
    assert metrics["self_distillation/teacher_topk_mass"] == pytest.approx(0.8)


def test_sdpo_topk_logits_processor_fails_fast_on_nan_teacher_logprobs():
    values = torch.tensor([[-0.1, float("nan")], [-0.2, -0.3]])
    ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    data = TensorDict(
        {
            "teacher_logprobs": torch.nested.nested_tensor_from_jagged(values, offsets=offsets),
            "teacher_ids": torch.nested.nested_tensor_from_jagged(ids, offsets=offsets),
        },
        batch_size=[],
    )
    config = SimpleNamespace(policy_loss={"sdpo_alpha": 0.5, "sdpo_distillation_add_tail": True})
    student_logits = torch.randn(1, 2, 4)

    with pytest.raises(RuntimeError, match="teacher_topk_log_probs reached actor loss"):
        sdpo_losses._sdpo_topk_logits_processor(config, student_logits=student_logits, data=data)


def test_sdpo_topk_loss_fails_fast_on_invalid_logprob_mass():
    # Finite, but not log-probabilities: exp(0) + exp(0) > 1.
    student = torch.tensor([[[0.0, 0.0]]])
    teacher = torch.log_softmax(torch.tensor([[[1.0, 0.0]]]), dim=-1)

    with pytest.raises(RuntimeError, match="Invalid SDPO top-k log-prob mass"):
        sdpo_losses._sdpo_jensen_shannon_topk_loss(
            student_topk_log_probs=student,
            teacher_topk_log_probs=teacher,
            alpha=0.5,
            add_tail=True,
        )


def test_sdpo_topk_loss_tail_is_finite_for_near_unit_fp16_mass(monkeypatch):
    monkeypatch.setenv("SDPO_FAIL_FAST_NONFINITE", "1")
    logits = torch.tensor([[[12.0, -10.0, -11.0]]], dtype=torch.float16)
    log_probs = torch.log_softmax(logits.float(), dim=-1).to(torch.float16)

    loss = sdpo_losses._sdpo_jensen_shannon_topk_loss(
        student_topk_log_probs=log_probs,
        teacher_topk_log_probs=log_probs,
        alpha=0.5,
        add_tail=True,
    )

    assert torch.isfinite(loss).all()


def test_sdpo_topk_logits_processor_fails_fast_on_student_nan(monkeypatch):
    monkeypatch.setenv("SDPO_FAIL_FAST_NONFINITE", "1")
    values = torch.tensor([[-0.1, -0.2], [-0.2, -0.3]])
    ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    data = TensorDict(
        {
            "teacher_logprobs": torch.nested.nested_tensor_from_jagged(values, offsets=offsets),
            "teacher_ids": torch.nested.nested_tensor_from_jagged(ids, offsets=offsets),
        },
        batch_size=[],
    )
    config = SimpleNamespace(policy_loss={"sdpo_alpha": 0.5, "sdpo_distillation_add_tail": True})
    student_logits = torch.randn(1, 2, 4)
    student_logits[0, 0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="actor_student_logits"):
        sdpo_losses._sdpo_topk_logits_processor(config, student_logits=student_logits, data=data)


def test_sdpo_topk_logits_processor_ignores_non_response_prediction_rows(monkeypatch):
    monkeypatch.setenv("SDPO_FAIL_FAST_NONFINITE", "1")
    # Full sequence: three prompt tokens plus two response tokens. Only the
    # prompt-last and first response-token positions predict response tokens.
    topk = 2
    values = torch.zeros(5, topk)
    values[2] = torch.log_softmax(torch.tensor([1.0, 0.0]), dim=-1)
    values[3] = torch.log_softmax(torch.tensor([0.0, 1.0]), dim=-1)
    ids = torch.zeros(5, topk, dtype=torch.long)
    ids[2] = torch.tensor([1, 2])
    ids[3] = torch.tensor([2, 3])
    offsets = torch.tensor([0, 5], dtype=torch.long)
    data = TensorDict(
        {
            "teacher_logprobs": torch.nested.nested_tensor_from_jagged(values, offsets=offsets),
            "teacher_ids": torch.nested.nested_tensor_from_jagged(ids, offsets=offsets),
            "prompts": torch.tensor([[10, 11, 12]]),
            "responses": torch.tensor([[13, 14]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        },
        batch_size=[],
    )
    config = SimpleNamespace(policy_loss={"sdpo_alpha": 0.5, "sdpo_distillation_add_tail": True})
    student_logits = torch.full((1, 5, 4), -10.0)
    # These invalid rows would have mass > 1 if zero-filled placeholder top-k
    # IDs/logprobs were validated as real distributions.
    student_logits[0, [0, 1, 4], 0] = 10.0
    student_logits[0, 2, [1, 2]] = torch.tensor([1.0, 0.0])
    student_logits[0, 3, [2, 3]] = torch.tensor([0.0, 1.0])

    output = sdpo_losses._sdpo_topk_logits_processor(config, student_logits=student_logits, data=data)

    assert torch.isfinite(output["sdpo_distillation_losses"]).all()
    assert output["sdpo_distillation_losses"].shape == torch.Size([1, 5])
    assert output["sdpo_student_mass"][0, 0].item() == pytest.approx(1.0)
    assert output["sdpo_teacher_mass"][0, 4].item() == pytest.approx(1.0)


def test_sdpo_topk_logits_processor_ignores_unselected_response_rows(monkeypatch):
    monkeypatch.setenv("SDPO_FAIL_FAST_NONFINITE", "1")
    # Full sequence: three prompt tokens plus two response tokens. The first
    # response prediction row is a zero-filled placeholder, but the SDPO target
    # guard excludes that response token, so only the second response token is
    # allowed to participate in top-k validation/JSD.
    topk = 2
    values = torch.zeros(5, topk)
    values[3] = torch.log_softmax(torch.tensor([0.0, 1.0]), dim=-1)
    ids = torch.zeros(5, topk, dtype=torch.long)
    ids[3] = torch.tensor([2, 3])
    offsets = torch.tensor([0, 5], dtype=torch.long)
    data = TensorDict(
        {
            "teacher_logprobs": torch.nested.nested_tensor_from_jagged(values, offsets=offsets),
            "teacher_ids": torch.nested.nested_tensor_from_jagged(ids, offsets=offsets),
            "prompts": torch.tensor([[10, 11, 12]]),
            "responses": torch.tensor([[13, 14]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "response_mask": torch.tensor([[1, 1]]),
            "self_distillation_target_token_mask": torch.tensor([[0.0, 1.0]]),
        },
        batch_size=[],
    )
    config = SimpleNamespace(policy_loss={"sdpo_alpha": 0.5, "sdpo_distillation_add_tail": True})
    student_logits = torch.full((1, 5, 4), -10.0)
    student_logits[0, 2, 0] = 10.0
    student_logits[0, 3, [2, 3]] = torch.tensor([0.0, 1.0])

    output = sdpo_losses._sdpo_topk_logits_processor(config, student_logits=student_logits, data=data)

    assert torch.isfinite(output["sdpo_distillation_losses"]).all()
    assert output["sdpo_teacher_mass"][0, 2].item() == pytest.approx(1.0)
    assert output["sdpo_teacher_mass"][0, 3].item() <= 1.0


def test_sdpo_selected_mask_refuses_nested_prompt_fallback():
    values = torch.log_softmax(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    data = TensorDict(
        {
            "teacher_logprobs": torch.nested.nested_tensor_from_jagged(values, offsets=offsets),
            "teacher_ids": torch.nested.nested_tensor_from_jagged(ids, offsets=offsets),
            "prompts": torch.nested.nested_tensor_from_jagged(torch.tensor([10]), offsets=torch.tensor([0, 1])),
            "responses": torch.nested.nested_tensor_from_jagged(torch.tensor([11]), offsets=torch.tensor([0, 1])),
            "attention_mask": torch.tensor([[1, 1]]),
            "self_distillation_target_token_mask": torch.tensor([[1.0]]),
        },
        batch_size=[],
    )
    config = SimpleNamespace(policy_loss={"sdpo_alpha": 0.5, "sdpo_distillation_add_tail": True})

    with pytest.raises(RuntimeError, match="Cannot align SDPO selected response mask"):
        sdpo_losses._sdpo_topk_logits_processor(config, student_logits=torch.randn(1, 2, 4), data=data)


def test_response_shaped_sampled_teacher_logprobs_are_not_unpadded_as_full_sequence():
    data = TensorDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5], [0, 0, 6, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
            "response_mask": torch.tensor([[1, 1], [1, 0]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 2]]),
            "teacher_logprobs": torch.randn(2, 2),
        },
        batch_size=[2],
    )

    converted = sdpo_padding.left_right_2_no_padding(data)

    assert not converted["teacher_logprobs"].is_nested
    assert converted["teacher_logprobs"].shape == torch.Size([2, 2])


def test_full_sequence_topk_teacher_tensors_are_unpadded_to_nested_values():
    topk = 3
    data = TensorDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5], [0, 0, 6, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
            "response_mask": torch.tensor([[1, 1], [1, 0]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 2]]),
            "teacher_logprobs": torch.randn(2, 5, topk),
            "teacher_ids": torch.ones(2, 5, topk, dtype=torch.long),
        },
        batch_size=[2],
    )

    converted = sdpo_padding.left_right_2_no_padding(data)

    assert converted["teacher_logprobs"].is_nested
    assert converted["teacher_ids"].is_nested
    assert converted["teacher_logprobs"].values().shape == torch.Size([8, topk])
    assert converted["teacher_ids"].values().shape == torch.Size([8, topk])
