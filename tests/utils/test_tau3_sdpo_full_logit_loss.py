import importlib.util
from pathlib import Path

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
