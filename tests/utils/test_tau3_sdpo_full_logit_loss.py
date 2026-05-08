import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]

LOSSES_SPEC = importlib.util.spec_from_file_location(
    "sdpo_losses_test",
    ROOT / "verl" / "workers" / "utils" / "losses.py",
)
sdpo_losses = importlib.util.module_from_spec(LOSSES_SPEC)
assert LOSSES_SPEC.loader is not None
LOSSES_SPEC.loader.exec_module(sdpo_losses)


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
