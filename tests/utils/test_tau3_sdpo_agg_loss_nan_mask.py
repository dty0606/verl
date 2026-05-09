import pytest

torch = pytest.importorskip("torch")

from verl.trainer.ppo.core_algos import agg_loss


@pytest.mark.parametrize("mode", ["seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"])
def test_seq_agg_loss_ignores_masked_nan(mode):
    loss_mat = torch.tensor([[1.0, float("nan")]])
    loss_mask = torch.tensor([[1.0, 0.0]])

    loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=loss_mask,
        loss_agg_mode=mode,
        dp_size=1,
        global_batch_size=torch.tensor(1.0),
        loss_scale_factor=2 if mode == "seq-mean-token-sum-norm" else None,
    )

    assert torch.isfinite(loss)
