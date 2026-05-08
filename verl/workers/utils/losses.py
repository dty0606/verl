# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.distributed as dist
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def _add_tail_log_prob(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a residual probability bucket for top-k distillation."""
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _renormalize_topk_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def _sdpo_jensen_shannon_topk_loss(
    *,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    alpha: float,
    add_tail: bool,
) -> torch.Tensor:
    """Official-SDPO-style top-k distribution loss.

    ``alpha=0`` is forward KL, ``alpha=1`` is reverse KL, and values in
    between are generalized Jensen-Shannon distillation. This intentionally
    matches the upstream SDPO implementation, including the special-cased KL
    endpoints and the unnormalized interior JSD expression.
    """
    teacher_topk_log_probs = teacher_topk_log_probs.detach()
    if add_tail:
        student_distill_log_probs = _add_tail_log_prob(student_topk_log_probs)
        teacher_distill_log_probs = _add_tail_log_prob(teacher_topk_log_probs)
    else:
        student_distill_log_probs = _renormalize_topk_log_probs(student_topk_log_probs)
        teacher_distill_log_probs = _renormalize_topk_log_probs(teacher_topk_log_probs)

    if alpha == 0.0:
        kl_loss = F.kl_div(student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
    elif alpha == 1.0:
        kl_loss = F.kl_div(teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True)
    else:
        alpha_tensor = torch.tensor(alpha, dtype=student_distill_log_probs.dtype, device=student_distill_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_distill_log_probs + torch.log1p(-alpha_tensor),
                    teacher_distill_log_probs + torch.log(alpha_tensor),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
        kl_loss = torch.lerp(kl_student, kl_teacher, alpha_tensor)

    return kl_loss.sum(dim=-1)


def _sdpo_topk_logits_processor(config: ActorConfig, student_logits: torch.Tensor, data: TensorDict) -> dict[str, torch.Tensor]:
    """Compute SDPO top-k distribution losses during actor forward."""
    if "teacher_logprobs" not in data or "teacher_ids" not in data:
        raise ValueError("SDPO top-k distillation requires teacher_logprobs and teacher_ids in the batch")

    teacher_topk_log_probs = data["teacher_logprobs"]
    teacher_topk_ids = data["teacher_ids"]
    if not teacher_topk_log_probs.is_nested or not teacher_topk_ids.is_nested:
        raise ValueError("SDPO top-k distillation expects no-padding nested teacher_logprobs/teacher_ids")

    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0).to(student_logits.device)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0).to(device=student_logits.device, dtype=torch.long)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    if teacher_topk_log_probs.shape[:2] != student_logits.shape[:2]:
        raise ValueError(
            "SDPO top-k teacher tensors must align with actor logits; "
            f"got teacher={tuple(teacher_topk_log_probs.shape)} actor={tuple(student_logits.shape)}"
        )

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)

    alpha = float(config.policy_loss.get("sdpo_alpha", 0.5))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"SDPO alpha must be in [0, 1], got {alpha}")
    add_tail = bool(config.policy_loss.get("sdpo_distillation_add_tail", True))
    per_token_loss = _sdpo_jensen_shannon_topk_loss(
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        alpha=alpha,
        add_tail=add_tail,
    )
    return {
        "sdpo_distillation_losses": per_token_loss,
        "sdpo_student_mass": student_topk_log_probs.exp().sum(dim=-1),
        "sdpo_teacher_mass": teacher_topk_log_probs.exp().sum(dim=-1),
    }


def _set_sdpo_topk_mass_metrics(
    pg_metrics: dict[str, float],
    sdpo_loss_mask: torch.Tensor,
    sdpo_student_mass: torch.Tensor | None = None,
    sdpo_teacher_mass: torch.Tensor | None = None,
) -> None:
    """Emit stable SDPO top-k mass metric keys across all loss branches."""
    pg_metrics["self_distillation/student_topk_mass"] = 0.0
    pg_metrics["self_distillation/teacher_topk_mass"] = 0.0
    if sdpo_student_mass is None or sdpo_teacher_mass is None:
        return

    selected = sdpo_loss_mask.bool()
    if bool(selected.any().item()):
        pg_metrics["self_distillation/student_topk_mass"] = (
            sdpo_student_mass[selected].float().mean().detach().item()
        )
        pg_metrics["self_distillation/teacher_topk_mass"] = (
            sdpo_teacher_mass[selected].float().mean().detach().item()
        )


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output=None, data: TensorDict = None, dp_group=None, student_logits=None, **kwargs):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    if student_logits is not None:
        return _sdpo_topk_logits_processor(config, student_logits=student_logits, data=data)

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)
    sdpo_full_logit_losses = None
    sdpo_student_mass = None
    sdpo_teacher_mass = None
    if model_output.get("sdpo_distillation_losses", None) is not None:
        sdpo_full_logit_losses = no_padding_2_padding(model_output["sdpo_distillation_losses"], data)
        sdpo_student_mass = no_padding_2_padding(model_output["sdpo_student_mass"], data)
        sdpo_teacher_mass = no_padding_2_padding(model_output["sdpo_teacher_mass"], data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    if loss_mode == "sdpo":
        missing = {"teacher_logprobs", "self_distillation_mask"} - set(data.keys())
        if missing:
            raise ValueError(f"SDPO loss requires {sorted(missing)} in the training batch")
        fields.extend(["teacher_logprobs", "self_distillation_mask"])
        if "teacher_ids" in data:
            fields.append("teacher_ids")
        if "self_distillation_target_token_mask" in data:
            fields.append("self_distillation_target_token_mask")
        if "self_distillation_loss_mask" in data:
            fields.append("self_distillation_loss_mask")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    if loss_mode == "sdpo":
        teacher_log_prob = data["teacher_logprobs"]
        if teacher_log_prob.dim() == 3 and teacher_log_prob.size(-1) == 1:
            teacher_log_prob = teacher_log_prob.squeeze(-1)
        if "self_distillation_target_token_mask" in data:
            sdpo_loss_mask = data["self_distillation_target_token_mask"].to(response_mask.dtype)
        else:
            sdpo_loss_mask = data.get("self_distillation_loss_mask", response_mask).to(response_mask.dtype)
            sdpo_loss_mask = sdpo_loss_mask * data["self_distillation_mask"].to(sdpo_loss_mask.dtype).unsqueeze(1)
        sdpo_batch_num_tokens = sdpo_loss_mask.sum().to(log_prob.device)
        sdpo_global_batch_size = (sdpo_loss_mask.sum(dim=-1) > 0).sum().to(log_prob.device)
        if dp_group is not None and dist.is_available() and dist.is_initialized():
            dist.all_reduce(sdpo_batch_num_tokens, op=dist.ReduceOp.SUM, group=dp_group)
            dist.all_reduce(sdpo_global_batch_size, op=dist.ReduceOp.SUM, group=dp_group)
        if sdpo_batch_num_tokens.item() == 0:
            pg_loss = log_prob.sum() * 0.0
            pg_metrics = {
                "actor/pg_clipfrac": 0.0,
                "actor/ppo_kl": 0.0,
                "actor/pg_clipfrac_lower": 0.0,
                "self_distillation/empty_target_batch": 1.0,
                "self_distillation/teacher_selected_fraction": 0.0,
                "self_distillation/token_fraction": 0.0,
                "self_distillation/full_logit_distillation": 0.0,
                "self_distillation/alpha": 0.0,
                "self_distillation/student_topk_mass": 0.0,
                "self_distillation/teacher_topk_mass": 0.0,
            }
        else:
            alpha = float(config.policy_loss.get("sdpo_alpha", 1.0))
            full_logit_distillation = bool(config.policy_loss.get("sdpo_full_logit_distillation", False))
            if full_logit_distillation:
                if sdpo_full_logit_losses is None:
                    raise ValueError(
                        "SDPO full-logit/top-k distillation is enabled but actor forward did not produce "
                        "sdpo_distillation_losses. Check distillation_use_topk/teacher_ids wiring."
                    )
                per_token_loss = sdpo_full_logit_losses
            else:
                if alpha != 1.0:
                    raise ValueError("Sampled-token SDPO supports reverse KL only (alpha=1.0)")
                if teacher_log_prob.dim() == 3 and teacher_log_prob.size(-1) != 1:
                    raise ValueError("Sampled-token SDPO received top-k teacher_logprobs; enable full-logit distillation")
                log_ratio = log_prob - teacher_log_prob
                per_token_loss = log_ratio.detach() * log_prob
            is_clip = config.policy_loss.get("sdpo_is_clip", 2.0)
            if is_clip is not None:
                negative_approx_kl = torch.clamp((log_prob - old_log_prob).detach(), min=-20.0, max=20.0)
                per_token_loss = per_token_loss * torch.exp(negative_approx_kl).clamp(max=float(is_clip))
            if rollout_is_weights is not None:
                per_token_loss = per_token_loss * rollout_is_weights
            pg_loss = agg_loss(
                loss_mat=per_token_loss,
                loss_mask=sdpo_loss_mask,
                loss_agg_mode=loss_agg_mode,
                dp_size=config.global_batch_info["dp_size"],
                batch_num_tokens=sdpo_batch_num_tokens.clamp(min=1.0),
                global_batch_size=sdpo_global_batch_size.clamp(min=1.0),
                loss_scale_factor=config.loss_scale_factor,
            ) * float(config.policy_loss.get("sdpo_loss_coef", 1.0))
            pg_metrics = {
                "actor/pg_clipfrac": 0.0,
                "actor/ppo_kl": masked_mean(log_prob - old_log_prob, sdpo_loss_mask.bool()).detach().item(),
                "actor/pg_clipfrac_lower": 0.0,
                "self_distillation/empty_target_batch": 0.0,
                "self_distillation/teacher_selected_fraction": (
                    data["self_distillation_mask"].float().mean().detach().item()
                ),
                "self_distillation/token_fraction": (
                    sdpo_loss_mask.float().sum() / response_mask.float().sum().clamp(min=1.0)
                ).detach().item(),
                "self_distillation/full_logit_distillation": 1.0 if full_logit_distillation else 0.0,
                "self_distillation/alpha": alpha,
            }
            _set_sdpo_topk_mass_metrics(pg_metrics, sdpo_loss_mask, sdpo_student_mass, sdpo_teacher_mass)
    else:
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
