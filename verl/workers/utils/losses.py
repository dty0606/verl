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
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


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


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

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
        sdpo_loss_mask = data.get("self_distillation_loss_mask", response_mask).to(response_mask.dtype)
        sdpo_loss_mask = sdpo_loss_mask * data["self_distillation_mask"].to(sdpo_loss_mask.dtype).unsqueeze(1)
        if sdpo_loss_mask.sum().item() == 0:
            pg_loss = log_prob.sum() * 0.0
            pg_metrics = {
                "actor/pg_clipfrac": 0.0,
                "actor/ppo_kl": 0.0,
                "actor/pg_clipfrac_lower": 0.0,
                "self_distillation/empty_target_batch": 1.0,
                "self_distillation/teacher_selected_fraction": 0.0,
            }
        else:
            alpha = float(config.policy_loss.get("sdpo_alpha", 1.0))
            if alpha != 1.0:
                raise ValueError("Latest-VERL vanilla SDPO currently supports response-token reverse KL only (alpha=1.0)")
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
                batch_num_tokens=sdpo_loss_mask.sum().clamp(min=1.0),
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
            }
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
