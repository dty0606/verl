# W&B Run rag0ih8i SDPO OOM Diagnostics

Run: `LOCAL-TAU3-SDPO-original-ema_ref-json-real_sft_step800-west_p5_original_sdpo_safe_8kreprompt4k_30step_02_auto_prefix_24k_48k`

Project: `dtygame1/SDPO-vllm-v1-original-sdpo-safe`

URL: <https://wandb.ai/dtygame1/SDPO-vllm-v1-original-sdpo-safe/runs/rag0ih8i>

Pulled files:

- `config.yaml`
- `output.log`
- `wandb-metadata.json`
- `wandb-summary.json`
- `history.jsonl`
- `history_interesting.json`
- `topk_pre_logprob.json`

## Key Finding

The run name says `safe_8kreprompt4k`, but the actual W&B config did **not**
use an 8K response cap.

Actual config:

```text
data.max_prompt_length                    = 8192
data.max_response_length                  = 24576
max_model_len                             = 49152
tau3.sdpo.max_reprompt_len                = 4096
actor_rollout_ref.rollout.log_prob_*      = micro_batch_size_per_gpu=1, dynamic_bsz=false, max_token_len_per_gpu=49152
actor_rollout_ref.ref.log_prob_*          = micro_batch_size_per_gpu=1, dynamic_bsz=false, max_token_len_per_gpu=49152
```

So the teacher reprompt cap was correctly reduced to 4K, but response length
remained at the 24K capacity-profile value.

Root cause:

- `scripts/p5_run_east_original_sdpo_full.sh` exports default `MAX_RESPONSE_LENGTH=24576`.
- It then calls `scripts/p5_run_vllm_v1_capacity_matrix.sh`.
- The capacity matrix hardcodes profile `02_auto_prefix_24k_48k` as
  `MAX_RESPONSE_LENGTH=24576` and `MAX_MODEL_LEN=49152`, overriding the intended
  safe cap when that profile is selected.

## W&B History

W&B history contains only steps 1 through 7.

The downloaded `output.log` contains pre-logprob diagnostics for step 8, but no
completed `step:8` metric line and no traceback. This means the process likely
died after step-8 pre-logprob diagnostics and before W&B received the training
metrics for step 8.

## Step-Level Memory Summary

```text
step  resp_max  prompt_max  global_max  old_peak  student_peak  teacher_peak  student_before_free  teacher_before_free
1     3056      4388        55144       6.66      19.62         17.08         68.19                55.23
2     2578      4210        50732       11.91     23.88         20.13         57.80                45.85
3     3626      4334        60188       12.46     26.60         22.76         57.35                43.21
4     5296      4334        60091       13.35     31.01         27.23         56.43                35.79
5     3566      4291        58851       12.56     27.10         23.25         57.27                39.28
6     3842      4241        56462       12.65     27.54         23.76         57.05                37.93
7     3114      4321        52664       12.37     26.12         22.34         57.31                43.56
```

Units for memory columns: GiB.

Interpretation:

- The old actor sampled-token logprob pass is much cheaper.
- The SDPO actor-student top-k pass is the largest observed completed phase.
- The EMA teacher gather pass is slightly smaller than actor-student because the
  teacher prompt is shorter than original prompt in this run.
- Step 4 already reaches `31.01 GiB` allocated / `40.08 GiB` reserved in the
  actor-student pass with `response_mask_max=4397`.

## Step-8 Pre-Logprob Spike

Step 8 pre-logprob diagnostics:

```text
phase=actor_student
attention_mask_max = 28749
attention_mask_mean = 7045.78125
response_mask_max = 23679
response_mask_mean = 1764.3125
topk = 100

phase=ema_ref_teacher
attention_mask_max = 27058
attention_mask_mean = 5368.96875
response_mask_max = 23679
response_mask_mean = 1764.3125
gather_response_ids_shape = [64, 24576, 100]
topk = 100
```

This is the smoking gun. One rollout produced `23679` assistant-response tokens,
near the 24K cap. That creates a one-sequence actor-student logprob forward around:

```text
28749 tokens x vocab_size
```

and a teacher forward around:

```text
27058 tokens x vocab_size
```

Because W&B has no step-8 completed metrics, the crash likely happened during
the step-8 actor-student top-k pass or shortly after it.

## Current Conclusion

This run does not prove that an 8K response cap still OOMs. It proves that the
supposed safe run was still effectively a 24K-response run because the selected
capacity profile overwrote `MAX_RESPONSE_LENGTH`.

For the next safe test, either:

- bypass `p5_run_vllm_v1_capacity_matrix.sh` and invoke `run_local_tau3_sdpo_live_p5.sh` directly with `MAX_RESPONSE_LENGTH=8192`, `MAX_MODEL_LEN=16384`, and `SDPO_MAX_REPROMPT_LEN=4096`; or
- add a custom capacity profile that uses the caller-provided `MAX_RESPONSE_LENGTH`
  and `MAX_MODEL_LEN` instead of hardcoded `24k/48k`.
