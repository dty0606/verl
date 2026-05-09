# W&B Run `oe1zswum` Diagnosis

Run: `LOCAL-TAU3-SDPO-original-ema_ref-json-real_sft_step800-west_p5_original_sdpo_true8k_resp_100step_direct_8kresp_16kmodel`

URL: <https://wandb.ai/dtygame1/SDPO-vllm-v1-original-sdpo-safe/runs/oe1zswum>

Pulled locally: 2026-05-09

## Config Proof

This was the intended direct true-8K run, not the earlier capacity-matrix false
8K run.

```text
data.max_prompt_length                      = 8192
data.max_response_length                    = 8192
max_model_len                               = 16384
tau3.sdpo.max_reprompt_len                  = 4096
actor_rollout_ref.rollout.n                 = 8
actor_rollout_ref.rollout.gpu_memory_utilization = 0.5
actor_rollout_ref.actor.ppo_max_token_len_per_gpu = 16384
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu = 16384
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu = 16384
trainer.total_training_steps                = 100
trainer.test_freq                           = 10
trainer.save_freq                           = 10
```

## Result

The run logged through global step 21, then OOMed during step 22.

W&B marks the run state as `finished`, but `output.log` ends with a real
`torch.OutOfMemoryError`. Treat the W&B state as unreliable for this failure.

The traceback is in actor training/backward, not checkpoint save and not the
teacher logprob pass:

```text
WorkerDict.actor_rollout_ref_update_actor()
  ActorRolloutRefWorker.update_actor()
  TrainingWorker.train_mini_batch()
  TrainingWorker.train_batch()
  FSDPTransformerImpl.forward_backward_batch()
  loss.backward()

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.72 GiB.
GPU 0 total 79.18 GiB, 2.24 GiB free.
Process memory in use 75.25 GiB.
Allocated by PyTorch 73.44 GiB.
```

## Step-22 Pre-OOM Diagnostics

Step 22 completed both pre-logprob diagnostic prints before the actor backward
OOM:

```text
actor_student:
  attention_mask_max  = 12526
  attention_mask_mean = 8190.55
  response_mask_max   = 7737
  response_mask_mean  = 3176.45

ema_ref_teacher:
  attention_mask_max  = 10847
  attention_mask_mean = 6482.19
  response_mask_max   = 7737
  response_mask_mean  = 3176.45
```

Interpretation: the 8K response cap was active. The failure happened because the
batch had many long responses near the cap, especially mean response mask jumping
to about 3.18K at step 22.

## Last Logged Completed Step

Step 21:

```text
response_length/max                         = 8192
response_length/mean                        = 2772.34
prompt_length/max                           = 4334
global_seqlen/max                           = 80205
global_seqlen/mean                          = 56002.75
self_distillation/teacher_prompt_token_mean = 2466.28
sdpo_logprob/actor_student peak allocated   = 36.37 GiB
sdpo_logprob/ema_ref_teacher peak allocated = 33.69 GiB
actor/loss                                  = 4.05
actor/grad_norm                             = 0.71
tau3_live/terminal_fraction                 = 0.859
```

The logprob peaks were high but not fatal. The fatal peak occurred during actor
backward on step 22, after teacher/student logprob construction.

## Token-Length Trend

The run crossed into saturated 8K responses starting around step 12.

```text
step  resp_max  resp_mean  student_peak_GiB  teacher_peak_GiB
1     3024      2034.55    19.49             16.89
8     3249      2107.92    26.20             22.36
11    4989      2258.44    29.06             26.48
12    8192      2480.12    37.87             33.93
17    8192      2723.64    37.80             33.91
20    8192      2788.52    37.50             33.37
21    8192      2772.34    36.37             33.69
22    7737*     3176.45*   OOM during actor backward
```

`*` Step 22 values are pre-logprob diagnostic values, not completed W&B scalar
history.

## Current Read

This failure is different from the earlier false-8K OOM:

- The response cap is now correctly applied.
- Checkpointing and EMA teacher save are not the issue.
- Student and teacher logprob passes complete at step 22.
- The fatal pressure is actor update/backward with long response-heavy batches.

The model appears to drift toward long/saturated responses after roughly step
12. By step 22, enough examples are long that even per-GPU microbatch 1 and
gradient checkpointing are not enough on 80GB H100.

## Candidate Next Runs

For a stable audit, the blunt safe recipe is:

```text
MAX_RESPONSE_LENGTH=6144
MAX_MODEL_LEN=12288 or 14336
SDPO_MAX_REPROMPT_LEN=4096
```

If this still OOMs or if we want more margin:

```text
MAX_RESPONSE_LENGTH=4096
MAX_MODEL_LEN=12288
SDPO_MAX_REPROMPT_LEN=4096
```

Engineering fixes worth considering before changing the scientific setup:

- Add actor update CUDA memory diagnostics analogous to logprob diagnostics.
- Add a guard or curriculum that masks/penalizes response-saturated samples
  earlier, because saturation begins around step 12.
- Investigate response-only LM-head projection for logprob/top-k.
- Investigate whether thinking/tool chatter should receive SDPO JSD loss.
- Consider sequence parallelism for actor update if supported in this FSDP path.
