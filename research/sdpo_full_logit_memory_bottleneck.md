# SDPO Full-Logit Memory Bottleneck

Last updated: 2026-05-09

## Purpose

Document the memory bottleneck observed when running faithful Tau3 SDPO with
EMA teacher and full-logit/top-k distillation. This note is meant to preserve
the current shared understanding while we continue adding P5 metrics and
engineering experiments.

## Short Version

The primary bottleneck is not the 4B model weights. It is the transient
full-vocabulary logits tensor used by SDPO distribution matching:

```text
num_unpadded_tokens_in_forward x vocab_size
```

For Qwen-style models, `vocab_size` is roughly 150K tokens. Even a seemingly
moderate multi-turn trajectory can create large tensors once prompt and response
tokens are projected through the LM head.

GRPO mainly needs sampled-token logprobs shaped like:

```text
num_response_tokens
```

Faithful SDPO full-logit/top-k distillation needs distribution information shaped
transiently like:

```text
num_prompt_plus_response_tokens x vocab_size
```

then keeps a compressed top-k representation shaped like:

```text
num_response_tokens x top_k
```

The expensive part is the transient full-vocab projection and normalization.

## One-Page Token And Memory Map

This table is the current mental model for Tau3 SDPO memory. It separates
tokens that shape the forward pass from tokens that actually receive SDPO loss.

| Component | Symbol | Where it comes from | Student input? | Teacher input? | Full-vocab LM head today? | SDPO loss / backprop? | Main control knob | Optimization question |
|---|---:|---|---:|---:|---:|---:|---|---|
| Original task/system/dialogue prompt | `P_fail` | Dataset prompt plus chat template, tools, policy, current user state | Yes | No, replaced by reprompt | Yes, because current packed forward projects prompt+response | No direct loss, but prompt states condition response tokens | `MAX_PROMPT_LENGTH` | Can we avoid projecting prompt positions through LM head and only project response prediction positions? |
| Failed rollout response, including thinking/tool text | `R_fail` | Actor-generated failed trajectory used as SDPO target | Yes | Yes | Yes | Yes for response-mask tokens; gradients update actor/student only | `MAX_RESPONSE_LENGTH`, target guard masks | Should thinking tokens receive SDPO JSD loss, or should we mask/strip parts of reasoning/tool chatter? |
| Teacher reprompt scaffold | `P'_base` | SDPO instruction/template around feedback or peer demo | No | Yes | Yes today | No direct loss | `SDPO_MAX_REPROMPT_LEN` | Can we make teacher context shorter without losing corrective signal? |
| Environment feedback | `F` | Tau3 evaluator feedback / diagnostic JSON for failed rollout | No | Often yes in original arm | Yes if included in `P'_fail` | No direct loss | feedback serialization, `SDPO_MAX_REPROMPT_LEN` | Can note-style summaries preserve the useful correction with fewer tokens? |
| Successful peer demo | `D_success` | First eligible successful rollout from same UID/group, assistant response only after stripping | No | Yes when available | Yes if included in `P'_fail` | No direct loss | peer selection + stripping policy | Should we choose shortest/cleanest/highest-margin success instead of first success? |
| Teacher reprompt total | `P'_fail = P'_base + F/D_success` | Built by SDPO teacher batch construction | No | Yes | Yes today | No direct loss | `SDPO_MAX_REPROMPT_LEN` | Can memory/note SDPO compress `F`/`D_success` while keeping task-specific guidance? |
| Student forward length per row | `T_s = len(P_fail)+len(R_fail)` | Original context plus failed response | Yes | No | `T_s x V` transient | Response positions only | prompt/response caps, logprob microbatch | Can we compute logits only at response prediction positions? |
| Teacher forward length per row | `T_t = len(P'_fail)+len(R_fail)` | Reprompt plus same failed response | No | Yes | `T_t x V` transient | No teacher backprop; probabilities supervise student | reprompt/response caps | Can teacher compute only student top-k token logits instead of full vocab? |
| Per-rank packed tokens | `S_rank = sum(T_i)` | Multiple rollout rows assigned to one DP rank after length balancing | Yes | Yes, separately | `S_rank x V` transient in worst packed/microbatch case | Depends on response masks | `*_MAX_TOKEN_LEN_PER_GPU`, microbatch/dynamic bsz | Should we enforce smaller logprob microbatches or token caps even if slower? |
| Vocabulary dimension | `V` | Model tokenizer / LM head size, about 150K for Qwen3.5 | Yes | Yes | Multiplies every projected position | Needed for exact softmax/top-k today | model choice only | Can chunked vocab projection or selected-vocab projection preserve exact/near-exact SDPO? |
| Distillation support | `K=100 + tail` | Student top-k IDs plus optional residual probability tail | Yes | Teacher gathers on student support | Stored as `response_tokens x K`, after full projection | Yes | `SDPO_DISTILLATION_TOPK`, `SDPO_DISTILLATION_ADD_TAIL` | Is smaller `K`, adaptive `K`, or sampled-token SDPO enough for Tau3? |

Approximate transient BF16 logit memory per forward is:

```text
bytes ~= projected_tokens x vocab_size x 2
```

The important distinction: the **loss mask is response-only**, but the **current
full-vocab projection is prompt-plus-response**. That gap is the main engineering
opening.

## Real Step-10 Scale From P5 Artifact

Artifact: `research/diagnostics/sdpo_vanilla_peer_full.zip`, member
`rollout_data/10.jsonl` plus `full_log.txt`.

Step 10 aggregate metrics:

```text
global rows                         = 64
train_batch_size x rollout.n         = 8 x 8
DP ranks / GPUs                      = 8
rows per rank, approximately         = 8
prompt_length/mean                   = 4204.875
prompt_length/max                    = 4241
response_length/mean                 = 3022.09375
response_length/max                  = 6349
student forward mean per row         = 4204.875 + 3022.09375 = 7226.96875
global_seqlen/mean per rank          = 7226.96875 x 8 = 57815.75
global_seqlen/max per rank           = 79335
teacher_prompt_token_mean            = 2152.625
teacher forward mean per row         = 2152.625 + 3022.09375 = 5174.71875
```

Important interpretation:

- `global_seqlen/*` is a per-DP-rank aggregate after sequence balancing, not
  necessarily one single forward sequence.
- With `log_prob_micro_batch_size_per_gpu=1`, a typical logprob forward is closer
  to one sequence, e.g. around `[7.2K, vocab]` for the student pass in this step.
- If dynamic batching or a larger logprob microbatch is enabled, multiple rows can
  be packed into one forward and the memory peak can approach rank aggregate scale.

## Mathematical Path

For one failed rollout row:

```text
P_fail = original rollout prompt tokens
R_fail = original failed assistant response tokens
P'_fail = SDPO teacher reprompt tokens
V = vocab size
K = distillation top-k, currently 100
```

Student context:

```text
X_student = [P_fail, R_fail]
H_student = Transformer(X_student)
Z_student = LMHead(H_student)
Z_student shape = [len(P_fail) + len(R_fail), V]
```

For response token `r_t`, the prediction position is:

```text
student_position(t) = len(P_fail) - 1 + t
pi_student_t = softmax(Z_student[student_position(t), :])
I_t = topK(pi_student_t, K)
```

Teacher context:

```text
X_teacher = [P'_fail, R_fail]
H_teacher = TeacherTransformer(X_teacher)
Z_teacher = LMHead(H_teacher)
Z_teacher shape = [len(P'_fail) + len(R_fail), V]
```

The teacher does not keep its own independent top-k. It gathers probabilities on
the student top-k IDs:

```text
teacher_position(t) = len(P'_fail) - 1 + t
pi_teacher_t = softmax(Z_teacher[teacher_position(t), :])
teacher_topk_t = pi_teacher_t[I_t]
```

The compared distribution support is:

```text
student top-K IDs + residual tail bucket
```

not:

```text
student top-K union teacher top-K
```

If a token is in the student top-K, the teacher can push it up or down by assigning
high or low probability to that same token. If a teacher-preferred token is not in
the student top-K, it is only represented through the tail bucket, so token-specific
teacher preference is lost.

## LM Head Cost

For decoder LMs:

```text
H shape = [T, d_model]
W_vocab shape = [V, d_model]
b shape = [V]
Z = H W_vocab^T + b
Z shape = [T, V]
```

The prompt must be processed by the transformer because response probabilities are
conditional on the prompt. However, SDPO only needs logits at response prediction
positions:

```text
prompt_last, response_0, ..., response_{n-2}
```

The current implementation computes logits for all unpadded prompt and response
positions, then slices/gathers later. This is correct but memory-wasteful for SDPO:

```text
H = Transformer(prompt + response)
Z = LMHead(H)                         # [prompt + response, V]
use response prediction positions later
```

The ideal shape would be:

```text
H = Transformer(prompt + response)
H_target = H[prompt_last : response_last_minus_1]
Z_target = LMHead(H_target)           # [response, V]
```

That still requires the prompt for attention/context, but avoids prompt-token LM
head projection.

## Current Code Path

Relevant files:

- `verl/trainer/ppo/ray_trainer.py`
- `verl/workers/engine/fsdp/transformer_impl.py`
- `verl/workers/utils/losses.py`
- `verl/workers/utils/padding.py`

High-level pipeline:

```text
1. Student/actor logprob pass
   - input: original prompt + original response
   - computes full logits
   - returns student top-K IDs and logprobs

2. Teacher/ref logprob pass
   - input: teacher reprompt + same original response
   - computes full logits
   - gathers logprobs on student top-K IDs

3. Actor update pass
   - recomputes student logits with gradient
   - gathers teacher IDs
   - applies top-K + tail JSD loss
```

The passes are sequential at the trainer level. Inside each pass, the transformer
forward is parallel over token positions under causal masking.

## Thinking Tokens And Backpropagation

For the failed target trajectory, `<think>` tokens are included if they are
assistant-generated response tokens and survive the target mask. We strip thinking
from the successful peer demonstration inserted into the teacher prompt, not from
the failed response being distilled.

For a thinking token `r_t`, SDPO computes:

```text
pi_student(. | P_fail, r_<t)
pi_teacher(. | P'_fail, r_<t)
```

Then JSD/top-k loss backprops only into the actor/student:

```text
loss
  -> student logprobs
  -> student logits
  -> LM head
  -> transformer hidden states
  -> actor parameters
```

Teacher logprobs are detached and do not receive gradients.

This is why target guards matter. If corrupted open-thinking, repetition, tool-loop,
or response-saturated rows survive the target mask, SDPO can train on those tokens.

## Why Reducing Max Response Length May Still Fail

Reducing `MAX_RESPONSE_LENGTH` lowers the worst-case `T`, but it does not remove
the full-vocab projection pattern. OOM can still occur if:

- one or more responses are long enough to spike `[T, V]`;
- dynamic batching or larger logprob microbatches pack multiple sequences together;
- teacher reprompt plus response length is high;
- actor update recomputes student logits with gradients;
- temporary tensors from `log_softmax`, `topk`, `gather`, or tail/JSD coexist.

Therefore this is a real systems bottleneck for phase-4 Tau3 SDPO on 80GB GPUs,
not just a bad length setting.

## Optimization Candidates

### Candidate A: Response-Only LM Head Projection

Keep full transformer context over prompt plus response, but project only response
prediction positions through the LM head.

Expected benefit:

```text
[prompt + response, V] -> [response, V]
```

This is exact for response-token SDPO and should reduce prompt-logit waste.

Risk:

- requires changing model-output/logprob plumbing;
- must preserve alignment with no-padding and response masks;
- may be harder if HuggingFace forward eagerly returns full logits.

### Candidate B: Chunked Vocab Projection With Streaming Top-K And LogSumExp

Avoid materializing `[T, V]` by scanning vocabulary chunks:

```text
for vocab_chunk:
    z_chunk = H @ W_chunk.T
    update logsumexp accumulator
    update top-K heap / selected-token logits
```

This can be exact if it preserves the full-vocab normalizer:

```text
log p(k) = z_k - logsumexp(z_all_vocab)
```

Expected benefit:

```text
[T, V] -> [T, vocab_chunk]
```

Risk:

- nontrivial engineering;
- must support student top-K selection and teacher gather;
- must integrate with FSDP, tied embeddings, dtype, and possibly tensor parallelism.

### Candidate C: Approximate Selected-K Only Teacher

Compute only:

```text
H_teacher @ W_selected.T
```

shape:

```text
[T, K]
```

This is much cheaper but not exact because it lacks the full-vocab normalizer and
tail mass. It changes the objective and should be treated as an ablation, not
faithful SDPO.

### Candidate D: Sampled-Token SDPO

Use only the generated token logprob instead of distribution-level top-K/JSD.

Benefit:

- much cheaper;
- closer to GRPO-style sampled-token memory.

Cost:

- loses distribution-level self-distillation signal;
- weaker claim relative to full-logit SDPO.

### Candidate E: Stronger Runtime Guards

Reduce pathological `T` before logits:

- lower `MAX_RESPONSE_LENGTH`;
- lower `MAX_MODEL_LEN`;
- lower `SDPO_MAX_REPROMPT_LEN`;
- stop or mask rows with open thinking, repetition, tool loops, or response saturation;
- keep `log_prob_micro_batch_size_per_gpu=1`;
- avoid dynamic logprob batching unless diagnostics prove safe.

This is blunt but currently the safest P5 recipe.

## Metrics To Keep Adding

The recent diagnostics are the right direction. We should track:

```text
sdpo_logprob/student/pre_total_tokens
sdpo_logprob/teacher/pre_total_tokens
sdpo_logprob/*/prompt_tokens
sdpo_logprob/*/response_tokens
sdpo_logprob/*/max_sequence_tokens
sdpo_logprob/*/effective_microbatch_tokens
sdpo_logprob/*/vocab_size
sdpo_cuda_memory/*/allocated_gb
sdpo_cuda_memory/*/reserved_gb
sdpo_cuda_memory/*/max_allocated_gb
sdpo_cuda_memory/*/phase
```

Also keep W&B metrics:

```text
prompt_length/*
response_length/*
global_seqlen/*
self_distillation/teacher_prompt_token_mean
self_distillation/teacher_prompt_saturation_fraction
self_distillation/target_guard_*
actor/self_distillation/empty_target_batch
actor/self_distillation/token_fraction
```

## Current Interpretation

For multi-turn tool agents, faithful full-logit SDPO has a fundamentally higher
memory footprint than GRPO/DAPO-style policy optimization. The method may still be
scientifically valuable because it provides dense teacher-conditioned token-level
learning, especially on terminal-reward sparse multi-turn tasks. But on 80GB GPUs,
the full-vocab projection is now a first-class methodological constraint.

If a safer 8K or 16K SDPO run beats GRPO, that is still a meaningful result. If
full-logit SDPO remains infeasible without chunked LM-head optimization, that is
also a publishable systems observation for adapting self-distillation policy
optimization to long-horizon tool agents.

## Open Questions

- Does response-only LM-head projection reduce enough memory without destabilizing
  FSDP/no-padding plumbing?
- Can we implement exact chunked vocab top-K/logsumexp cleanly in the current engine?
- How much quality do we lose with sampled-token SDPO versus full-logit/top-K JSD?
- Are OOMs dominated by student top-K, teacher gather, or actor update with gradient?
- How often do target guards zero rows after expensive teacher/student logprob work?
- Is a curriculum or hard response cap enough for paper-grade Tau3 SDPO?
