# Multi-Turn SDPO/GRPO Masking Failure Modes

Status: research note for future paper writing.

## Core Takeaway

Original SDPO is well matched to **wrong-but-scorable** rollouts: a sampled
response may be wrong, but it is still bounded enough that a feedback-conditioned
self-teacher can re-score its tokens and provide dense correction.

Tau3-style thinking-on multi-turn tool agents expose a harder case:
**wrong-and-corrupted** rollouts. A rollout can fail because it enters a runaway
single `<think>` span, loops across multiple assistant turns, repeats tool calls,
or exhausts the response budget with malformed continuation. These trajectories
are not merely incorrect; they may be poor targets for token-level
self-distillation unless explicitly masked or downweighted.

This distinction is likely a transferable failure mode when moving GRPO/SDPO
from single-turn or bounded-response RLVR tasks into long-horizon multi-turn
tool-agent training.

## One-Batch Mental Model

For a batch with `B` prompts and `rollout.n = n`, the effective rollout batch is
`B * n` full trajectories grouped by prompt `uid`.

Each Tau3 trajectory contains:

- User-simulator turns.
- Assistant turns, including visible `<think>...</think>`, tool calls, and final answers.
- Tool results and environment observations.
- Terminal reward and optional environment feedback.

Only assistant-generated response/action tokens should receive policy loss.
User-simulator and tool-result tokens are context.

## Original SDPO Routing

Original SDPO does not train on every rollout equally. It builds a
self-teacher reprompt when a sampled row has corrective teacher context:

- A successful peer rollout for the same prompt `uid`.
- Or environment/rich feedback for the failed attempt.

For a failed rollout `A2` with a successful peer `A1`, the teacher input is:

```text
teacher_prefix =
  original prompt
  + "Correct solution:"
  + A1 successful trajectory as demonstration
  + "Correctly solve the original question."

teacher_input =
  teacher_prefix
  + A2 original sampled response tokens
```

The student input is:

```text
student_input =
  original prompt
  + A2 original sampled response tokens
```

The teacher and student both compute log-probabilities for the same `A2` target
tokens. The teacher does not generate a replacement trajectory.

For an all-fail group with no successful peer, each failed rollout uses its own
environment feedback:

```text
teacher for B1 = original prompt + feedback(B1) + B1 target tokens
teacher for B2 = original prompt + feedback(B2) + B2 target tokens
```

Failed rollouts are not used as "correct solution" demonstrations for each
other unless we explicitly add a new memory/peer-routing mechanism.

## Masking: What Original SDPO Covers

The core SDPO target mask is:

```text
loss_mask = response_mask * self_distillation_mask
```

Where:

- `response_mask` selects generated assistant tokens.
- `self_distillation_mask` selects rows with usable teacher-side corrective context.

Paper-aligned / implementation-aligned behavior:

- Rows with no successful peer and no feedback can be masked out.
- Successful peer demonstrations can have `<think>...</think>` stripped before
  insertion into the teacher prompt.
- The sampled target rollout's own `<think>` tokens are not automatically
  removed by that demonstration-stripping flag.

Important boundary:

```text
demo/context thinking: can be stripped to keep teacher context concise
target/response thinking: remains target tokens unless extra masks are added
```

The paper/repo behavior confirms the stripping flag exists, but the authors do
not appear to explicitly state the motivation for setting it to true. Any
rationale such as "avoid prompt pollution" or "avoid noisy scratchpad
demonstrations" should be presented as our engineering interpretation, not as an
author-stated claim.

## Tau3-Specific Hardening

For Tau3, the active-row rule we want is more defensive than the generic SDPO
principle:

```text
is_failure = reward < failure_threshold
is_not_env_error = not env_error
has_solution = same UID has a successful peer rollout
has_feedback = this rollout has non-empty environment feedback
has_memory = teacher-side memory is enabled and retrieved

self_distillation_mask = 1 if:
  is_failure
  and is_not_env_error
  and (has_solution or usable_feedback or usable_memory)
```

This is partly original SDPO and partly Tau3 adaptation:

- Original SDPO principle: train only when the teacher reprompt has useful
  corrective context, such as successful peer or feedback.
- Tau3 adaptation: exclude env/runtime errors, focus original SDPO targets on
  failed rows, and optionally allow memory notes as another teacher-context
  source.

## Failure Mode: Corrupted Thinking As Target

A failed corrupted rollout can still be selected if it has a successful peer or
feedback:

```text
failed rollout has explosive <think>
feedback exists
self_distillation_mask = 1
response_mask includes assistant thinking tokens
=> corrupted thinking receives SDPO loss unless extra target-side mask exists
```

This creates a failure mode:

- SDPO can clean successful demonstrations while still scoring failed corrupted
  target tokens.
- Long target trajectories can dominate the loss by token count.
- Teacher re-scoring may be noisy when the target is a 20k-token loop rather
  than a bounded wrong answer.
- The method assumes the target response is wrong-but-scorable; Tau3 sometimes
  produces wrong-and-corrupted targets.

## Relation To GRPO/DAPO

GRPO and DAPO do not use teacher reprompts. They update from scalar reward via
policy-gradient losses on sampled assistant tokens.

Key contrast:

- GRPO/DAPO need reward diversity inside the same prompt group. All-fail groups
  provide weak or zero group-relative signal.
- SDPO can use all-fail groups if environment feedback exists.
- Memory/Note-SDPO can further make peer-sparse all-fail groups trainable by
  adding teacher-side retrieved notes.

However, all methods share the same target-side corruption risk when assistant
tokens include runaway thinking or long continuation. Reward/teacher routing
does not by itself solve token-level target hygiene.

## Paper-Writing Claims To Preserve

Safe claims:

- Original SDPO masks samples without usable teacher context; it is not
  successful-rollout-only and not all-rollout training.
- Successful peer thinking is stripped when used as teacher demonstration in the
  official implementation/config.
- Target rollout thinking is not automatically stripped by the demonstration
  stripping flag.
- Multi-turn tool agents introduce a distinct class of corrupted target
  trajectories that require target-side filtering or downweighting.

Claims requiring evidence from our runs:

- SDPO early-stage accuracy can improve faster than GRPO because it gets dense
  teacher-context signal from peer/feedback routing.
- SDPO later collapses because long corrupted targets dominate the token-level
  self-distillation signal.
- Memory/Note-SDPO improves peer-sparse all-fail groups by making teacher
  context more useful.

Avoid overclaiming:

- Do not claim the original SDPO authors intended thinking stripping for a
  specific reason unless we cite an explicit statement.
- Do not claim memory is original SDPO; it is our extension.
- Do not claim actor-visible retrieval unless memory is actually injected into
  actor rollout context. Our current proposal is teacher-side privileged memory.

## Guardrails To Track

Minimum metrics for Tau3 SDPO/GRPO transfer experiments:

- `self_distillation_mask` / selected row fraction.
- Selected token fraction after `response_mask` and target guard masking.
- Env-error excluded fraction.
- Response length and max response length hit rate.
- Open `<think>` at clip.
- Repetition ratio / repeated phrase spans.
- Tool-loop counts and repeated same-tool calls.
- All-fail group fraction.
- Feedback-available and feedback-used fractions.
- Successful-peer-available fraction.
- Memory-used and random-memory-used fractions for Note-SDPO.

Potential target-side hygiene:

- Mask env/runtime-error rows.
- Mask or downweight overlong nonterminal rows.
- Mask open-think-at-clip rows.
- Downweight repeated-loop spans.
- Add stop-boundary / decision-sufficiency auxiliary targets.
- Keep successful demonstration thinking stripped, but treat this as prompt
  hygiene, not enough for target hygiene.

## Current Implementation Status

The Tau3 SDPO code now separates three masks/tensors:

- `self_distillation_mask`: row routing, preserving original SDPO-style
  peer/feedback eligibility.
- `self_distillation_loss_mask`: token mask after Tau3 target-side guardrails.
- `self_distillation_target_token_mask`: explicit combined token target mask
  used by the actor loss.

The faithful-EMA path is explicit:

```text
teacher_backend = ema_ref:
  teacher logprobs are computed with the colocated ref/self-teacher
  actor update runs
  teacher <- (1 - teacher_update_rate) * teacher + teacher_update_rate * actor

teacher_backend = actor_snapshot:
  cheaper latest-VERL approximation using the current actor snapshot
```

This means future paper writing should distinguish:

- Original-style SDPO routing.
- Tau3 target-side guardrails.
- Faithful EMA teacher vs actor-snapshot approximation.
- Memory/Note-SDPO extensions.

## References

- SDPO project page: https://self-distillation.github.io/SDPO
- Official SDPO repo: https://github.com/lasgroup/SDPO
- Local Tau3 memory probe plan: `research/memory_sdpo_teacher_probe_plan.md`
- Local Note-SDPO pipeline: `research/note_sdpo_bank_pipeline.md`
