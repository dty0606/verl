# Teacher-Side Memory-SDPO Probe Plan

Status: first implementation slice added behind config flags.

## Goal

Test whether compact train-only correction memory improves the original SDPO teacher on peer-sparse Tau3 failures before spending a full P5 training run.

The deployed actor/student context is unchanged. Memory is privileged teacher context only:

```text
actor/student:
  same Tau3 prompt/history used by GRPO and original SDPO

teacher:
  same original SDPO reprompt
  + successful peer when available
  + compact train-only memory card when enabled and no successful peer exists

loss:
  same SDPO response-token distillation path
```

## Files

- `verl/utils/tau3_sdpo_memory.py` loads JSONL/JSON memory cards, strips raw thinking, and retrieves either relevant or random cards.
- `research/memory_cards/tau3_airline_seed_cards.jsonl` contains small hand-written train-only workflow cards.
- `scripts/tau3/build_sdpo_memory_teacher_probe.py` builds offline T0/T1/T2 teacher prompts from failed rollout JSONLs.
- `verl/trainer/config/tau3_sdpo_live.yaml` adds `tau3.sdpo.memory.*` config.
- `run_local_tau3_sdpo_live_p5.sh` forwards `SDPO_MEMORY_*` env vars into Hydra.

## Offline Probe

Build prompt variants from pulled rollout JSONLs:

```bash
python scripts/tau3/build_sdpo_memory_teacher_probe.py \
  --rollout /path/to/rollout_data \
  --memory-path research/memory_cards/tau3_airline_seed_cards.jsonl \
  --output-jsonl /tmp/tau3_memory_teacher_probe_prompts.jsonl \
  --max-samples 100
```

Prompt arms:

```text
T0_original: original SDPO feedback context only
T1_relevant_memory: original SDPO + relevant compact memory card
T2_random_memory: original SDPO + random compact memory card
```

Green-light a tiny Memory-SDPO run only if T1 produces cleaner, more actionable teacher outputs than T0 and T2 on hard workflow prefixes.

## Training Toggle

Enable the live teacher-side memory path:

```bash
export SDPO_MEMORY_ENABLED=true
export SDPO_MEMORY_PATH=research/memory_cards/tau3_airline_seed_cards.jsonl
export SDPO_MEMORY_MODE=relevant
export SDPO_MEMORY_INJECT_WHEN=no_solution
export SDPO_ARM=original
bash run_local_tau3_sdpo_live_p5.sh "$TASK_PATH" memory_sdpo_probe json
```

Useful W&B metrics:

```text
self_distillation/memory_available_fraction
self_distillation/memory_used_fraction
self_distillation/memory_random_used_fraction
self_distillation/memory_no_solution_used_fraction
self_distillation/memory_section_char_mean
self_distillation/teacher_prompt_token_mean
self_distillation/teacher_prompt_saturation_fraction
```

## Controls

Run at least:

```text
A0: original SDPO, memory disabled
A1: original SDPO + relevant memory
A2: original SDPO + random memory
```

Later controls:

```text
A3: raw retrieved trajectory memory
A4: matched-token irrelevant card
A5: same task-cluster blocked retrieval
```

Do not claim memory helps unless relevant memory beats random memory on hard workflow clusters without increasing response clipping or teacher prompt saturation.
