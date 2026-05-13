# Start Here For Kiro: Tau3 SDPO Latest-VERL Execution

Date: 2026-05-02

This document is the orientation note for a fresh Kiro/P5 work directory. It explains which repositories matter, why there are two repos, how Codex/Kiro/P5 sync should work, and what to do next.

## Current Kiro/P5 Sync Rule

Kiro/P5 execution uses **S3 snapshots, not Git**. Do not ask P5 to `git pull`,
`git fetch`, or validate code with `git log`. The runnable tree is the snapshot
published to `s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/repo/`.

Create or refresh the execution repo from S3:

```bash
export S3_PREFIX=s3://tianyd-rlvr-research/tau3-sdpo/latest-verl
mkdir -p ~/tau3_sdpo_workspace/verl_tau3_sdpo
aws s3 sync "$S3_PREFIX/repo/" ~/tau3_sdpo_workspace/verl_tau3_sdpo/ \
  --exclude ".git/*" \
  --region us-west-2
```

Suggested remote paths:

```bash
LATEST_VERL_ROOT=~/tau3_sdpo_workspace/verl_tau3_sdpo
```

GitHub/private remotes are local archival conveniences only. If historical
evidence from the old archive is needed, use an existing local copy or request a
narrow artifact; do not make Git cloning part of P5 launch instructions.

## What Each Repo Is For

### Old repo: `dty0606_SDPO`

Remote:

- `https://github.com/dty0606/SDPO.git`

Branch:

- `codex/sdpo-tau3-transformers5-qwen35`

Use this as the archive/source-evidence repo. It contains the historical SDPO paper work, old VERL fork, old TRL SFT experiments, prior P5 recipes, evidence summaries, generated-data notes, and debugging logs.

Do not use it as the active training/RL execution base unless explicitly asked. The old fork repeatedly hit Qwen3.5/Transformers/vLLM/VERL integration boundaries.

Important old-repo files and folders:

- `research/session_sync.md`
- `research/P5_RECIPES.md`
- `research/sft_5k_plan/FULL_PARAM_SFT_5K_PLAN.md`
- `research/evidence/grpo_n8_b12_33steps_garbage/FINDINGS.md`
- `research/evidence/grpo_n8_b12_33steps_garbage.zip`
- old launchers under `scripts/tau3/`
- old Tau3/SDPO implementation patches under `verl/`

Historical evidence to carry forward:

- Qwen/Qwen3.5-4B was the best practical base seed by GPU-hour among tested models.
- LoRA SFT looked good in standalone vLLM eval, but LoRA + old VERL agent-loop rollouts produced corrupted/gibberish decoded output.
- The 1K full-parameter thinking SFT pilot ran and overfit.
- The next real SFT target is 5K to 10K accepted thinking-on airline trajectories before augmentation.
- Canonical Tau3 airline split is 30 train / 20 test; do not generate SFT data from the 20 test tasks.

### New repo: `verl_tau3_sdpo`

Remote:

- `https://github.com/dty0606/verl_tau3_sdpo.git`

Branch:

- `main`

Use this as the active execution repo. It is a private latest-VERL fork with only the minimum Tau3/SDPO carry-over needed for current work.

This repo intentionally removed `.github/workflows/*` for private handoff because the available GitHub token could not push workflow files. That does not affect P5 execution.

Important new-repo files:

- `START_HERE_FOR_KIRO.md`
- `research/session_sync.md`
- `research/migration/latest_verl_tau3_sdpo_plan.md`
- `research/migration/qwen35_vlm_sft_rl_implementation_plan.md`
- `research/migration/sdpo_latest_verl_port_notes.md`
- `research/sft_5k_plan/FULL_PARAM_SFT_5K_PLAN.md`
- `research/P5_RECIPES.md`
- `datasets/tau3_live_airline_canonical_split.json`
- `scripts/tau3/build_protocol_sft_data.py`
- `scripts/tau3/check_thinking_sft_readiness.py`
- `scripts/tau3/run_tau3_verl_sft_full_thinking.sh`
- `run_local_tau3_grpo_live_p5.sh`
- `run_local_tau3_sdpo_live_p5.sh`
- `examples/data_preprocess/tau3_live_multiturn.py`
- `examples/sglang_multiturn/config/tool_config/tau3_live_airline_tool_config.yaml`
- `examples/sglang_multiturn/config/interaction_config/tau3_live_interaction_config.yaml`
- `verl/tools/tau3_live_tool.py`
- `verl/interactions/tau3_live_interaction.py`
- `verl/experimental/agent_loop/tool_agent_loop.py`
- `verl/experimental/agent_loop/tool_parser.py`
- `verl/utils/reward_score/feedback/tau3_live.py`
- `verl/trainer/config/tau3_grpo_live.yaml`

## Why We Need The New Repo

The old repo is useful, but not the active base anymore.

The old path was:

```text
TRL SFT in old repo -> manual Qwen3.5 checkpoint/config surgery -> old VERL GRPO/SDPO
```

That failed around:

- Qwen3.5 text-config vs VLM/composite config mismatch,
- old VERL/vLLM max-model-len and multimodal handling,
- LoRA/vLLM/VERL weight-sync corruption in multi-turn agent rollouts,
- cross-framework checkpoint incompatibility.

The new path is:

```text
latest VERL SFT -> VLM-format Qwen3.5 HF export -> vLLM --language-model-only -> latest VERL GRPO -> latest VERL SDPO
```

The current key decision is:

- train from original `Qwen/Qwen3.5-4B`,
- export/save as VLM-format `qwen3_5` / `Qwen3_5ForConditionalGeneration`,
- serve with vLLM `--language-model-only`,
- avoid `qwen3_5_text` / `Qwen3_5ForCausalLM` checkpoints for rollout until upstream vLLM text-only support is clearly merged and locally verified.

Read `research/migration/qwen35_vlm_sft_rl_implementation_plan.md` before running any SFT or GRPO.

## How Sync Works

### Codex <-> Kiro

Use GitHub as the source of code/document sync.

Recommended loop:

1. Codex edits code/docs locally.
2. Codex commits and pushes to `https://github.com/dty0606/verl_tau3_sdpo.git`.
3. Kiro pulls in the fresh P5 work directory.
4. Kiro runs bounded P5 tasks.
5. Kiro commits concise code/doc updates and pushes back to GitHub.
6. Codex pulls/reviews the Kiro changes.

Do not rely on hidden chat memory. Repo-visible markdown is the shared source of truth.

### Kiro / Laptop <-> SageMaker Code Editor P5

Use S3 sync for large runtime artifacts and generated data, not Git.

Use Git for:

- source code,
- launch scripts,
- concise plans,
- concise result summaries,
- manifests,
- small smoke logs if deliberately promoted.

Use S3 for:

- generated SFT trajectories,
- accepted/rejected trajectory dumps,
- checkpoints,
- rollout data,
- wandb/offline artifacts,
- large logs,
- temporary debug bundles.

The exact bucket/prefix may differ by AWS account. Before running, choose and record a prefix in `research/session_sync.md`, for example:

```bash
S3_PREFIX=s3://<bucket>/tau3-sdpo/latest-verl
```

Typical sync pattern:

```bash
# Pull generated data/checkpoints from P5 to another machine.
aws s3 sync "$S3_PREFIX/generated_sft/" ./generated_sft/
aws s3 sync "$S3_PREFIX/checkpoints/" ./checkpoints/

# Push concise generated-data outputs from laptop/Kiro to P5 if needed.
aws s3 sync ./generated_sft/ "$S3_PREFIX/generated_sft/"
```

Never commit bulky generated data or checkpoints to GitHub.

## Current Work Right Now

The user is currently generating 10K thinking-on SFT trajectories on P5.

Important:

- Use only the 30 canonical train tasks from `datasets/tau3_live_airline_canonical_split.json`.
- Do not generate SFT trajectories from the 20 canonical test tasks.
- The SFT target is 5K to 10K accepted real trajectories before augmentation.
- ID randomization/augmentation may expand rows later, but it does not replace real accepted trajectory diversity.
- The data should include `<think>...</think>` supervision because the best previous SFT baseline used thinking-on trajectories.

After generation, the next P5-side data task is to build latest-VERL SFT parquet:

```bash
python scripts/tau3/build_protocol_sft_data.py \
  --input /path/to/accepted_tau3_train_trajectories \
  --output-dir datasets/tau3_sft_thinking_train_only \
  --train-only-holdout \
  --include-thinking-traces \
  --id-transform randomize \
  --id-variants 3 \
  --include-original-id-variant \
  --overwrite
```

The builder rejects canonical test-task validation by default. If it complains about test-task rows, fix the input root; do not override for final SFT data.

## Immediate Execution Order

Do not start full SFT just because 10K generation is running. First prove the checkpoint/rollout mechanics.

1. Record P5 package versions and git SHA.
2. Run a tiny latest-VERL full-parameter SFT export smoke.
3. Verify `global_step_*/huggingface/config.json` is VLM-format:
   - `model_type == "qwen3_5"`
   - `architectures` contains `Qwen3_5ForConditionalGeneration`
4. Serve that exact `huggingface` checkpoint with standalone vLLM and `--language-model-only`.
5. Confirm output is coherent and not gibberish.
6. Run a one-step GRPO smoke from that checkpoint.
7. Only after the smoke passes, run the real SFT from the 10K generated/accepted data.

The detailed commands are in:

- `research/migration/qwen35_vlm_sft_rl_implementation_plan.md`
- `scripts/tau3/run_tau3_verl_sft_full_thinking.sh`
- `run_local_tau3_grpo_live_p5.sh`

## Environment Control On P5

The P5 environment is the runtime truth.

Before any long run, capture:

```bash
cd ~/tau3_sdpo_workspace/verl_tau3_sdpo
pwd
find . -maxdepth 1 -type f | sort | head -20

python - <<'PY'
import importlib.metadata as md
for pkg in ["torch", "transformers", "vllm", "verl", "liger-kernel"]:
    try:
        print(pkg, md.version(pkg))
    except Exception as exc:
        print(pkg, "UNKNOWN", exc)
PY
nvidia-smi
```

Known environment constraints:

- Latest VERL Qwen3.5 examples in this repo reference vLLM around `0.18.0` and Transformers around `5.3.0`.
- Public vLLM text-only Qwen3.5 support is not yet a safe assumption.
- vLLM VLM-format Qwen3.5 with `--language-model-only` is the intended serving path.
- Liger/FLCE is likely required for long-context full-parameter SFT because Qwen3.5 vocabulary cross entropy can OOM at long sequence lengths.

If the environment differs, record it before changing anything.

## Recipes

New latest-VERL recipe pointers:

- `research/P5_RECIPES.md`
- `research/migration/qwen35_vlm_sft_rl_implementation_plan.md`
- `research/sft_5k_plan/FULL_PARAM_SFT_5K_PLAN.md`
- `scripts/tau3/run_tau3_verl_sft_full_thinking.sh`
- `run_local_tau3_grpo_live_p5.sh`
- `run_local_tau3_sdpo_live_p5.sh`

Old historical recipe pointers:

- `dty0606_SDPO/research/P5_RECIPES.md`
- `dty0606_SDPO/research/sft_5k_plan/FULL_PARAM_SFT_5K_PLAN.md`
- old `scripts/tau3/` launchers

Use old recipes for historical context only. Before copying any old command, check:

- paths,
- branch/repo root,
- S3 prefixes,
- model checkpoint format,
- LoRA assumptions,
- old VERL-specific config keys.

## Do Not Do Yet

- Do not run long GRPO/SDPO until SFT export and one-step GRPO smoke pass.
- Do not use LoRA for the main path.
- Do not patch a text-only checkpoint `config.json` and call it VLM-valid.
- Do not use canonical test tasks for SFT generation or validation.
- Do not bulk-copy old evidence/checkpoints/outputs into the new private repo.
- Do not port old SDPO actor loss hooks wholesale; latest VERL needs a new SDPO integration path.

## Write Back After Each Remote Run

Update these files with concise results:

- `research/session_sync.md`
- `research/migration/qwen35_vlm_sft_rl_implementation_plan.md` if the plan changes
- optionally add a small `research/runs/<run_id>.md` summary for important smoke/full runs

Record:

- git commit SHA,
- P5 package versions,
- exact command,
- S3 prefix for bulky artifacts,
- checkpoint path,
- config shape,
- pass/fail,
- next recommended action.

Keep bulky artifacts in S3, not GitHub.
