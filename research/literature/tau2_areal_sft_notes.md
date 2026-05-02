# AReaL Tau2 SFT Precedent

Last updated: 2026-05-02

## Why This Matters

Our Qwen3.5 thinking-on SFT path cannot safely train every assistant
`<think>` span from one full rendered trajectory. The Qwen3.5 chat template
intentionally strips historical assistant reasoning in multi-turn rendering.

AReaL Tau2 is the closest public precedent for our setting: customer-service
tool agents trained through SFT then verifiable-reward RL on tau-style domains.

## Public Format

Dataset: `inclusionAI/AReaL-tau2-data`

Paper: `From Self-Evolving Synthetic Data to Verifiable-Reward RL:
Post-Training Multi-turn Interactive Tool-Using Agents`

The dataset card describes `tau2_sft_train.jsonl` as one assistant turn in
context:

```json
{
  "messages": [
    {"role": "system", "content": "...policy and tools..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "tool", "content": "..."}
  ],
  "answer": {
    "role": "assistant",
    "content": "...",
    "thinking": "...",
    "tool_calls": []
  },
  "metadata": {
    "source_dialog_id": "airline_dialog_42",
    "turn_index": 2,
    "correct": 1,
    "reward": 1.0
  }
}
```

This supports our local `TurnSFTDataset` design:

```text
prior messages/history -> current assistant answer
```

## Local Decision

Use turn-per-row SFT for Qwen3.5 thinking traces:

- `messages`: prior conversation history only.
- `answer`: current assistant target only, with `thinking`, `content`, and
  optional `tool_calls`.
- Loss mask: target assistant suffix only.
- Historical assistant reasoning may be stripped from `messages`; it was
  already supervised when that assistant turn was the row target.

This keeps training aligned with rollout inference: given history, generate the
next assistant action.

## Related Precedent

- VERL multi-turn docs explicitly warn that Qwen/QwQ/Qwen3 reasoning templates
  remove internal reasoning from historical turns and can break delta
  tokenization.
- ReTool and ReTool-style multi-turn datasets support the broader pattern of
  cold-start SFT before multi-turn tool RL, but AReaL Tau2 is the closest match
  to our tau customer-service domains.

## Sources

- https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data
- https://arxiv.org/pdf/2601.22607
- https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html
- https://huggingface.co/datasets/swordfaith/ReTool-SFT-multi-turn
