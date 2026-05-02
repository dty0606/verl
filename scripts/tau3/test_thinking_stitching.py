#!/usr/bin/env python3
"""Quick test: verify thinking trace stitching works on a real trajectory.

Run on P5 (needs full env) or any machine with the SDPO venv:
    python3 scripts/tau3/test_thinking_stitching.py outputs/tau3_protocol_candidates_thinking_v1
"""
import json
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.utils.tau3_data import extract_messages_for_sft

root = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs" / "tau3_protocol_candidates_thinking_v1")
trajs = sorted(glob.glob(f"{root}/*/trajectories/*.json"))

# Find first successful trajectory with thinking
test_traj = None
for f in trajs:
    t = json.load(open(f))
    if float(t.get("final_reward", 0) or 0) < 1.0:
        continue
    if not any(turn.get("thinking_text") for turn in t.get("turns", [])):
        continue
    test_traj = t
    test_file = f
    break

if test_traj is None:
    print("ERROR: No successful trajectory with thinking_text found!")
    sys.exit(1)

print(f"=== Testing: {Path(test_file).name} ===")
print(f"task={test_traj['task_id']} turns={len(test_traj['turns'])}")

# Without thinking traces
msgs_no = extract_messages_for_sft(test_traj, include_system_prompt=False)
asst_no = [m for m in msgs_no if m.get("role") == "assistant"]
think_no = sum(1 for m in asst_no if "<think>" in str(m.get("content", "")))
print(f"\nWithout --include-thinking-traces:")
print(f"  {len(msgs_no)} msgs, {len(asst_no)} assistant, {think_no} with <think>")

# With thinking traces
msgs_yes = extract_messages_for_sft(test_traj, include_system_prompt=False, include_thinking_traces=True)
asst_yes = [m for m in msgs_yes if m.get("role") == "assistant"]
think_yes = sum(1 for m in asst_yes if "<think>" in str(m.get("content", "")))
print(f"\nWith --include-thinking-traces:")
print(f"  {len(msgs_yes)} msgs, {len(asst_yes)} assistant, {think_yes} with <think>")

# Verify the Tau3 assistant greeting was stripped for Qwen3.5 chat-template compatibility.
first_non_system = next(m for m in msgs_yes if m.get("role") != "system")
assert first_non_system.get("role") == "user", f"First non-system role must be user, got {first_non_system.get('role')}"
print(f"\n  Leading assistant greeting stripped: YES")

# Verify thinking text aligns to real assistant turns, not shifted by the stripped greeting.
thinking_texts = [str(turn.get("thinking_text") or "").strip() for turn in test_traj.get("turns", [])]
assert len(asst_yes) == len(thinking_texts), (
    f"Assistant/turn count mismatch after greeting strip: assistant={len(asst_yes)} turns={len(thinking_texts)}"
)
for i, (message, thinking_text) in enumerate(zip(asst_yes, thinking_texts, strict=False)):
    content = str(message.get("content", ""))
    if thinking_text:
        assert thinking_text in content, f"thinking_text[{i}] missing from assistant[{i}] content"
    else:
        assert "<think>" not in content, f"assistant[{i}] unexpectedly has <think> for empty thinking_text"
print("  Thinking trace alignment: YES")

# Show first 3 agent assistant messages.
agent_assts = [m for m in asst_yes if "<think>" in str(m.get("content", ""))]
for i, m in enumerate(agent_assts[:3]):
    content = str(m.get("content", ""))[:400]
    tc = m.get("tool_calls")
    print(f"\n  agent_assistant[{i}]: tool_calls={bool(tc)}")
    print(f"    content[:400]: {content!r}")

# Sanity: thinking traces should not alter message count.
assert len(msgs_no) == len(msgs_yes), f"Message count changed: {len(msgs_no)} vs {len(msgs_yes)}"
print(f"\n  Message count preserved: YES ({len(msgs_yes)})")
print("\nALL CHECKS PASSED")
