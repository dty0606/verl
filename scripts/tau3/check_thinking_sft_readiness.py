#!/usr/bin/env python3
"""Check if thinking-on trajectories are ready for SFT.

Inspects trajectory JSON structure to see if thinking traces are present
and whether extract_messages_for_sft preserves or strips them.
"""
import json
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAJ_ROOT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs" / "tau3_protocol_candidates_thinking_v1")

trajs = sorted(glob.glob(f"{TRAJ_ROOT}/*/trajectories/*.json"))
print(f"Total trajectories: {len(trajs)}")

# Sample first 5 for detailed inspection
for f in trajs[:5]:
    t = json.load(open(f))
    task_id = t.get("task_id")
    reward = t.get("final_reward", 0)
    thinking_mode = t.get("thinking_mode")
    chat_kwargs = t.get("chat_template_kwargs", {})
    enable_thinking = chat_kwargs.get("enable_thinking") if chat_kwargs else None
    turns = t.get("turns", [])

    print(f"\n--- {Path(f).name} ---")
    print(f"  task={task_id} reward={reward} thinking_mode={thinking_mode} enable_thinking={enable_thinking}")
    print(f"  turns={len(turns)}")

    # Check if turns have thinking_text field
    has_thinking_text = 0
    thinking_chars_total = 0
    for turn in turns:
        tt = turn.get("thinking_text", "")
        if tt:
            has_thinking_text += 1
            thinking_chars_total += len(tt)

    print(f"  turns_with_thinking_text: {has_thinking_text}/{len(turns)}")
    print(f"  thinking_chars_total: {thinking_chars_total}")

    # Check if messages exist (simulation_run path)
    tau3_result = t.get("tau3_live_result", {})
    sim_run = tau3_result.get("simulation_run") or tau3_result.get("simulation_run_json")
    has_messages = False
    if isinstance(sim_run, dict):
        msgs = sim_run.get("messages")
        if msgs:
            has_messages = True
            # Check if any assistant message contains <think>
            think_in_msgs = sum(1 for m in msgs if m.get("role") == "assistant" and "<think>" in str(m.get("content", "")))
            print(f"  simulation_run messages: {len(msgs)}, assistant msgs with <think>: {think_in_msgs}")
    elif isinstance(sim_run, str):
        try:
            parsed = json.loads(sim_run)
            msgs = parsed.get("messages", [])
            has_messages = True
            think_in_msgs = sum(1 for m in msgs if m.get("role") == "assistant" and "<think>" in str(m.get("content", "")))
            print(f"  simulation_run messages (from str): {len(msgs)}, with <think>: {think_in_msgs}")
        except:
            pass

    if not has_messages:
        # Check direct messages field
        msgs = t.get("messages")
        if msgs:
            has_messages = True
            think_in_msgs = sum(1 for m in msgs if m.get("role") == "assistant" and "<think>" in str(m.get("content", "")))
            print(f"  direct messages: {len(msgs)}, with <think>: {think_in_msgs}")

    if not has_messages:
        print(f"  NO messages found! Keys: {sorted(t.keys())}")
        if tau3_result:
            print(f"  tau3_live_result keys: {sorted(tau3_result.keys())}")

    # Show first turn raw output snippet
    if turns:
        raw = turns[0].get("raw_model_output", "")[:300]
        think = (turns[0].get("thinking_text") or "")[:200]
        print(f"  turn0 thinking_text[:200]: {think!r}")
        print(f"  turn0 raw_output[:300]: {raw!r}")

# Aggregate stats
print("\n\n=== AGGREGATE ===")
has_thinking_field = 0
has_thinking_content = 0
has_messages_field = 0
has_sim_run = 0
thinking_in_messages = 0

for f in trajs:
    t = json.load(open(f))
    turns = t.get("turns", [])

    any_thinking = any(turn.get("thinking_text") for turn in turns)
    if any(("thinking_text" in turn) for turn in turns):
        has_thinking_field += 1
    if any_thinking:
        has_thinking_content += 1

    tau3_result = t.get("tau3_live_result", {})
    sim_run = tau3_result.get("simulation_run") or tau3_result.get("simulation_run_json")
    if sim_run:
        has_sim_run += 1
        if isinstance(sim_run, dict):
            msgs = sim_run.get("messages", [])
        elif isinstance(sim_run, str):
            try:
                msgs = json.loads(sim_run).get("messages", [])
            except:
                msgs = []
        else:
            msgs = []
        if any("<think>" in str(m.get("content", "")) for m in msgs if m.get("role") == "assistant"):
            thinking_in_messages += 1
    elif t.get("messages"):
        has_messages_field += 1

print(f"Trajectories with thinking_text field in turns: {has_thinking_field}/{len(trajs)}")
print(f"Trajectories with non-empty thinking_text: {has_thinking_content}/{len(trajs)}")
print(f"Trajectories with simulation_run: {has_sim_run}/{len(trajs)}")
print(f"Trajectories with direct messages: {has_messages_field}/{len(trajs)}")
print(f"Trajectories with <think> in simulation_run messages: {thinking_in_messages}/{len(trajs)}")

# Key question: does extract_messages_for_sft include thinking?
print("\n=== EXTRACT_MESSAGES_FOR_SFT TEST ===")
try:
    from verl.utils.tau3_data import extract_messages_for_sft
    # Pick a successful trajectory with thinking
    test_traj = None
    for f in trajs:
        t = json.load(open(f))
        if float(t.get("final_reward", 0) or 0) >= 1.0:
            if any(turn.get("thinking_text") for turn in t.get("turns", [])):
                test_traj = t
                test_file = f
                break

    if test_traj:
        print(f"Testing with: {Path(test_file).name}")
        msgs = extract_messages_for_sft(test_traj, include_system_prompt=True)
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        think_in_extracted = sum(1 for m in assistant_msgs if "<think>" in str(m.get("content", "")))
        print(f"  Extracted {len(msgs)} messages, {len(assistant_msgs)} assistant")
        print(f"  Assistant msgs with <think>: {think_in_extracted}")
        if assistant_msgs:
            first_content = str(assistant_msgs[0].get("content", ""))[:400]
            print(f"  First assistant content[:400]: {first_content!r}")
    else:
        print("No successful trajectory with thinking found!")
except Exception as e:
    print(f"Could not import extract_messages_for_sft: {e}")
    print("This is expected on laptop - the key question is whether thinking_text")
    print("is stored in the trajectory turns but NOT in the simulation_run messages.")
