#!/usr/bin/env python3
"""Patch eval_tau3_base_models.py with v3 parser diagnostics.

Adds:
1. A startup sanity check that verifies the parser produces int types for
   numeric Qwen XML parameters (runs in every GPU subprocess).
2. A PRE_ENV_ACTION_FOR_ENV log line immediately before env.step for
   baggage/book_reservation calls, proving whether ints survive to the env
   boundary.

Usage:
    python3 scripts/tau3/patch_eval_v3_diagnostics.py ~/SDPO-qwen35/scripts/eval_tau3_base_models.py

To revert:
    The script creates a .bak backup before patching.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

STARTUP_DIAGNOSTIC = '''
# === V3 PARSER DIAGNOSTIC (added by patch_eval_v3_diagnostics.py) ===
import verl.utils.tau3_action_parser as _parser_mod
import json as _diag_json
print(f"PARSER_FILE={_parser_mod.__file__}", flush=True)
_diag_raw = '<tool_call><function=update_reservation_baggages><parameter=nonfree_baggages>0</parameter><parameter=total_baggages>3</parameter></function></tool_call>'
_diag_action = _diag_json.loads(_parser_mod.parse_model_output_to_tau_action(_diag_raw).action_for_env)["arguments"]
_diag_types = {k: type(v).__name__ for k, v in _diag_action.items()}
print(f"PARSER_TYPES={_diag_types}", flush=True)
print(f"PARSER_VALUES={_diag_action}", flush=True)
assert isinstance(_diag_action["nonfree_baggages"], int), f"FAIL: nonfree_baggages is {type(_diag_action['nonfree_baggages']).__name__}"
assert isinstance(_diag_action["total_baggages"], int), f"FAIL: total_baggages is {type(_diag_action['total_baggages']).__name__}"
print("PARSER_SANITY_CHECK=PASS", flush=True)
# === END V3 PARSER DIAGNOSTIC ===
'''

def _pre_env_diagnostic(indent: str) -> str:
    inner = indent + "    "
    return (
        f'{indent}# === V3 PRE_ENV DIAGNOSTIC ===\n'
        f'{indent}if tool_name in ("update_reservation_baggages", "book_reservation"):\n'
        f'{inner}import json as _pej\n'
        f'{inner}_pre_env_args = _pej.loads(parsed.action_for_env).get("arguments", {{}})\n'
        f'{inner}_pre_env_types = {{k: type(v).__name__ for k, v in _pre_env_args.items()}}\n'
        f'{inner}print(f"PRE_ENV_ACTION_FOR_ENV tool={{tool_name}} args={{_pre_env_args}}", flush=True)\n'
        f'{inner}print(f"PRE_ENV_ARG_TYPES={{_pre_env_types}}", flush=True)\n'
        f'{indent}# === END V3 PRE_ENV DIAGNOSTIC ===\n'
    )


def patch(eval_script: Path) -> None:
    text = eval_script.read_text(encoding="utf-8")

    # Check if already patched
    if "V3 PARSER DIAGNOSTIC" in text:
        print(f"Already patched: {eval_script}")
        return

    # Backup
    backup = eval_script.with_suffix(".py.bak")
    shutil.copy2(eval_script, backup)
    print(f"Backup: {backup}")

    # 1. Insert startup diagnostic after the tau3 parser import. Support both
    # single-line imports and parenthesized multi-line imports.
    marker = "from verl.utils.tau3_action_parser import"
    lines = text.split("\n")
    insert_after = None
    for i, line in enumerate(lines):
        if marker in line:
            if "(" in line and ")" not in line:
                j = i
                while j < len(lines) and ")" not in lines[j]:
                    j += 1
                if j >= len(lines):
                    print(f"ERROR: Could not find end of multi-line tau3 parser import in {eval_script}")
                    sys.exit(1)
                insert_after = j
            else:
                insert_after = i
            break

    if insert_after is None:
        print(f"ERROR: Could not find '{marker}' in {eval_script}")
        sys.exit(1)

    lines.insert(insert_after + 1, STARTUP_DIAGNOSTIC)

    # 2. Insert PRE_ENV diagnostic before Tau3GymLiveSessionManager.step_action.
    # Preserve the external evaluator's indentation instead of assuming a fixed
    # number of spaces.
    text = "\n".join(lines)
    step_pattern = re.compile(
        r"(?m)^(?P<indent>\s*)should_stop,\s*observation,\s*_,\s*additional\s*=\s*"
        r"Tau3GymLiveSessionManager\.step_action\("
    )
    step_match = step_pattern.search(text)
    if step_match is None:
        print(f"ERROR: Could not find step_action call in {eval_script}")
        sys.exit(1)

    text = text[: step_match.start()] + _pre_env_diagnostic(step_match.group("indent")) + text[step_match.start() :]

    eval_script.write_text(text, encoding="utf-8")
    print(f"Patched: {eval_script}")
    print("Diagnostics added:")
    print("  - Startup parser sanity check (asserts int types)")
    print("  - PRE_ENV_ACTION_FOR_ENV log for baggage/book calls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_script", type=Path, help="Path to eval_tau3_base_models.py")
    args = parser.parse_args()

    if not args.eval_script.exists():
        print(f"ERROR: {args.eval_script} does not exist")
        sys.exit(1)

    patch(args.eval_script)


if __name__ == "__main__":
    main()
