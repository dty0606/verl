from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.utils.tau3_live_runtime import get_domain_env_and_tasks, scenario_to_instruction_text, to_jsonable


def _parse_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task ids are not allowed: {ids}")
    return ids


def _select_tasks(all_tasks: list, ids: list[str] | None = None, count: int | None = None, offset: int = 0) -> list:
    by_id = {str(getattr(task, "id", idx)): task for idx, task in enumerate(all_tasks)}
    if ids:
        missing = [task_id for task_id in ids if task_id not in by_id]
        if missing:
            raise ValueError(f"Requested task ids not found: {missing}. Available sample: {list(by_id)[:10]}")
        return [by_id[task_id] for task_id in ids]

    if count is None or count < 0:
        return all_tasks[offset:]
    return all_tasks[offset : offset + count]


def build_rows(
    *,
    domain: str,
    split_name: str,
    tasks: list,
    system_prompt: str,
    interaction_name: str,
    runtime: str,
    feedback_mode: str,
):
    rows = []
    for idx, task in enumerate(tasks):
        task_payload = to_jsonable(task)
        task_id = str(task_payload.get("id", getattr(task, "id", idx)))
        user_instruction = scenario_to_instruction_text(getattr(task, "user_scenario", None))
        rows.append(
            {
                "data_source": "tau3_live",
                "agent_name": "tool_agent",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_instruction},
                ],
                "ability": "tau3_live",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(task_payload, ensure_ascii=False),
                },
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    "teacher_feedback_format": feedback_mode,
                    "interaction_kwargs": {
                        "name": interaction_name,
                        "runtime": runtime,
                        "domain": domain,
                        "task_id": task_id,
                        "task_split": split_name,
                    },
                },
            }
        )
    return rows


if __name__ == "__main__":
    import datasets

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", default="airline", choices=["airline", "retail", "telecom"])
    parser.add_argument("--train-count", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--train-task-ids", default="", help="Comma-separated explicit train task ids.")
    parser.add_argument("--test-task-ids", default="", help="Comma-separated explicit eval task ids.")
    parser.add_argument(
        "--split-manifest",
        default=str(ROOT / "datasets" / "tau3_live_airline_canonical_split.json"),
        help="Canonical split JSON with train_task_ids/test_task_ids. Empty string disables manifest loading.",
    )
    parser.add_argument("--interaction-name", default="tau3_live")
    parser.add_argument("--runtime", default=os.environ.get("TAU3_LIVE_RUNTIME", "official_gym"))
    parser.add_argument("--feedback-mode", default=os.environ.get("TAU3_LIVE_FEEDBACK_FORMAT", "json"))
    args = parser.parse_args()

    get_environment, get_tasks = get_domain_env_and_tasks(args.domain)
    env = get_environment()
    base_system_prompt = (
        env.get_policy()
        if hasattr(env, "get_policy")
        else "You are a customer-support agent. Help the user while following policy and using tools when needed."
    )
    tool_call_instruction = (
        "\n\nAdditional tool-use requirements:\n"
        "- If thinking mode is enabled by the chat template, keep the final actionable response/tool call clean.\n"
        "- When using a tool, output exactly one valid tool call and nothing else outside the valid tool call.\n"
        "- Follow the tool-call format required by the model/chat template.\n"
        "- Do not include markdown fences.\n"
        "- Use the provided tool names and required argument names exactly."
    )
    system_prompt = f"{base_system_prompt}{tool_call_instruction}"
    all_tasks = sorted(get_tasks(), key=lambda item: int(getattr(item, "id", 0)))

    train_ids = _parse_ids(args.train_task_ids)
    test_ids = _parse_ids(args.test_task_ids)
    split_manifest = None
    if not train_ids and not test_ids and args.split_manifest:
        manifest_path = Path(args.split_manifest).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        if manifest_path.exists():
            split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train_ids = [str(item) for item in split_manifest.get("train_task_ids", [])]
            test_ids = [str(item) for item in split_manifest.get("test_task_ids", [])]

    if train_ids or test_ids:
        overlap = sorted(set(train_ids) & set(test_ids))
        if overlap:
            raise ValueError(f"Train/test task ids must be disjoint, found overlap: {overlap}")
        train_tasks = _select_tasks(all_tasks, ids=train_ids)
        test_tasks = _select_tasks(all_tasks, ids=test_ids)
    else:
        if args.train_count + args.test_count > len(all_tasks):
            raise ValueError(
                f"Requested {args.train_count + args.test_count} tasks but only found {len(all_tasks)} tasks in {args.domain}."
            )
        train_tasks = _select_tasks(all_tasks, count=args.train_count, offset=0)
        test_tasks = _select_tasks(all_tasks, count=args.test_count, offset=args.train_count)
        print("WARNING: using sorted first-N/next-M task slicing; prefer explicit official task ids for final claims.")

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    common = dict(
        domain=args.domain,
        system_prompt=system_prompt,
        interaction_name=args.interaction_name,
        runtime=args.runtime,
        feedback_mode=args.feedback_mode,
    )

    train_dataset = datasets.Dataset.from_list(build_rows(split_name="train", tasks=train_tasks, **common))
    test_dataset = datasets.Dataset.from_list(build_rows(split_name="test", tasks=test_tasks, **common))

    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(output_dir, "test.parquet"))

    metadata = {
        "domain": args.domain,
        "runtime": args.runtime,
        "feedback_mode": args.feedback_mode,
        "train_task_ids": [str(getattr(t, "id", "")) for t in train_tasks],
        "test_task_ids": [str(getattr(t, "id", "")) for t in test_tasks],
        "split_manifest": args.split_manifest if split_manifest is not None else None,
        "split_note": "Use the 30 official train tasks for training data generation; keep the 20 test tasks held out.",
    }
    Path(output_dir, "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote tau3 live dataset to {output_dir}")
    print(f"Runtime: {args.runtime}")
    print(f"Feedback mode: {args.feedback_mode}")
    print(f"Train tasks: {len(train_tasks)} {[str(getattr(t, 'id', '')) for t in train_tasks[:10]]}")
    print(f"Test tasks:  {len(test_tasks)} {[str(getattr(t, 'id', '')) for t in test_tasks[:10]]}")
