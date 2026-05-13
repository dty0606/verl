from __future__ import annotations

import json
from dataclasses import asdict
import os
from typing import Literal

from verl.utils.tau3_diagnostics import TauDiagnostic


def diagnostic_to_dict(diag: TauDiagnostic) -> dict:
    return asdict(diag)


def render_plain_text_feedback(diag: TauDiagnostic) -> str:
    d = diagnostic_to_dict(diag)
    verified_state = "; ".join(d.get("verified_state_summary") or []) or "none confidently verified"
    workflow_status = d.get("workflow_status") or {}
    workflow_line = ", ".join(f"{k}={v}" for k, v in workflow_status.items())
    tool_focus = d.get("tool_focus") or {}
    policy_focus = d.get("policy_or_state_focus") or {}
    priority_focus = d.get("priority_focus") or []

    lines = [
        "Heuristic Tau3 diagnostic feedback:",
        f"Primary failure: {d['primary_failure_type']}.",
        f"Failure stage: {d['failure_stage']}.",
        f"Localized turn: {d['localized_turn_index']}.",
        f"Verified state: {verified_state}.",
        f"Workflow status: {workflow_line}.",
        (
            "Tool focus: "
            f"last_tool={tool_focus.get('last_tool')}, "
            f"failing_tool={tool_focus.get('failing_tool')}, "
            f"expected_next={tool_focus.get('missing_or_expected_tool')}, "
            f"missing_fields={tool_focus.get('missing_fields')}, "
            f"invalid_fields={tool_focus.get('invalid_fields')}."
        ),
        (
            "Policy/state focus: "
            f"policy_issue={policy_focus.get('policy_issue')}, "
            f"state_tracking_issue={policy_focus.get('state_tracking_issue')}, "
            f"entity_or_record_at_risk={policy_focus.get('entity_or_record_at_risk')}."
        ),
    ]
    for item in priority_focus:
        lines.append(
            f"Priority {item.get('priority')}: target={item.get('target')}; "
            f"evidence={item.get('evidence')}; repair={item.get('repair')}"
        )
    lines.extend(
        [
            f"Do not focus on: {', '.join(d.get('do_not_focus_on') or [])}.",
            d.get("teacher_instruction") or "",
            f"diagnostic_version={d['diagnostic_version']}; confidence={d['diagnostic_confidence']}",
        ]
    )
    return "\n".join(line for line in lines if line)


def render_json_feedback(diag: TauDiagnostic) -> dict:
    """Return a dict so RayPPOTrainer can serialize deterministically."""

    return diagnostic_to_dict(diag)


def render_endpoint_boundary_feedback(diag: TauDiagnostic) -> dict:
    d = diagnostic_to_dict(diag)
    endpoint = dict(d.get("endpoint_boundary") or {})
    return {
        "mode": "mt_stepo_endpoint_boundary",
        "eligible": bool(endpoint.get("sdpo_eligible", False)),
        "confidence": endpoint.get("confidence"),
        "failure_bucket": endpoint.get("failure_bucket"),
        "boundary_turn_index": endpoint.get("boundary_turn_index"),
        "verified_state": endpoint.get("verified_state") or [],
        "endpoint_class": endpoint.get("endpoint_class"),
        "allowed_next_actions": endpoint.get("allowed_next_actions") or [],
        "forbidden_next_actions": endpoint.get("forbidden_next_actions") or [],
        "instruction": endpoint.get("repair_instruction")
        or (
            "Use hindsight to identify the first endpoint boundary. "
            "Do not repeat reads or searches already completed. "
            "Choose the next endpoint class instead of another information-gathering step."
        ),
    }


def normalize_feedback_mode(mode: str | None) -> Literal["plain_text", "json", "none"]:
    value = (mode or "none").strip().lower()
    if value in {"text", "plain", "plain_text", "plaintext"}:
        return "plain_text"
    if value in {"structured", "json"}:
        return "json"
    if value in {"none", "off", "false", "0"}:
        return "none"
    raise ValueError(f"Unsupported tau3 feedback mode: {mode}")


def normalize_feedback_renderer(renderer: str | None) -> Literal["diagnostic", "endpoint_boundary"]:
    value = (renderer or os.environ.get("TAU3_FEEDBACK_RENDERER", "diagnostic")).strip().lower()
    if value in {"diagnostic", "default", "full"}:
        return "diagnostic"
    if value in {"endpoint_boundary", "mt_stepo", "endpoint"}:
        return "endpoint_boundary"
    raise ValueError(f"Unsupported tau3 feedback renderer: {renderer}")


def render_feedback(diag: TauDiagnostic, mode: str | None, renderer: str | None = None):
    normalized = normalize_feedback_mode(mode)
    normalized_renderer = normalize_feedback_renderer(renderer)
    if normalized == "plain_text":
        return render_plain_text_feedback(diag)
    if normalized == "json":
        if normalized_renderer == "endpoint_boundary":
            return render_endpoint_boundary_feedback(diag)
        return render_json_feedback(diag)
    return ""
