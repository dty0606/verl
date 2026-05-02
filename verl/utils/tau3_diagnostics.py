from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


FORBIDDEN_DIAGNOSTIC_KEYS = {
    "correct_user_id",
    "correct_action_sequence",
    "expected_action_sequence",
    "target_database_state",
    "ground_truth",
    "expected_final_answer",
    "correct_final_answer",
}

FAILURE_TYPE_ORDER = [
    "incomplete_workflow",
    "read_and_stall",
    "unsupported_final_claim",
    "unverified_write_action",
    "wrong_write_action",
    "invalid_tool_arguments",
    "missing_required_argument",
    "policy_grounding_error",
    "state_tracking_error",
    "invalid_escalation",
    "valid_escalation_or_transfer",
    "asks_user_instead_of_finishing",
    "length_or_step_budget_failure",
    "unknown",
]

READ_TOOL_PATTERNS = (
    "get_",
    "search_",
    "list_",
    "calculate",
)

WRITE_TOOL_PATTERNS = (
    "cancel_",
    "update_",
    "book_",
    "modify_",
    "return_",
    "exchange_",
)

TRANSFER_TOOL_NAMES = {"transfer_to_human_agents"}
SUCCESS_TERMINAL_TYPES = {"user_stop", "done", "agent_or_user_stop"}
POSITIVE_CLAIM_PATTERNS = [
    r"\b(?:successfully\s+)?cancel(?:led|ed)\b",
    r"\bcancellation (?:is|was) complete\b",
    r"\brefund (?:has been|was) processed\b",
    r"\brefund has been issued\b",
    r"\b(?:successfully\s+)?refund(?:ed)?\b",
    r"\b(?:successfully\s+)?rebook(?:ed)?\b",
    r"\b(?:successfully\s+)?update(?:d)?\b",
    r"\breservation (?:has been )?(?:updated|changed|booked)\b",
    r"\bbooking (?:is|was) complete\b",
]
REQUEST_MORE_INFO_PATTERNS = [
    r"\bcould you\b",
    r"\bcan you provide\b",
    r"\bplease provide\b",
    r"\bconfirm\b",
    r"\bwhat is your\b",
    r"\bwhich\b",
    r"\bdo you want\b",
    r"\bwould you like\b",
]
POLICY_GROUNDING_PATTERNS = [
    r"\bpolicy\b",
    r"\bnot eligible\b",
    r"\bnot allowed\b",
    r"\binsurance\b",
    r"\bbasic economy\b",
    r"\bnot refundable\b",
    r"\bunable to process\b",
]
ARGUMENT_ERROR_PATTERNS = [
    r"missing required argument[s]?:?\s*([a-zA-Z0-9_\[\]\.]+)?",
    r"invalid argument[s]?:?\s*([a-zA-Z0-9_\[\]\.]+)?",
    r"expected\s+([a-zA-Z0-9_]+)\s+got\s+([a-zA-Z0-9_]+)",
    r"validation error",
    r"schema validation error",
]


@dataclass
class TauDiagnostic:
    diagnostic_version: str
    success: bool
    reward: float
    primary_failure_type: str
    failure_stage: str
    localized_turn_index: int | None
    verified_state_summary: list[str]
    workflow_status: dict[str, bool]
    tool_focus: dict[str, Any]
    policy_or_state_focus: dict[str, Any]
    priority_focus: list[dict[str, Any]]
    do_not_focus_on: list[str]
    teacher_instruction: str
    diagnostic_confidence: str
    failed_components: list[str]
    endpoint_boundary: dict[str, Any] | None
    trace_evidence: dict[str, Any]


def parse_json_maybe(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def compact_text(text: str, limit: int = 280) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _strip_think(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", " ", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    return " ".join(cleaned.split())


def _contains_any_regex(text: str, patterns: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def _json_text(*payloads: Any) -> str:
    parts: list[str] = []
    for payload in payloads:
        if payload is None:
            continue
        if isinstance(payload, str):
            parts.append(payload)
        else:
            try:
                parts.append(json.dumps(payload, ensure_ascii=False))
            except Exception:
                parts.append(str(payload))
    return " ".join(parts).lower()


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _tool_role(name: str | None) -> str:
    tool = str(name or "").strip().lower()
    if not tool:
        return "none"
    if tool in TRANSFER_TOOL_NAMES:
        return "transfer"
    if tool == "done":
        return "done"
    if tool.startswith(WRITE_TOOL_PATTERNS):
        return "write"
    if tool.startswith(READ_TOOL_PATTERNS) or tool.endswith("_status") or tool.endswith("_details"):
        return "read"
    return "other"


def _tool_summary_phrase(name: str) -> str:
    tool = str(name or "").strip().lower()
    mapping = {
        "get_user_details": "user details retrieved",
        "get_reservation_details": "reservation details retrieved",
        "get_flight_status": "flight status retrieved",
        "search_direct_flight": "candidate direct flights searched",
        "search_onestop_flight": "candidate one-stop flights searched",
        "transfer_to_human_agents": "human transfer initiated",
        "get_order_details": "order details retrieved",
        "get_product_details": "product details retrieved",
        "get_user_profile": "user profile retrieved",
        "list_all_airports": "airport options retrieved",
        "calculate": "cost estimate computed",
    }
    if tool in mapping:
        return mapping[tool]
    if _tool_role(tool) == "write":
        return tool.replace("_", " ") + " attempted"
    if _tool_role(tool) == "read":
        return tool.replace("_", " ") + " retrieved"
    return tool.replace("_", " ")


def _last_tool_by_role(tool_calls: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    for call in reversed(tool_calls):
        if _tool_role(call.get("name")) == role:
            return call
    return None


def _extract_simulation_tool_calls(simulation_run: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for message in simulation_run.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            tool_calls.append(
                {
                    "name": call.get("name"),
                    "arguments": call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                    "argument_keys": sorted((call.get("arguments") or {}).keys()) if isinstance(call.get("arguments"), dict) else [],
                    "role": _tool_role(call.get("name")),
                    "success": None,
                    "turn_idx": message.get("turn_idx"),
                }
            )
    return tool_calls


def _extract_live_tool_calls(live_result: dict[str, Any], simulation_run: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer compact runtime tool events, then enrich/fall back with simulation/executed tool data."""

    simulation_calls = _extract_simulation_tool_calls(simulation_run)
    tool_events = live_result.get("tool_events") or []
    if isinstance(tool_events, list) and tool_events:
        merged_calls: list[dict[str, Any]] = []
        next_sim_idx = 0
        for idx, event in enumerate(tool_events):
            if not isinstance(event, dict):
                continue
            name = event.get("name")
            if not name:
                continue
            merged = {
                "name": name,
                "arguments": event.get("arguments") if isinstance(event.get("arguments"), dict) else {},
                "argument_keys": sorted((event.get("arguments") or {}).keys())
                if isinstance(event.get("arguments"), dict)
                else list(event.get("argument_keys") or []),
                "role": event.get("role") or _tool_role(name),
                "success": event.get("success"),
                "turn_idx": event.get("turn_idx", idx + 1),
            }

            matched_sim_call = None
            for sim_idx in range(next_sim_idx, len(simulation_calls)):
                candidate = simulation_calls[sim_idx]
                if candidate.get("name") == name:
                    matched_sim_call = candidate
                    next_sim_idx = sim_idx + 1
                    break
            if matched_sim_call is not None:
                if matched_sim_call.get("arguments"):
                    merged["arguments"] = matched_sim_call["arguments"]
                    merged["argument_keys"] = sorted(matched_sim_call["arguments"].keys())
                if matched_sim_call.get("turn_idx") is not None:
                    merged["turn_idx"] = matched_sim_call["turn_idx"]

            merged_calls.append(merged)
        return merged_calls

    if simulation_calls:
        return simulation_calls

    executed_tools = live_result.get("executed_tools") or []
    if isinstance(executed_tools, list) and executed_tools:
        return [
            {
                "name": name,
                "arguments": {},
                "argument_keys": [],
                "role": _tool_role(name),
                "success": True,
                "turn_idx": None,
            }
            for name in executed_tools
            if name
        ]

    return []


def _infer_failed_components(reward_info: dict[str, Any], live_result: dict[str, Any]) -> list[str]:
    reward = float(live_result.get("final_reward", live_result.get("reward", 0.0)) or 0.0)
    if reward >= 1.0:
        return []

    failed: list[str] = []
    if live_result.get("last_parse_error"):
        failed.append("ACTION")
    action_checks = reward_info.get("action_checks")
    if isinstance(action_checks, list) and action_checks:
        failed.append("ACTION")
    communicate_checks = reward_info.get("communicate_checks")
    communicate_info = ((reward_info.get("info") or {}).get("communicate") or {}) if isinstance(reward_info.get("info"), dict) else {}
    if (isinstance(communicate_checks, list) and communicate_checks) or communicate_info:
        failed.append("COMMUNICATION")
    db_check = reward_info.get("db_check")
    env_assertions = reward_info.get("env_assertions")
    if db_check is not None or (isinstance(env_assertions, list) and env_assertions):
        failed.append("ENV")
    nl_assertions = reward_info.get("nl_assertions")
    if (isinstance(nl_assertions, list) and nl_assertions) or isinstance(nl_assertions, dict):
        failed.append("NL_ASSERTION")
    info_note = _json_text((reward_info.get("info") or {}).get("note") if isinstance(reward_info.get("info"), dict) else reward_info.get("info"))
    if "termination" in info_note or "max_steps" in info_note or "running" in info_note:
        failed.append("TERMINATION")
    terminal_reason = str(live_result.get("terminal_reason", "") or "").lower()
    if terminal_reason in {"max_user_turns", "max_steps", "truncated", "parse_error", "running", "context_overflow"}:
        failed.append("TERMINATION")
    return sorted(set(failed)) or ["UNKNOWN"]


def _extract_argument_error_fields(text: str) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    invalid: list[str] = []
    lowered = str(text or "")
    for pattern in ARGUMENT_ERROR_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            groups = [g for g in match.groups() if g]
            if "missing required" in pattern and groups:
                missing.append(groups[0])
            elif "invalid argument" in pattern and groups:
                invalid.append(groups[0])
            elif "expected" in pattern and groups:
                invalid.append("type_mismatch")
            elif "validation error" in pattern or "schema validation error" in pattern:
                invalid.append("schema_validation")
    return _unique_preserve_order(missing), _unique_preserve_order(invalid)


def _infer_primary_failure_type(
    *,
    reward: float,
    live_result: dict[str, Any],
    reward_info: dict[str, Any],
    simulation_run: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> str:
    if reward >= 1.0:
        if any(_tool_role(call.get("name")) == "transfer" for call in tool_calls):
            return "valid_escalation_or_transfer"
        return "unknown"

    terminal_reason = str(live_result.get("terminal_reason", "") or "").lower()
    final_text = _strip_think(live_result.get("latest_assistant_message", ""))
    reward_text = _json_text(reward_info, simulation_run.get("info"), live_result.get("last_parse_error"))
    db_match = bool((reward_info.get("db_check") or {}).get("db_match"))
    write_calls = [call for call in tool_calls if _tool_role(call.get("name")) == "write"]
    read_calls = [call for call in tool_calls if _tool_role(call.get("name")) == "read"]
    transfer_calls = [call for call in tool_calls if _tool_role(call.get("name")) == "transfer"]
    last_tool_role = _tool_role(tool_calls[-1].get("name")) if tool_calls else "none"

    if (
        terminal_reason in {"running", "max_steps", "max_user_turns", "context_overflow", "truncated"}
        or "max_steps" in reward_text
        or "context overflow" in reward_text
        or "terminated prematurely" in reward_text
    ):
        return "length_or_step_budget_failure"

    if live_result.get("last_parse_error"):
        if "missing" in str(live_result.get("last_parse_error")).lower() and "argument" in str(live_result.get("last_parse_error")).lower():
            return "missing_required_argument"
        return "invalid_tool_arguments"

    if "missing required argument" in reward_text:
        return "missing_required_argument"
    if (
        "invalid argument" in reward_text
        or "expected list got string" in reward_text
        or "validation error" in reward_text
        or "schema validation error" in reward_text
        or "type error" in reward_text
    ):
        return "invalid_tool_arguments"

    if _contains_any_regex(final_text, POSITIVE_CLAIM_PATTERNS) and not db_match and not write_calls:
        return "unsupported_final_claim"

    if write_calls and not db_match:
        if _contains_any_regex(final_text, POSITIVE_CLAIM_PATTERNS):
            return "unsupported_final_claim"
        if last_tool_role == "write":
            return "unverified_write_action"
        return "wrong_write_action"

    if len(read_calls) >= 2 and not write_calls and not transfer_calls:
        return "read_and_stall"
    if last_tool_role == "read" and not write_calls and not transfer_calls:
        return "incomplete_workflow"

    if _contains_any_regex(final_text, REQUEST_MORE_INFO_PATTERNS):
        return "asks_user_instead_of_finishing"

    if transfer_calls:
        return "invalid_escalation"

    if _contains_any_regex(final_text, POLICY_GROUNDING_PATTERNS):
        return "policy_grounding_error"

    reservation_ids = {
        str((call.get("arguments") or {}).get("reservation_id"))
        for call in tool_calls
        if (call.get("arguments") or {}).get("reservation_id") is not None
    }
    order_ids = {
        str((call.get("arguments") or {}).get("order_id"))
        for call in tool_calls
        if (call.get("arguments") or {}).get("order_id") is not None
    }
    if len(reservation_ids) > 1 or len(order_ids) > 1:
        return "state_tracking_error"

    return "unknown"


def _infer_failure_stage(primary_failure_type: str, tool_calls: list[dict[str, Any]]) -> str:
    if primary_failure_type in {"length_or_step_budget_failure", "read_and_stall", "incomplete_workflow"}:
        return "information_gathering"
    if primary_failure_type in {"missing_required_argument", "invalid_tool_arguments", "wrong_write_action", "unverified_write_action"}:
        return "action_execution"
    if primary_failure_type in {"unsupported_final_claim", "asks_user_instead_of_finishing"}:
        return "final_response"
    if primary_failure_type in {"policy_grounding_error", "invalid_escalation", "valid_escalation_or_transfer"}:
        return "policy_or_escalation"
    if primary_failure_type == "state_tracking_error":
        return "state_tracking"
    if any(_tool_role(call.get("name")) == "write" for call in tool_calls):
        return "write_or_verification"
    return "unknown"


def _build_verified_state_summary(tool_calls: list[dict[str, Any]], live_result: dict[str, Any]) -> list[str]:
    summaries = [_tool_summary_phrase(call.get("name") or "") for call in tool_calls if _tool_role(call.get("name")) in {"read", "write", "transfer"}]
    if float(live_result.get("final_reward", 0.0) or 0.0) >= 1.0:
        summaries.append("final task state verified by evaluator")
    return _unique_preserve_order(summaries)[:4]


def _build_workflow_status(
    *,
    reward: float,
    reward_info: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    primary_failure_type: str,
) -> dict[str, bool]:
    read_calls = [call for call in tool_calls if _tool_role(call.get("name")) == "read"]
    write_calls = [call for call in tool_calls if _tool_role(call.get("name")) == "write"]
    transfer_calls = [call for call in tool_calls if _tool_role(call.get("name")) == "transfer"]
    db_match = bool((reward_info.get("db_check") or {}).get("db_match"))
    return {
        "information_gathered": bool(read_calls),
        "required_decision_made": bool(write_calls or transfer_calls or primary_failure_type in {"policy_grounding_error", "valid_escalation_or_transfer", "invalid_escalation"}),
        "required_write_action_completed": bool(write_calls and db_match),
        "final_state_verified": bool(reward >= 1.0 or db_match),
    }


def _extract_entity_or_record_at_risk(tool_calls: list[dict[str, Any]]) -> str | None:
    for key in ("reservation_id", "order_id", "user_id", "payment_id", "product_id"):
        for call in reversed(tool_calls):
            value = (call.get("arguments") or {}).get(key)
            if value not in {None, ""}:
                return f"{key}={value}"
    return None


def _state_tracking_issue_from_tool_calls(tool_calls: list[dict[str, Any]]) -> str | None:
    reservation_ids = sorted(
        {
            str((call.get("arguments") or {}).get("reservation_id"))
            for call in tool_calls
            if (call.get("arguments") or {}).get("reservation_id") is not None
        }
    )
    order_ids = sorted(
        {
            str((call.get("arguments") or {}).get("order_id"))
            for call in tool_calls
            if (call.get("arguments") or {}).get("order_id") is not None
        }
    )
    user_ids = sorted(
        {
            str((call.get("arguments") or {}).get("user_id"))
            for call in tool_calls
            if (call.get("arguments") or {}).get("user_id") is not None
        }
    )
    if len(reservation_ids) > 1:
        return f"multiple reservation_ids referenced: {reservation_ids}"
    if len(order_ids) > 1:
        return f"multiple order_ids referenced: {order_ids}"
    if len(user_ids) > 1:
        return f"multiple user_ids referenced: {user_ids}"
    return None


def _build_tool_focus(
    *,
    primary_failure_type: str,
    reward_info: dict[str, Any],
    live_result: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    last_tool_call = tool_calls[-1] if tool_calls else None
    last_tool = last_tool_call.get("name") if last_tool_call else None
    last_write_call = _last_tool_by_role(tool_calls, "write")
    parse_error = str(live_result.get("last_parse_error") or "")
    error_text = _json_text(reward_info, parse_error)
    missing_fields, invalid_fields = _extract_argument_error_fields(error_text)

    failing_tool = None
    if primary_failure_type in {
        "invalid_tool_arguments",
        "missing_required_argument",
        "wrong_write_action",
        "unverified_write_action",
    }:
        if primary_failure_type in {"wrong_write_action", "unverified_write_action"} and last_write_call is not None:
            failing_tool = last_write_call.get("name")
        else:
            failing_tool = last_tool

    expected_tool = None
    if primary_failure_type in {"length_or_step_budget_failure", "read_and_stall", "incomplete_workflow"}:
        expected_tool = "required state-changing or final policy action"
    elif primary_failure_type in {"unsupported_final_claim", "unverified_write_action", "wrong_write_action"}:
        expected_tool = "verified backend-changing action plus confirmation"
    elif primary_failure_type == "invalid_escalation":
        expected_tool = "task-completing action or policy-justified transfer"
    elif primary_failure_type == "asks_user_instead_of_finishing":
        expected_tool = "next required tool or policy action"

    return {
        "last_tool": last_tool,
        "failing_tool": failing_tool,
        "missing_or_expected_tool": expected_tool,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
    }


def _build_policy_or_state_focus(
    *,
    primary_failure_type: str,
    final_text: str,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    policy_issue = None
    state_tracking_issue = _state_tracking_issue_from_tool_calls(tool_calls)
    if primary_failure_type in {"policy_grounding_error", "invalid_escalation"}:
        policy_issue = compact_text(final_text, limit=180)
    if primary_failure_type == "state_tracking_error" and state_tracking_issue is None:
        state_tracking_issue = "multiple entities appear to have been mixed during the rollout"
    return {
        "policy_issue": policy_issue,
        "state_tracking_issue": state_tracking_issue,
        "entity_or_record_at_risk": _extract_entity_or_record_at_risk(tool_calls),
    }


def _build_endpoint_boundary(
    *,
    primary_failure_type: str,
    live_result: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    workflow_status: dict[str, bool],
    final_text: str,
) -> dict[str, Any]:
    read_calls = [call for call in tool_calls if _tool_role(call.get("name")) == "read"]
    write_calls = [call for call in tool_calls if _tool_role(call.get("name")) == "write"]
    transfer_calls = [call for call in tool_calls if _tool_role(call.get("name")) == "transfer"]
    verified_state = _build_verified_state_summary(tool_calls, live_result)
    turn_index = live_result.get("localized_turn_index", live_result.get("turn_count"))

    if (
        primary_failure_type in {"length_or_step_budget_failure", "read_and_stall", "incomplete_workflow"}
        and len(read_calls) >= 2
        and not write_calls
        and not transfer_calls
    ):
        return {
            "mt_stepo_version": "mt_stepo_v0",
            "sdpo_eligible": True,
            "confidence": "high",
            "failure_bucket": "read_search_no_commit",
            "boundary_turn_index": turn_index,
            "verified_state": verified_state,
            "endpoint_class": "CONFIRM_OR_WRITE_OR_POLICY_ENDPOINT",
            "allowed_next_actions": [
                "ask one minimal confirmation if policy requires confirmation",
                "execute the required write tool when policy and confirmation allow it",
                "issue concise policy refusal when the request is not allowed",
                "transfer_to_human_agents when escalation is required",
            ],
            "forbidden_next_actions": [
                "repeat already-completed read/search tool",
                "summarize without endpoint",
                "ask for already-known reservation or user information",
            ],
            "repair_instruction": (
                "State was gathered but the trajectory failed to commit. "
                "Choose the next endpoint class instead of another read or search."
            ),
        }

    if primary_failure_type in {"missing_required_argument", "invalid_tool_arguments"}:
        if not verified_state and not tool_calls:
            return {
                "mt_stepo_version": "mt_stepo_v0",
                "sdpo_eligible": False,
                "confidence": "low",
                "failure_bucket": "not_endpoint_repairable",
                "boundary_turn_index": turn_index,
                "verified_state": verified_state,
                "endpoint_class": "UNKNOWN",
                "allowed_next_actions": [],
                "forbidden_next_actions": [],
                "repair_instruction": (
                    "Skip MT-STEPO for this sample. The failing call is not grounded enough for local repair."
                ),
            }
        return {
            "mt_stepo_version": "mt_stepo_v0",
            "sdpo_eligible": False,
            "confidence": "medium",
            "failure_bucket": "tool_execution_or_argument_error",
            "boundary_turn_index": turn_index,
            "verified_state": verified_state,
            "endpoint_class": "WRITE_ARGUMENT_REPAIR",
            "allowed_next_actions": [
                "retry the same endpoint tool with grounded required arguments",
                "ask one minimal question only if a required field is truly missing",
            ],
            "forbidden_next_actions": [
                "invent IDs or payment methods",
                "retry malformed arguments",
                "change the target reservation without evidence",
            ],
            "repair_instruction": (
                "The endpoint tool was plausible, but the call failed on arguments or schema. "
                "Skip MT-STEPO v0 for this sample until exact boundary localization is available."
            ),
        }

    if primary_failure_type == "unsupported_final_claim" and not write_calls:
        return {
            "mt_stepo_version": "mt_stepo_v0",
            "sdpo_eligible": False,
            "confidence": "medium",
            "failure_bucket": "policy_or_transfer_boundary",
            "boundary_turn_index": turn_index,
            "verified_state": verified_state,
            "endpoint_class": "FINAL_VERIFICATION_OR_POLICY_ENDPOINT",
            "allowed_next_actions": [
                "issue the required refusal or transfer endpoint when policy requires it",
                "perform the missing verified backend action before claiming success",
                "give a concise final answer only after the required endpoint is complete",
            ],
            "forbidden_next_actions": [
                "claim success without backend verification",
                "continue reading after the needed state is already known",
                "ask for already-known information",
            ],
            "repair_instruction": (
                "The trajectory made an unsupported final claim without a verified endpoint. "
                "Skip MT-STEPO v0 for this sample until exact boundary localization is available."
            ),
        }

    if primary_failure_type in {"unverified_write_action", "wrong_write_action"} or (
        primary_failure_type == "unsupported_final_claim" and bool(write_calls)
    ):
        return {
            "mt_stepo_version": "mt_stepo_v0",
            "sdpo_eligible": False,
            "confidence": "medium",
            "failure_bucket": "wrong_or_unsafe_write_endpoint",
            "boundary_turn_index": turn_index,
            "verified_state": verified_state,
            "endpoint_class": "VERIFY_POLICY_BEFORE_WRITE",
            "allowed_next_actions": [
                "verify policy preconditions before the write",
                "ask for confirmation before the write when required",
                "refuse or transfer if policy blocks the write",
            ],
            "forbidden_next_actions": [
                "write before required confirmation",
                "write when policy eligibility is false",
                "claim success without backend verification",
            ],
            "repair_instruction": (
                "A state-changing endpoint was attempted or claimed without sufficient verification. "
                "Skip MT-STEPO v0 for this sample until exact boundary localization is available."
            ),
        }

    if primary_failure_type in {"invalid_escalation", "policy_grounding_error", "asks_user_instead_of_finishing"}:
        return {
            "mt_stepo_version": "mt_stepo_v0",
            "sdpo_eligible": False,
            "confidence": "medium",
            "failure_bucket": "policy_or_transfer_boundary",
            "boundary_turn_index": turn_index,
            "verified_state": verified_state,
            "endpoint_class": "REFUSE_TRANSFER_OR_FINAL_POLICY_ACTION",
            "allowed_next_actions": [
                "issue the required refusal or policy-limited final answer",
                "transfer_to_human_agents when escalation is justified",
                "use one minimal follow-up only if policy truly requires it",
            ],
            "forbidden_next_actions": [
                "continue searching after the policy decision is clear",
                "ask for already-known information",
                "claim a policy outcome without the required endpoint action",
            ],
            "repair_instruction": (
                "The policy or escalation boundary was visible, but the trajectory did not close cleanly. "
                "Skip MT-STEPO v0 for this sample until exact boundary localization is available."
            ),
        }

    return {
        "mt_stepo_version": "mt_stepo_v0",
        "sdpo_eligible": False,
        "confidence": "low",
        "failure_bucket": "not_endpoint_repairable",
        "boundary_turn_index": turn_index,
        "verified_state": verified_state,
        "endpoint_class": "UNKNOWN",
        "allowed_next_actions": [],
        "forbidden_next_actions": [],
        "repair_instruction": (
            "Skip MT-STEPO for this sample. Use the standard diagnostic as fallback context."
        ),
    }


def _priority_item(priority: int, target: str, evidence: str, repair: str) -> dict[str, Any]:
    return {
        "priority": priority,
        "target": target,
        "evidence": compact_text(evidence, limit=220),
        "repair": compact_text(repair, limit=220),
    }


def _build_priority_focus(
    *,
    primary_failure_type: str,
    final_text: str,
    reward_info: dict[str, Any],
    tool_focus: dict[str, Any],
    workflow_status: dict[str, bool],
    tool_calls: list[dict[str, Any]],
    live_result: dict[str, Any],
) -> list[dict[str, Any]]:
    last_tool = tool_focus.get("last_tool")
    expected_tool = tool_focus.get("missing_or_expected_tool")
    info_note = compact_text((_json_text(reward_info.get("info")) or str(live_result.get("terminal_reason") or "")), limit=180)
    items: list[dict[str, Any]] = []

    if primary_failure_type == "length_or_step_budget_failure":
        items.append(
            _priority_item(
                1,
                "finish_within_budget",
                info_note or "The rollout terminated because the step or length budget was exhausted.",
                "Skip redundant reads, move to the decisive action earlier, and verify the required state before the final response.",
            )
        )
    elif primary_failure_type in {"missing_required_argument", "invalid_tool_arguments"}:
        fields = tool_focus.get("missing_fields") or tool_focus.get("invalid_fields") or []
        evidence = "The tool call used missing or invalid required arguments."
        if fields:
            evidence += f" Fields implicated: {fields}."
        items.append(
            _priority_item(
                1,
                "fix_tool_arguments",
                evidence,
                "Recover the required fields from verified tool state before issuing the next write action.",
            )
        )
        items.append(
            _priority_item(
                2,
                "recheck_schema_before_write",
                f"Failing tool: {tool_focus.get('failing_tool') or last_tool}.",
                "Validate field presence and types against the tool contract before retrying the action.",
            )
        )
    elif primary_failure_type == "unsupported_final_claim":
        items.append(
            _priority_item(
                1,
                "avoid_unverified_claim",
                "The assistant claimed success or completion without verified backend evidence.",
                "Do not claim completion until the required tool action has succeeded and the post-action state has been observed.",
            )
        )
        items.append(
            _priority_item(
                2,
                "complete_required_action",
                f"Workflow status before final response: {workflow_status}.",
                "Use the verified state to perform the missing backend action before the final answer.",
            )
        )
    elif primary_failure_type in {"unverified_write_action", "wrong_write_action"}:
        items.append(
            _priority_item(
                1,
                "verify_write_effect",
                f"Write-oriented tool activity occurred, but the final state was not verified. Last tool: {last_tool}.",
                "After the write action, inspect the resulting state and only then communicate completion.",
            )
        )
        items.append(
            _priority_item(
                2,
                "target_correct_mutation",
                f"Expected next tool or action: {expected_tool or 'verified backend-changing action'}.",
                "Ensure the mutation targets the correct reservation/order/entity and matches the user request exactly.",
            )
        )
    elif primary_failure_type in {"read_and_stall", "incomplete_workflow"}:
        items.append(
            _priority_item(
                1,
                "complete_required_workflow",
                f"The rollout stopped after read/search activity. Last tool: {last_tool}.",
                "Stop gathering redundant information and use the verified state to execute the next required action.",
            )
        )
        items.append(
            _priority_item(
                2,
                "promote_decision_from_state",
                f"Verified state available: {', '.join(_build_verified_state_summary(tool_calls, live_result)) or 'minimal state'}",
                "Convert the gathered facts into a concrete decision, write action, or valid transfer instead of another read step.",
            )
        )
    elif primary_failure_type == "asks_user_instead_of_finishing":
        items.append(
            _priority_item(
                1,
                "finish_without_redundant_confirmation",
                compact_text(final_text, limit=180),
                "Only ask the user for more information if policy strictly requires it; otherwise finish the task with the current verified state.",
            )
        )
    elif primary_failure_type in {"policy_grounding_error", "invalid_escalation", "valid_escalation_or_transfer"}:
        items.append(
            _priority_item(
                1,
                "ground_policy_or_transfer",
                compact_text(final_text, limit=180) or "The final response relied on policy or escalation language.",
                "Use transfer or refusal only when the visible task state justifies it, and communicate the policy reason clearly and minimally.",
            )
        )
    elif primary_failure_type == "state_tracking_error":
        items.append(
            _priority_item(
                1,
                "stabilize_target_entity",
                "The rollout appears to mix or lose track of which reservation/order/entity is active.",
                "Anchor every subsequent action to the currently verified entity before attempting a write or final claim.",
            )
        )
    else:
        items.append(
            _priority_item(
                1,
                "review_visible_failure_boundary",
                "The rollout failed but the dominant repair target was not confidently isolated.",
                "Review the visible trajectory, identify the first clearly incorrect step, and repair that step before changing later turns.",
            )
        )

    return items[:3]


def _build_do_not_focus_on(primary_failure_type: str) -> list[str]:
    generic = ["irrelevant earlier dialogue", "stale search results", "unverified assistant claims"]
    if primary_failure_type in {"missing_required_argument", "invalid_tool_arguments"}:
        return generic + ["rephrasing the final answer before the tool call is fixed"]
    if primary_failure_type in {"read_and_stall", "incomplete_workflow"}:
        return generic + ["additional redundant read/search actions"]
    if primary_failure_type in {"unsupported_final_claim", "unverified_write_action", "wrong_write_action"}:
        return generic + ["final-answer polish before backend verification"]
    return generic


def _diagnostic_confidence(primary_failure_type: str) -> str:
    if primary_failure_type in {
        "length_or_step_budget_failure",
        "missing_required_argument",
        "invalid_tool_arguments",
        "read_and_stall",
        "incomplete_workflow",
        "unsupported_final_claim",
        "unverified_write_action",
    }:
        return "high"
    if primary_failure_type in {"wrong_write_action", "policy_grounding_error", "invalid_escalation", "valid_escalation_or_transfer"}:
        return "medium"
    if primary_failure_type == "state_tracking_error":
        return "medium"
    return "low"


def _build_trace_evidence(
    *,
    live_result: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    reward_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "domain": live_result.get("domain"),
        "task_id": live_result.get("task_id"),
        "terminal_reason": live_result.get("terminal_reason"),
        "turn_count": live_result.get("turn_count", 0),
        "executed_tools": [call.get("name") for call in tool_calls if call.get("name")],
        "tool_call_count": len(tool_calls),
        "last_parse_error": compact_text(live_result.get("last_parse_error", ""), limit=160),
    }


def diagnostic_scalar_metrics(diag: TauDiagnostic) -> dict[str, float]:
    primary = diag.primary_failure_type
    high = 1.0 if diag.diagnostic_confidence == "high" else 0.0
    medium = 1.0 if diag.diagnostic_confidence == "medium" else 0.0
    low = 1.0 if diag.diagnostic_confidence == "low" else 0.0
    fallback = 1.0 if diag.diagnostic_version.endswith("_fallback") else 0.0
    success = 1.0 - fallback

    diagnostic_json_valid = 1.0
    try:
        json.dumps(asdict(diag), ensure_ascii=False, sort_keys=True)
    except Exception:
        diagnostic_json_valid = 0.0

    token_like_length = len(re.findall(r"\w+|[^\w\s]", json.dumps(asdict(diag), ensure_ascii=False, sort_keys=True)))
    return {
        "s_sdpo/diagnostic_success_fraction": success,
        "s_sdpo/diagnostic_failure_fraction": 1.0 - success,
        "s_sdpo/diagnostic_fallback_fraction": fallback,
        "s_sdpo/diagnostic_json_valid_fraction": diagnostic_json_valid,
        "s_sdpo/diagnostic_avg_focus_items": float(len(diag.priority_focus)),
        "s_sdpo/diagnostic_avg_token_length": float(token_like_length),
        "s_sdpo/diagnostic_confidence_high_fraction": high,
        "s_sdpo/diagnostic_confidence_medium_fraction": medium,
        "s_sdpo/diagnostic_confidence_low_fraction": low,
        "s_sdpo/failure_type_read_and_stall_fraction": 1.0 if primary == "read_and_stall" else 0.0,
        "s_sdpo/failure_type_incomplete_workflow_fraction": 1.0 if primary == "incomplete_workflow" else 0.0,
        "s_sdpo/failure_type_unsupported_final_claim_fraction": 1.0 if primary == "unsupported_final_claim" else 0.0,
        "s_sdpo/failure_type_unverified_write_action_fraction": 1.0 if primary == "unverified_write_action" else 0.0,
        "s_sdpo/failure_type_invalid_tool_arguments_fraction": 1.0 if primary == "invalid_tool_arguments" else 0.0,
        "s_sdpo/failure_type_state_tracking_error_fraction": 1.0 if primary == "state_tracking_error" else 0.0,
        "s_sdpo/failure_type_unknown_fraction": 1.0 if primary == "unknown" else 0.0,
    }


def assert_no_oracle_leakage(diag: TauDiagnostic) -> None:
    payload = json.dumps(asdict(diag), ensure_ascii=False).lower()
    for key in FORBIDDEN_DIAGNOSTIC_KEYS:
        assert key.lower() not in payload, f"Forbidden leakage key found: {key}"


def _fallback_diagnostic(reward: float, terminal_reason: str) -> TauDiagnostic:
    return TauDiagnostic(
        diagnostic_version="s_sdpo_v1_fallback",
        success=reward >= 1.0,
        reward=reward,
        primary_failure_type="unknown",
        failure_stage="unknown",
        localized_turn_index=None,
        verified_state_summary=[],
        workflow_status={
            "information_gathered": False,
            "required_decision_made": False,
            "required_write_action_completed": False,
            "final_state_verified": reward >= 1.0,
        },
        tool_focus={
            "last_tool": None,
            "failing_tool": None,
            "missing_or_expected_tool": None,
            "missing_fields": [],
            "invalid_fields": [],
        },
        policy_or_state_focus={
            "policy_issue": None,
            "state_tracking_issue": None,
            "entity_or_record_at_risk": None,
        },
        priority_focus=[
            _priority_item(
                1,
                "review_full_trajectory",
                f"Structured diagnostic extraction failed. terminal_reason={terminal_reason or 'unknown'}",
                "Review the full trajectory, complete the required workflow, avoid unsupported final claims, and verify tool results before the final response.",
            )
        ],
        do_not_focus_on=["diagnostic formatting itself"],
        teacher_instruction="Use the full trajectory as fallback context. Do not copy or explain this JSON.",
        diagnostic_confidence="low",
        failed_components=["UNKNOWN"],
        endpoint_boundary=None,
        trace_evidence={
            "terminal_reason": terminal_reason,
            "fallback": True,
        },
    )


def _compile_diagnostic(live_result: dict[str, Any]) -> TauDiagnostic:
    reward = float(live_result.get("final_reward", live_result.get("reward", 0.0)) or 0.0)
    success = reward >= 1.0
    reward_info = parse_json_maybe(live_result.get("reward_info_json") or live_result.get("reward_info"))
    simulation_run = parse_json_maybe(live_result.get("simulation_run_json") or live_result.get("simulation_run"))
    tool_calls = _extract_live_tool_calls(live_result, simulation_run)
    failed_components = _infer_failed_components(reward_info, live_result)
    final_text = _strip_think(live_result.get("latest_assistant_message", ""))
    primary_failure_type = _infer_primary_failure_type(
        reward=reward,
        live_result=live_result,
        reward_info=reward_info,
        simulation_run=simulation_run,
        tool_calls=tool_calls,
    )
    failure_stage = _infer_failure_stage(primary_failure_type, tool_calls)
    workflow_status = _build_workflow_status(
        reward=reward,
        reward_info=reward_info,
        tool_calls=tool_calls,
        primary_failure_type=primary_failure_type,
    )
    tool_focus = _build_tool_focus(
        primary_failure_type=primary_failure_type,
        reward_info=reward_info,
        live_result=live_result,
        tool_calls=tool_calls,
    )
    policy_or_state_focus = _build_policy_or_state_focus(
        primary_failure_type=primary_failure_type,
        final_text=final_text,
        tool_calls=tool_calls,
    )
    endpoint_boundary = _build_endpoint_boundary(
        primary_failure_type=primary_failure_type,
        live_result=live_result,
        tool_calls=tool_calls,
        workflow_status=workflow_status,
        final_text=final_text,
    )
    if endpoint_boundary.get("sdpo_eligible"):
        tool_focus = {
            **tool_focus,
            "boundary_hint": endpoint_boundary.get("failure_bucket"),
            "boundary_confidence": endpoint_boundary.get("confidence"),
        }
    priority_focus = _build_priority_focus(
        primary_failure_type=primary_failure_type,
        final_text=final_text,
        reward_info=reward_info,
        tool_focus=tool_focus,
        workflow_status=workflow_status,
        tool_calls=tool_calls,
        live_result=live_result,
    )
    localized_turn_index = live_result.get("localized_turn_index")
    if localized_turn_index is None:
        localized_turn_index = tool_calls[-1].get("turn_idx") if tool_calls else live_result.get("turn_count")

    diag = TauDiagnostic(
        diagnostic_version="s_sdpo_v1",
        success=success,
        reward=reward,
        primary_failure_type=primary_failure_type if not success else ("valid_escalation_or_transfer" if primary_failure_type == "valid_escalation_or_transfer" else "unknown"),
        failure_stage="completed" if success else failure_stage,
        localized_turn_index=None if success else localized_turn_index,
        verified_state_summary=_build_verified_state_summary(tool_calls, live_result),
        workflow_status=workflow_status,
        tool_focus=tool_focus,
        policy_or_state_focus=policy_or_state_focus,
        priority_focus=priority_focus if not success else [
            _priority_item(
                1,
                "preserve_verified_behavior",
                "The rollout achieved a positive evaluator reward.",
                "Keep the verified workflow structure and backend-confirmed action ordering.",
            )
        ],
        do_not_focus_on=_build_do_not_focus_on(primary_failure_type),
        teacher_instruction="Use the diagnostic as a prioritized repair guide. Do not copy or explain this JSON.",
        diagnostic_confidence=_diagnostic_confidence(primary_failure_type if not success else "valid_escalation_or_transfer"),
        failed_components=failed_components,
        endpoint_boundary=endpoint_boundary,
        trace_evidence=_build_trace_evidence(live_result=live_result, tool_calls=tool_calls, reward_info=reward_info),
    )
    assert_no_oracle_leakage(diag)
    return diag


def build_safe_diagnostic(live_result: dict[str, Any]) -> TauDiagnostic:
    """Build a deterministic focus-map diagnostic with safe fallback."""

    reward = float(live_result.get("final_reward", live_result.get("reward", 0.0)) or 0.0)
    terminal_reason = str(live_result.get("terminal_reason", "") or "")
    try:
        return _compile_diagnostic(live_result)
    except Exception:
        return _fallback_diagnostic(reward, terminal_reason)
