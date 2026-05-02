from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from verl.utils.tau3_tool_schema_export import export_openai_tool_schemas, validate_openai_schema

VALID_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})
TRANSCRIPT_REPLAY_RE = re.compile(r"(?m)^\s*(user|tool|assistant)\s*:")
XML_TOOL_NAME_RE = re.compile(r"<function\s*=\s*([a-zA-Z_]\w*)\s*>")
JSON_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([a-zA-Z_]\w*)"')
FUNCTIONAL_TOOL_NAME_RE = re.compile(r"(?m)^\s*([a-zA-Z_]\w*)\s*\(")


@dataclass(slots=True)
class ValidationResult:
    accept: bool
    reason: str
    message_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _reject(reason: str, *, message_count: int = 0, tool_names: Optional[list[str]] = None, **details: Any) -> ValidationResult:
    return ValidationResult(
        accept=False,
        reason=reason,
        message_count=message_count,
        tool_names=list(tool_names or []),
        details=details,
    )


def _accept(*, message_count: int, tool_names: list[str], **details: Any) -> ValidationResult:
    return ValidationResult(
        accept=True,
        reason="ok",
        message_count=message_count,
        tool_names=list(tool_names),
        details=details,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tool_config_path(domain: str) -> Path:
    return _repo_root() / "examples" / "sglang_multiturn" / "config" / "tool_config" / f"tau3_live_{domain}_tool_config.yaml"


def _load_openai_schemas_from_yaml(domain: str) -> list[dict[str, Any]]:
    import yaml

    path = _tool_config_path(domain)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid tau3 tool config payload at {path}")

    tools = payload.get("tools")
    if not isinstance(tools, list):
        raise ValueError(f"Missing tools list in tau3 tool config {path}")

    schemas: list[dict[str, Any]] = []
    for tool_entry in tools:
        if not isinstance(tool_entry, Mapping):
            raise ValueError(f"Invalid tool entry in tau3 tool config {path}")
        schema = tool_entry.get("tool_schema")
        if not isinstance(schema, Mapping):
            raise ValueError(f"Missing tool_schema in tau3 tool config {path}")
        schema_dict = dict(schema)
        validate_openai_schema(schema_dict)
        params = schema_dict["function"]["parameters"]
        if "required" not in params:
            params["required"] = []
        schemas.append(schema_dict)
    return schemas


def _ensure_done_tool(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = {str(schema.get("function", {}).get("name", "")) for schema in schemas}
    if "done" in names:
        return schemas
    return [
        *schemas,
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Call this function when the task is complete.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                    "required": [],
                },
            },
        },
    ]


@lru_cache(maxsize=None)
def get_openai_tool_schemas(domain: str) -> list[dict[str, Any]]:
    try:
        return _ensure_done_tool(export_openai_tool_schemas(domain=domain, include_done=True))
    except Exception:
        return _ensure_done_tool(_load_openai_schemas_from_yaml(domain))


@lru_cache(maxsize=None)
def _allowed_tool_names(domain: str) -> frozenset[str]:
    schemas = get_openai_tool_schemas(domain)
    names: set[str] = set()
    for schema in schemas:
        validate_openai_schema(schema)
        names.add(str(schema["function"]["name"]))
    return frozenset(names)


def _extract_domain(candidate: Mapping[str, Any], explicit_domain: Optional[str]) -> str:
    if explicit_domain:
        return explicit_domain

    domain = candidate.get("domain")
    if isinstance(domain, str) and domain:
        return domain

    tau3_live_result = candidate.get("tau3_live_result")
    if isinstance(tau3_live_result, Mapping):
        result_domain = tau3_live_result.get("domain")
        if isinstance(result_domain, str) and result_domain:
            return result_domain

    return "airline"


def _strip_optional_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return parsed
    return None


def _coerce_message_list(value: Any) -> Optional[list[dict[str, Any]]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    messages: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        messages.append(dict(item))
    return messages


def _extract_messages(candidate: Mapping[str, Any]) -> Optional[list[dict[str, Any]]]:
    direct_messages = _coerce_message_list(candidate.get("messages"))
    if direct_messages is not None:
        return direct_messages

    simulation_sources = [
        candidate.get("simulation_run"),
        candidate.get("simulation_run_json"),
    ]

    tau3_live_result = candidate.get("tau3_live_result")
    if isinstance(tau3_live_result, Mapping):
        simulation_sources.extend(
            [tau3_live_result.get("simulation_run"), tau3_live_result.get("simulation_run_json")]
        )

    for source in simulation_sources:
        payload = _coerce_mapping(source)
        if payload is None:
            continue
        messages = _coerce_message_list(payload.get("messages"))
        if messages is not None:
            return messages

    return None


def _build_thinking_text_map(candidate: Mapping[str, Any]) -> list[str]:
    """Extract per-agent-turn thinking_text from the trajectory turns array.

    Returns a list where index *i* is the thinking_text for the *i*-th agent
    turn (0-indexed).  The simulation_run.messages array has an extra leading
    assistant greeting that is not an agent turn, so callers must skip that
    first assistant message when zipping with this list.
    """
    turns = candidate.get("turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes, bytearray)):
        return []
    return [
        _strip_optional_string(turn.get("thinking_text")) if isinstance(turn, Mapping) else ""
        for turn in turns
    ]


def _has_leading_assistant_greeting(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        return role == "assistant"
    return False


def extract_messages_for_sft(
    candidate: Mapping[str, Any],
    *,
    include_system_prompt: bool = True,
    include_thinking_traces: bool = False,
) -> list[dict[str, Any]]:
    messages = _extract_messages(candidate)
    if messages is None:
        raise ValueError("Candidate does not contain structured messages.")

    # Build thinking-text lookup when requested. Simulation traces include a
    # leading assistant greeting, but direct message lists may start with user.
    thinking_texts: list[str] = []
    if include_thinking_traces:
        thinking_texts = _build_thinking_text_map(candidate)

    normalized_messages: list[dict[str, Any]] = []
    system_prompt = _strip_optional_string(candidate.get("system_prompt")) if include_system_prompt else ""

    if system_prompt:
        first_message = messages[0] if messages else None
        if not (isinstance(first_message, Mapping) and first_message.get("role") == "system"):
            normalized_messages.append({"role": "system", "content": system_prompt})

    skip_first_assistant_for_thinking = _has_leading_assistant_greeting(messages)
    assistant_message_index = -1
    skipped_leading_greeting = False

    for message in messages:
        role = message.get("role")
        if role not in VALID_MESSAGE_ROLES:
            raise ValueError(f"Unknown message role: {role}")

        # Drop the leading assistant greeting (e.g. "Hi! How can I help you today?")
        # because Qwen3.5 chat template requires user-first after system.
        if (
            role == "assistant"
            and skip_first_assistant_for_thinking
            and not skipped_leading_greeting
            and not message.get("tool_calls")
        ):
            skipped_leading_greeting = True
            continue

        normalized: dict[str, Any] = {"role": role}
        tool_calls = message.get("tool_calls")

        if role == "assistant":
            assistant_message_index += 1

        # Resolve thinking text for this assistant turn. If the raw trace had a
        # greeting, it was dropped above, so kept assistant turns now align
        # directly with the trajectory turns array.
        thinking_prefix = ""
        thinking_index = assistant_message_index
        if (
            include_thinking_traces
            and role == "assistant"
            and thinking_index >= 0
            and thinking_index < len(thinking_texts)
        ):
            tt = thinking_texts[thinking_index]
            if tt:
                thinking_prefix = f"<think>\n{tt}\n</think>\n"

        if role == "assistant" and tool_calls is not None:
            content = message.get("content")
            if thinking_prefix:
                # Qwen tool-call messages have content=None; store thinking
                # there so the chat template renders it before the tool call.
                content = thinking_prefix.rstrip("\n") if content is None else f"{thinking_prefix}{content}"
            normalized["content"] = content
            normalized_tool_calls: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    raise ValueError("Assistant tool_calls must contain mappings.")
                tool_name = _extract_structured_tool_name(tool_call)
                if not tool_name:
                    raise ValueError("Assistant tool_call missing tool name.")
                arguments = _extract_structured_tool_arguments(tool_call)
                if isinstance(arguments, Mapping):
                    arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                elif arguments is None:
                    arguments = "{}"
                elif not isinstance(arguments, str):
                    raise ValueError(f"Unsupported tool_call arguments type: {type(arguments)!r}")
                normalized_tool_calls.append(
                    {
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments},
                    }
                )
            normalized["tool_calls"] = normalized_tool_calls
        else:
            content = message.get("content")
            if thinking_prefix and role == "assistant" and isinstance(content, str):
                content = f"{thinking_prefix}{content}"
            normalized["content"] = content

        normalized_messages.append(normalized)

    return normalized_messages


def _numeric_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _runtime_verified_success(candidate: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    tau3_live_result = candidate.get("tau3_live_result")
    if not isinstance(tau3_live_result, Mapping):
        return False, {"missing_tau3_live_result": True}

    runtime = _strip_optional_string(tau3_live_result.get("runtime")).lower()
    terminal_reason = _strip_optional_string(tau3_live_result.get("terminal_reason")).lower()
    status = _strip_optional_string(tau3_live_result.get("status")).lower()
    is_terminal = status == "terminated" or terminal_reason not in {"", "running"}
    runtime_ok = runtime in {"", "official_gym"}

    success_flag = candidate.get("success")
    rewards = [
        _numeric_or_none(candidate.get("final_reward")),
        _numeric_or_none(candidate.get("score")),
        _numeric_or_none(candidate.get("reward")),
        _numeric_or_none(tau3_live_result.get("final_reward")),
        _numeric_or_none(tau3_live_result.get("score")),
        _numeric_or_none(tau3_live_result.get("reward")),
    ]
    rewards = [reward for reward in rewards if reward is not None]
    max_reward = max(rewards) if rewards else None
    reward_success = max_reward is not None and max_reward >= 1.0

    accepted = bool(runtime_ok and is_terminal and (success_flag is True or reward_success))
    details = {
        "runtime": runtime,
        "status": status,
        "terminal_reason": terminal_reason,
        "success_flag": success_flag,
        "max_reward": max_reward,
    }
    return accepted, details


def _extract_structured_tool_name(tool_call: Mapping[str, Any]) -> Optional[str]:
    function_entry = tool_call.get("function")
    if isinstance(function_entry, Mapping):
        name = function_entry.get("name")
        if isinstance(name, str) and name:
            return name
    name = tool_call.get("name")
    if isinstance(name, str) and name:
        return name
    return None


def _extract_structured_tool_arguments(tool_call: Mapping[str, Any]) -> Any:
    function_entry = tool_call.get("function")
    if isinstance(function_entry, Mapping) and "arguments" in function_entry:
        return function_entry.get("arguments")
    return tool_call.get("arguments")


def _assistant_text_reject_reason(text: str, allowed_tool_names: frozenset[str]) -> Optional[str]:
    lowered = (text or "").lower()

    if TRANSCRIPT_REPLAY_RE.search(text):
        return "transcript_replay"
    if "```" in text:
        return "markdown_tool_or_fence"
    if "get_reservations" in lowered:
        return "known_fake_tool_get_reservations"

    for name in XML_TOOL_NAME_RE.findall(text):
        if name not in allowed_tool_names:
            return f"unknown_xml_tool:{name}"
    for name in JSON_TOOL_NAME_RE.findall(text):
        if name not in allowed_tool_names:
            return f"unknown_json_tool:{name}"
    for name in FUNCTIONAL_TOOL_NAME_RE.findall(text):
        if name not in allowed_tool_names:
            return f"unknown_functional_tool:{name}"

    return None


def _validate_assistant_tool_calls(
    tool_calls: Any,
    *,
    allowed_tool_names: frozenset[str],
    tool_names: list[str],
    message_index: int,
) -> Optional[ValidationResult]:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes, bytearray)) or len(tool_calls) == 0:
        return _reject("invalid_tool_calls", message_index=message_index)

    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            return _reject("invalid_tool_calls", message_index=message_index)
        name = _extract_structured_tool_name(tool_call)
        if not name:
            return _reject("missing_tool_name", message_index=message_index)
        if name == "get_reservations":
            return _reject("known_fake_tool_get_reservations", message_index=message_index, tool_name=name)
        if name not in allowed_tool_names:
            return _reject(f"unknown_structured_tool:{name}", message_index=message_index, tool_name=name)

        arguments = _extract_structured_tool_arguments(tool_call)
        if arguments is not None and not isinstance(arguments, (Mapping, str)):
            return _reject("invalid_tool_arguments", message_index=message_index, tool_name=name)

        if name not in tool_names:
            tool_names.append(name)

    return None


def validate_tau3_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    domain: str = "airline",
    allowed_tool_names: Optional[Sequence[str]] = None,
) -> ValidationResult:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return _reject("messages_not_list")

    normalized_messages = _coerce_message_list(messages)
    if normalized_messages is None or len(normalized_messages) == 0:
        return _reject("messages_not_list")

    allowed = frozenset(allowed_tool_names) if allowed_tool_names is not None else _allowed_tool_names(domain)
    tool_names: list[str] = []
    pending_tool_responses = 0

    for message_index, message in enumerate(normalized_messages):
        role = message.get("role")
        if role not in VALID_MESSAGE_ROLES:
            return _reject("unknown_message_role", message_index=message_index, role=role)

        content = message.get("content")
        tool_calls = message.get("tool_calls")

        if role == "tool":
            if tool_calls is not None:
                return _reject("tool_message_contains_tool_calls", message_index=message_index)
            if not isinstance(content, str):
                return _reject("tool_message_non_string_content", message_index=message_index)
            if pending_tool_responses <= 0:
                return _reject("tool_without_preceding_assistant_tool_call", message_index=message_index)
            pending_tool_responses -= 1
            continue

        if pending_tool_responses > 0:
            return _reject("assistant_tool_call_without_tool_response", message_index=message_index)

        if role in {"system", "user"}:
            if tool_calls is not None:
                return _reject("non_assistant_tool_calls", message_index=message_index, role=role)
            if not isinstance(content, str):
                return _reject("non_string_content", message_index=message_index, role=role)
            continue

        if tool_calls is not None:
            text = _strip_optional_string(content)
            if text:
                return _reject("assistant_content_with_tool_calls", message_index=message_index)
            tool_call_result = _validate_assistant_tool_calls(
                tool_calls,
                allowed_tool_names=allowed,
                tool_names=tool_names,
                message_index=message_index,
            )
            if tool_call_result is not None:
                return tool_call_result
            pending_tool_responses = len(tool_calls)
            continue

        if not isinstance(content, str) or not content.strip():
            return _reject("empty_assistant_content", message_index=message_index)

        reject_reason = _assistant_text_reject_reason(content, allowed)
        if reject_reason is not None:
            return _reject(reject_reason, message_index=message_index)

        for name in XML_TOOL_NAME_RE.findall(content):
            if name not in tool_names:
                tool_names.append(name)
        for name in JSON_TOOL_NAME_RE.findall(content):
            if name not in tool_names:
                tool_names.append(name)

    if pending_tool_responses > 0:
        return _reject("dangling_assistant_tool_call", message_index=len(normalized_messages) - 1)

    return _accept(message_count=len(normalized_messages), tool_names=tool_names, domain=domain)


def validate_sft_candidate(
    candidate: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    domain: Optional[str] = None,
    allowed_tool_names: Optional[Sequence[str]] = None,
    require_runtime_verified_success: bool = True,
) -> ValidationResult:
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        if require_runtime_verified_success:
            return _reject("missing_runtime_verified_success")
        return validate_tau3_messages(candidate, domain=domain or "airline", allowed_tool_names=allowed_tool_names)

    if not isinstance(candidate, Mapping):
        return _reject("candidate_not_mapping")

    selected_domain = _extract_domain(candidate, domain)

    parse_error = ""
    parsed_entry = candidate.get("parsed")
    if isinstance(parsed_entry, Mapping):
        parse_error = _strip_optional_string(parsed_entry.get("parse_error"))
    tau3_live_result = candidate.get("tau3_live_result")
    if not parse_error and isinstance(tau3_live_result, Mapping):
        parse_error = _strip_optional_string(tau3_live_result.get("last_parse_error"))
    if parse_error:
        return _reject("parse_error", parse_error=parse_error)

    if require_runtime_verified_success:
        success_ok, success_details = _runtime_verified_success(candidate)
        if not success_ok:
            return _reject("missing_runtime_verified_success", **success_details)

    messages = _extract_messages(candidate)
    if messages is None:
        return _reject("missing_messages")

    result = validate_tau3_messages(messages, domain=selected_domain, allowed_tool_names=allowed_tool_names)
    if not result.accept:
        return result

    result.details.update(
        {
            "candidate_domain": selected_domain,
            "runtime_verified_success": require_runtime_verified_success,
        }
    )
    return result


__all__ = [
    "ValidationResult",
    "extract_messages_for_sft",
    "get_openai_tool_schemas",
    "validate_sft_candidate",
    "validate_tau3_messages",
]
