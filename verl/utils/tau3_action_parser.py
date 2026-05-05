from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedTauAction:
    """Normalized action passed to tau AgentGymEnv.step(...).

    Important: this parser is allowed to normalize wrappers/format only. It must
    not add missing arguments, fix wrong values, or remap tool names, because that
    would give the policy hidden help during training.
    """

    action_for_env: str
    action_type: str  # tool_json | tool_functional | tool_qwen_xml | plain_text | invalid
    parse_error: str | None = None
    raw_model_output: str = ""
    stripped_thinking: bool = False


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think\b[^>]*>", flags=re.IGNORECASE)
_CLOSE_THINK_RE = re.compile(r"</think>", flags=re.IGNORECASE)


def strip_thinking_blocks(text: str) -> str:
    """Remove visible Qwen thinking content before parsing an executable action.

    The raw model output is still preserved on ParsedTauAction for audit. This
    parser-facing view is intentionally conservative: if a dangling opening
    <think> appears, drop everything from that tag onward so JSON/tool-call text
    inside an unfinished reasoning block cannot be executed as an action.
    """

    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    open_match = _OPEN_THINK_RE.search(cleaned)
    if open_match is not None:
        cleaned = cleaned[: open_match.start()]
    cleaned = _CLOSE_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def has_thinking_markup(text: str) -> bool:
    return bool(_OPEN_THINK_RE.search(text or "") or _CLOSE_THINK_RE.search(text or ""))


def extract_first_balanced_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON tool-call object from a model output.

    Expected object shape:
        {"name": "<tool_name>", "arguments": {...}}

    The implementation intentionally does not repair semantics.
    """

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text or ""):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return obj
    return None


def _coerce_json_like_argument_value(value: Any) -> Any:
    """Coerce JSON-shaped strings back to structured values when safe.

    We intentionally keep plain scalar strings as strings, even if they look like
    JSON scalars such as `19031` or `true`, because the tau tool schemas in this
    repo overwhelmingly expect string-valued scalar arguments. The only eager
    coercion we do is for container-shaped values that were flattened into
    JSON-serialized strings to satisfy verl's narrow tool schema.
    """

    if isinstance(value, str):
        stripped = value.strip()
        if (
            (stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))
        ):
            try:
                decoded = json.loads(stripped)
            except Exception:
                return value
            return _coerce_json_like_argument_value(decoded)
        return value
    if isinstance(value, list):
        return [_coerce_json_like_argument_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _coerce_json_like_argument_value(v) for k, v in value.items()}
    return value


def looks_like_functional_tool_call(text: str) -> bool:
    """Return True for strings like search_flights(origin='NYC')."""

    return bool(re.match(r"^\s*[a-zA-Z_]\w*\s*\(.*\)\s*$", (text or "").strip(), re.DOTALL))


def _parse_parameter_value(raw_value: str) -> Any:
    """Parse one Qwen XML ``<parameter=...>`` value.

    Qwen XML uses JSON syntax inside each parameter block: strings are quoted,
    arrays/objects are JSON, and scalar numbers/bools/null are unquoted. Keep
    malformed/unquoted non-JSON text as text, but preserve valid JSON scalar
    types so integer tool args such as baggage counts do not become strings.
    """

    value = (raw_value or "").strip()
    if not value:
        return ""
    try:
        decoded = json.loads(value)
    except Exception:
        return value
    if isinstance(decoded, (list, dict)):
        return _coerce_json_like_argument_value(decoded)
    if isinstance(decoded, str):
        return decoded
    return decoded


def extract_qwen_tool_call(text: str) -> dict[str, Any] | None:
    """Extract Qwen native <tool_call>...</tool_call> format into tau action JSON.

    Expected patterns observed in Qwen3.5 tool mode:
        <tool_call>
        <function=get_user_details>
        </function>
        </tool_call>

        <tool_call>
        <function=update_reservation_baggages>
        <parameter=baggage_updates>
        {"reservation_id": "..."}
        </parameter>
        </function>
        </tool_call>
    """

    raw = (text or "").strip()
    if "<tool_call>" not in raw or "</tool_call>" not in raw:
        return None

    function_match = re.search(
        r"<function\s*=\s*([a-zA-Z_]\w*)\s*>(.*?)</function>",
        raw,
        flags=re.DOTALL,
    )
    if function_match is None:
        return None

    tool_name = function_match.group(1).strip()
    function_body = function_match.group(2) or ""
    arguments: dict[str, Any] = {}

    for param_name, param_value in re.findall(
        r"<parameter\s*=\s*([a-zA-Z_]\w*)\s*>(.*?)</parameter>",
        function_body,
        flags=re.DOTALL,
    ):
        arguments[param_name.strip()] = _parse_parameter_value(param_value)

    return {"name": tool_name, "arguments": _coerce_json_like_argument_value(arguments)}


def extract_tool_name_and_arguments(
    text: str,
    parsed_action_type: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Extract a normalized tool name + arguments pair from raw model output."""

    text = strip_thinking_blocks(text)

    if parsed_action_type in {None, "tool_json"}:
        obj = extract_first_balanced_json_object(text)
        if isinstance(obj, dict):
            name = str(obj.get("name") or "").strip() or None
            arguments = _coerce_json_like_argument_value(obj.get("arguments") or {})
            return name, arguments

    if parsed_action_type in {None, "tool_qwen_xml"}:
        obj = extract_qwen_tool_call(text)
        if isinstance(obj, dict):
            return str(obj.get("name") or "").strip() or None, obj.get("arguments") or {}

    if parsed_action_type in {None, "tool_functional"} and looks_like_functional_tool_call(text):
        match = re.match(r"^\s*([a-zA-Z_]\w*)\s*\(", text or "")
        if match:
            return match.group(1), {}

    return None, None


def parse_model_output_to_tau_action(model_output: str) -> ParsedTauAction:
    """Convert a raw model response into an AgentGymEnv action string."""

    stripped_thinking = has_thinking_markup(model_output)
    raw = strip_thinking_blocks(model_output)

    obj = extract_first_balanced_json_object(raw)
    if obj is not None:
        obj["arguments"] = _coerce_json_like_argument_value(obj.get("arguments") or {})
        return ParsedTauAction(
            action_for_env=json.dumps(obj, ensure_ascii=False),
            action_type="tool_json",
            raw_model_output=model_output,
            stripped_thinking=stripped_thinking,
        )

    qwen_tool_obj = extract_qwen_tool_call(raw)
    if qwen_tool_obj is not None:
        return ParsedTauAction(
            action_for_env=json.dumps(qwen_tool_obj, ensure_ascii=False),
            action_type="tool_qwen_xml",
            raw_model_output=model_output,
            stripped_thinking=stripped_thinking,
        )

    if looks_like_functional_tool_call(raw):
        return ParsedTauAction(
            action_for_env=raw,
            action_type="tool_functional",
            raw_model_output=model_output,
            stripped_thinking=stripped_thinking,
        )

    if raw:
        return ParsedTauAction(
            action_for_env=raw,
            action_type="plain_text",
            raw_model_output=model_output,
            stripped_thinking=stripped_thinking,
        )

    return ParsedTauAction(
        action_for_env="[INVALID_EMPTY_ASSISTANT_OUTPUT]",
        action_type="invalid",
        parse_error="empty_model_output",
        raw_model_output=model_output,
        stripped_thinking=stripped_thinking,
    )


def tool_call_to_action(tool_name: str, arguments: dict[str, Any] | None) -> str:
    """Build a tau-compatible JSON action string from a parsed Verl tool call."""

    return json.dumps(
        {"name": tool_name, "arguments": _coerce_json_like_argument_value(arguments or {})},
        ensure_ascii=False,
    )
