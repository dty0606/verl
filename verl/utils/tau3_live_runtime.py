from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from types import MethodType, SimpleNamespace
from typing import Any


def get_tau_runtime() -> SimpleNamespace:
    """Best-effort loader for the public tau runtime.

    The live tau3 pilot is expected to run on a machine where the current
    `tau2-bench` package (used as tau3 runtime) is installed. We keep the import
    here so the rest of the SDPO repo remains importable on non-benchmark hosts.
    """
    try:
        from tau2.agent.llm_agent import LLMAgent
        from tau2.data_model.simulation import CommunicationMode
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.user.user_simulator import UserSimulator
    except ImportError as exc:  # pragma: no cover - live-only dependency
        raise RuntimeError(
            "Could not import tau2-bench runtime. Install the public tau2-bench "
            "package on the benchmark machine before running tau3 live SDPO."
        ) from exc

    return SimpleNamespace(
        CommunicationMode=CommunicationMode,
        EvaluationType=EvaluationType,
        evaluate_simulation=evaluate_simulation,
        Orchestrator=Orchestrator,
        UserSimulator=UserSimulator,
        LLMAgent=LLMAgent,
    )


def get_domain_env_and_tasks(domain: str):
    if domain == "airline":
        from tau2.domains.airline.environment import get_environment, get_tasks
    elif domain == "retail":
        from tau2.domains.retail.environment import get_environment, get_tasks
    elif domain == "telecom":
        from tau2.domains.telecom.environment import get_environment, get_tasks
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported tau3 domain: {domain}")
    return get_environment, get_tasks


def get_domain_tasks_for_split(domain: str, task_split_name: str | None = "base") -> list[Any]:
    _, get_tasks = get_domain_env_and_tasks(domain)
    requested_split = str(task_split_name or "base")
    try:
        return list(get_tasks(task_split_name))
    except TypeError:
        try:
            return list(get_tasks(task_split_name=task_split_name))
        except TypeError:
            if requested_split == "base":
                return list(get_tasks())
            raise
    except ValueError:
        if requested_split == "base":
            try:
                return list(get_tasks())
            except TypeError:
                pass
        raise


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return to_jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return to_jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def compact_text(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def scenario_to_instruction_text(user_scenario: Any) -> str:
    payload = to_jsonable(user_scenario)
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        persona = payload.get("persona")
        instructions = payload.get("instructions")
        if isinstance(instructions, dict):
            lines: list[str] = []
            persona_text = str(persona).strip() if persona else ""
            if persona_text:
                lines.append(f"Persona:\n{persona_text}")
            for label, key in (
                ("Known info", "known_info"),
                ("Reason for call", "reason_for_call"),
                ("Task instructions", "task_instructions"),
                ("Unknown info", "unknown_info"),
            ):
                value = str(instructions.get(key, "") or "").strip()
                if value:
                    lines.append(f"{label}:\n{value}")
            if lines:
                return "\n\n".join(lines)
    return json.dumps(payload, ensure_ascii=False)


def extract_expected_action_names(task_payload: dict[str, Any]) -> list[str]:
    expected_actions = list(task_payload.get("expected_actions") or [])
    names: list[str] = []
    for action in expected_actions:
        if isinstance(action, dict):
            name = action.get("name")
            if name:
                names.append(str(name))
    return names


WRITE_TOOLS = {
    "cancel_reservation",
    "update_reservation",
    "modify_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "send_certificate",
    "book_reservation",
}


def compute_live_task_reward(
    *,
    task_payload: dict[str, Any],
    executed_tools: list[str],
    final_assistant_text: str,
) -> float:
    """Lightweight task success approximation for the first live pilot.

    This is intentionally conservative and benchmark-facing:
    - refuse-like tasks need a refusal-style final response and no write tool calls
    - otherwise we check expected action names if present
    - otherwise we fall back to the expected write-action count when available

    The real tau evaluator can replace this later once the end-to-end runtime is
    available inside the training stack.
    """
    label = str(task_payload.get("label", "") or "").strip().lower()
    expected_action_names = extract_expected_action_names(task_payload)
    num_expected_actions = int(task_payload.get("num_expected_actions") or 0)
    final_text = (final_assistant_text or "").lower()
    executed_write_names = [name for name in executed_tools if name in WRITE_TOOLS]

    refuse_cues = ["cannot", "can't", "unable", "not eligible", "not allowed", "not covered"]
    if label == "refuse":
        if not executed_write_names and any(cue in final_text for cue in refuse_cues):
            return 1.0
        return 0.0

    if expected_action_names:
        if all(name in executed_tools for name in expected_action_names):
            return 1.0
        return 0.0

    if num_expected_actions > 0:
        return 1.0 if len(executed_write_names) >= num_expected_actions else 0.0

    return 0.0


class BedrockTau3UserSimulator:
    """Simple user-simulator fallback driven by the task scenario text.

    This is a practical fallback for the live SDPO pilot when the benchmark host
    has Bedrock configured but we are not directly calling tau2's internal
    orchestrator. It keeps the task/runtime public and benchmark-facing while the
    training logic remains inside SDPO.
    """

    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        self.region = region
        self._client = None
        self.max_retries = 3
        self.retry_delay_s = 1.0

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def generate_user_response(self, task_instruction: str, conversation: list[dict[str, Any]]) -> str:
        system_prompt = (
            "You are simulating the customer side of a tau3 benchmark conversation.\n"
            "Stay consistent with the task instruction and prior conversation.\n"
            "If the issue is fully resolved, reply with the exact token <END_OF_CONVERSATION>.\n\n"
            f"Task instruction:\n{task_instruction}"
        )

        messages = [{"role": "user", "content": [{"text": system_prompt}]}]
        for message in conversation:
            role = message.get("role")
            content = str(message.get("content", "") or "")
            if role == "assistant":
                messages.append({"role": "assistant", "content": [{"text": f"[Agent] {content}"}]})
            elif role == "user":
                messages.append({"role": "user", "content": [{"text": content}]})
            elif role == "tool":
                messages.append({"role": "assistant", "content": [{"text": f"[Tool result] {content}"}]})
        messages.append({"role": "user", "content": [{"text": "What do you say next as the customer?"}]})

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._get_client().converse(
                    modelId=self.model_id,
                    messages=messages,
                    inferenceConfig={"maxTokens": 220, "temperature": 0.6},
                )
                return response["output"]["message"]["content"][0]["text"]
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_error = exc
                if attempt + 1 == self.max_retries:
                    break
                time.sleep(self.retry_delay_s * (attempt + 1))
        raise RuntimeError(f"Bedrock user simulator failed after {self.max_retries} attempts: {last_error}")


@dataclass
class Tau3LiveSession:
    request_id: str
    domain: str
    task_id: str
    task_split: str
    task_payload: dict[str, Any]
    task_instruction: str
    env: Any
    max_steps: int
    user_simulator: BedrockTau3UserSimulator | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    last_tool_result_preview: str = ""
    final_assistant_message: str = ""
    latest_user_message: str = ""
    terminated: bool = False
    terminal_reason: str = "running"

    def build_summary(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "task_id": self.task_id,
            "task_split": self.task_split,
            "status": "terminated" if self.terminated else "running",
            "terminal_reason": self.terminal_reason,
            "turn_count": len(self.messages),
            "executed_tools": list(self.executed_tools),
            "last_tool_result_preview": self.last_tool_result_preview,
            "latest_user_message": compact_text(self.latest_user_message),
            "latest_assistant_message": compact_text(self.final_assistant_message),
        }


class Tau3LiveSessionManager:
    _sessions: dict[str, Tau3LiveSession] = {}
    _lock = Lock()

    @classmethod
    def start_session(
        cls,
        *,
        request_id: str,
        domain: str,
        task_id: str,
        task_split: str = "test",
        max_steps: int = 1000,
        user_model: str | None = None,
        user_region: str | None = None,
    ) -> Tau3LiveSession:
        if not user_model:
            raise ValueError(
                "tau3 live interaction requires a user simulator model. "
                "Set TAU3_LIVE_USER_MODEL before launching the live pilot."
            )
        get_environment, _ = get_domain_env_and_tasks(domain)
        available_tasks = get_domain_tasks_for_split(domain, task_split)
        task = None
        for item in available_tasks:
            if str(getattr(item, "id", "")) != str(task_id):
                continue
            task = item
            break
        if task is None:
            raise ValueError(f"tau3 task {task_id} not found in domain {domain} for split {task_split}")

        env = get_environment()
        if hasattr(env, "reset"):
            try:
                env.reset(task_id=int(task_id))
            except Exception as exc:
                raise RuntimeError(f"Failed to reset tau3 environment for task {task_id}: {exc}") from exc

        task_payload = to_jsonable(task)
        task_instruction = scenario_to_instruction_text(getattr(task, "user_scenario", None))
        simulator = None
        if user_model:
            simulator = BedrockTau3UserSimulator(model_id=user_model, region=user_region or "us-east-1")

        session = Tau3LiveSession(
            request_id=request_id,
            domain=domain,
            task_id=str(task_id),
            task_split=str(task_split),
            task_payload=task_payload,
            task_instruction=task_instruction,
            env=env,
            max_steps=max_steps,
            user_simulator=simulator,
        )
        with cls._lock:
            cls._sessions[request_id] = session
        return session

    @classmethod
    def get_session(cls, request_id: str) -> Tau3LiveSession:
        with cls._lock:
            if request_id not in cls._sessions:
                raise KeyError(f"tau3 live session {request_id} not found")
            return cls._sessions[request_id]

    @classmethod
    def execute_tool(cls, request_id: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = cls.get_session(request_id)
        tool_fn = getattr(session.env, tool_name, None)
        call_succeeded = False
        if tool_fn is None:
            result = {"error": f"Unknown tau3 tool: {tool_name}"}
        else:
            try:
                result = tool_fn(**arguments)
                call_succeeded = True
            except Exception as exc:
                result = {"error": str(exc)[:200]}

        rendered = result if isinstance(result, str) else json.dumps(to_jsonable(result), ensure_ascii=False)
        if call_succeeded:
            session.executed_tools.append(tool_name)
        session.last_tool_result_preview = compact_text(rendered, limit=300)
        session.tool_events.append(
            {
                "name": tool_name,
                "arguments": to_jsonable(arguments),
                "result_preview": session.last_tool_result_preview,
                "success": call_succeeded,
            }
        )
        if call_succeeded and tool_name == "transfer_to_human_agents":
            session.terminated = True
            session.terminal_reason = "transferred_to_human"
        return {
            "result_text": rendered,
            "tau3_live_result": session.build_summary(),
        }

    @classmethod
    def advance_user_turn(cls, request_id: str, messages: list[dict[str, Any]]) -> tuple[bool, str, float, dict[str, Any]]:
        session = cls.get_session(request_id)
        session.messages = list(messages)
        assistant_message = ""
        for item in reversed(messages):
            if item.get("role") == "assistant":
                assistant_message = str(item.get("content", "") or "")
                break
        session.final_assistant_message = assistant_message

        if session.terminated:
            summary = session.build_summary()
            summary["final_reward"] = compute_live_task_reward(
                task_payload=session.task_payload,
                executed_tools=session.executed_tools,
                final_assistant_text=session.final_assistant_message,
            )
            return True, "", float(summary["final_reward"]), {"tau3_live_result": summary}

        if len([msg for msg in messages if msg.get("role") == "user"]) >= session.max_steps:
            session.terminated = True
            session.terminal_reason = "max_user_turns"
            summary = session.build_summary()
            summary["final_reward"] = compute_live_task_reward(
                task_payload=session.task_payload,
                executed_tools=session.executed_tools,
                final_assistant_text=session.final_assistant_message,
            )
            return True, "", float(summary["final_reward"]), {"tau3_live_result": summary}

        user_response = "<END_OF_CONVERSATION>"
        if session.user_simulator is not None:
            user_response = session.user_simulator.generate_user_response(session.task_instruction, session.messages)

        if user_response.strip() == "<END_OF_CONVERSATION>":
            session.terminated = True
            session.terminal_reason = "user_signaled_done"
            summary = session.build_summary()
            summary["final_reward"] = compute_live_task_reward(
                task_payload=session.task_payload,
                executed_tools=session.executed_tools,
                final_assistant_text=session.final_assistant_message,
            )
            return True, "", float(summary["final_reward"]), {"tau3_live_result": summary}

        session.latest_user_message = user_response
        return False, user_response, 0.0, {"tau3_live_result": session.build_summary()}

    @classmethod
    def finalize_session(cls, request_id: str) -> None:
        with cls._lock:
            cls._sessions.pop(request_id, None)


def default_tau3_output_dir() -> Path:
    return Path(os.environ.get("TAU3_LIVE_OUTPUT_DIR", "./datasets/tau3_live_airline"))


# ---------------------------------------------------------------------------
# Official tau Gym path.
#
# This is intentionally added beside the legacy/proxy Tau3LiveSessionManager
# rather than replacing it. Select it with TAU3_LIVE_RUNTIME=official_gym or by
# passing runtime: official_gym in the interaction config.
# ---------------------------------------------------------------------------


def tau3_runtime_mode(value: str | None = None) -> str:
    mode = (value or os.environ.get("TAU3_LIVE_RUNTIME", "proxy_legacy")).strip().lower()
    if mode in {"gym", "official", "official_gym", "agent_gym", "agentgym"}:
        return "official_gym"
    if mode in {"proxy", "legacy", "proxy_legacy", "custom"}:
        return "proxy_legacy"
    raise ValueError(f"Unsupported TAU3_LIVE_RUNTIME={mode}")


def _parse_json_env(name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = os.environ.get(name)
    if not raw:
        return dict(default or {})
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"{name} must be valid JSON when set: {raw}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must decode to a JSON object: {raw}")
    return value


_ORIGINAL_TAU2_NL_ASSERTION_GENERATE: Any | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _strip_litellm_bedrock_prefix(model: str) -> str:
    return model[len("bedrock/") :] if model.startswith("bedrock/") else model


def _litellm_bedrock_model_id(model: str) -> str:
    if "/" in model:
        return model
    return f"bedrock/{model}"


def _format_tau2_messages_for_bedrock(messages: list[Any]) -> str:
    system_parts: list[str] = []
    body_parts: list[str] = []
    for message in messages:
        role = str(getattr(message, "role", "") or "")
        content = str(getattr(message, "content", "") or "")
        if role == "system":
            system_parts.append(content)
        else:
            body_parts.append(f"{role.upper() or 'MESSAGE'}:\n{content}")

    parts: list[str] = []
    if system_parts:
        parts.append("System instructions:\n" + "\n\n".join(system_parts))
    if body_parts:
        parts.append("Conversation and assertions:\n" + "\n\n".join(body_parts))
    return "\n\n".join(parts)


_TAU2_NL_ASSERTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "expectedOutcome": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "metExpectation": {"type": "boolean"},
                },
                "required": ["expectedOutcome", "reasoning", "metExpectation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def _bedrock_structured_nl_assertion_generate(
    model: str,
    messages: list[Any],
    tools: Any = None,
    tool_choice: Any = None,
    call_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """Structured Bedrock replacement for tau2's NL assertion generate call.

    tau2's retail evaluator parses the assistant response as JSON. Bedrock
    structured output makes that parse deterministic while keeping the patch
    local to the NL assertion evaluator.
    """
    del tools, tool_choice, call_name, kwargs
    try:
        import boto3
        from tau2.data_model.message import AssistantMessage
    except ImportError as exc:  # pragma: no cover - live-only dependency
        raise RuntimeError("TAU3_NL_ASSERTION_STRUCTURED_OUTPUT=1 requires boto3 and tau2.") from exc

    model_id = _strip_litellm_bedrock_prefix(os.environ.get("TAU3_NL_ASSERTION_MODEL", model).strip())
    region = os.environ.get("TAU3_NL_ASSERTION_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
    max_tokens = int(os.environ.get("TAU3_NL_ASSERTION_MAX_TOKENS", "1024"))
    temperature = float(os.environ.get("TAU3_NL_ASSERTION_TEMPERATURE", "0.0"))
    max_retries = int(os.environ.get("TAU3_NL_ASSERTION_MAX_RETRIES", "2"))

    prompt = _format_tau2_messages_for_bedrock(messages)
    client = boto3.client("bedrock-runtime", region_name=region)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
                outputConfig={
                    "textFormat": {
                        "type": "json_schema",
                        "structure": {
                            "jsonSchema": {
                                "schema": json.dumps(_TAU2_NL_ASSERTION_SCHEMA),
                                "name": "tau3_nl_assertion",
                                "description": "tau3 natural-language assertion evaluation",
                            }
                        },
                    }
                },
            )
            content = response["output"]["message"]["content"][0]["text"]
            return AssistantMessage(role="assistant", content=content)
        except Exception as exc:  # pragma: no cover - live Bedrock path
            last_error = exc
            if attempt < max_retries:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Bedrock NL assertion judge failed after {max_retries + 1} attempts: {last_error}")


def _configure_tau2_nl_assertion_judge_from_env() -> dict[str, Any]:
    """Patch tau2's NL assertion evaluator when explicitly requested.

    The patch is intentionally environment-gated. With TAU3_NL_ASSERTION_MODEL
    unset, current airline/retail behavior is unchanged.
    """
    model = os.environ.get("TAU3_NL_ASSERTION_MODEL", "").strip()
    if not model:
        return {"enabled": False}

    litellm_model = _litellm_bedrock_model_id(model)
    args = {
        "temperature": float(os.environ.get("TAU3_NL_ASSERTION_TEMPERATURE", "0.0")),
        "max_tokens": int(os.environ.get("TAU3_NL_ASSERTION_MAX_TOKENS", "1024")),
    }
    structured_output = _env_flag("TAU3_NL_ASSERTION_STRUCTURED_OUTPUT", default=False)

    try:
        import tau2.config as tau2_config
        import tau2.evaluator.evaluator_nl_assertions as nl_assertions
    except ImportError as exc:  # pragma: no cover - live-only dependency
        raise RuntimeError(
            "TAU3_NL_ASSERTION_MODEL was set, but tau2's NL assertion evaluator could not be imported."
        ) from exc

    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = litellm_model
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(args)
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = litellm_model
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(args)

    if structured_output:
        global _ORIGINAL_TAU2_NL_ASSERTION_GENERATE
        if _ORIGINAL_TAU2_NL_ASSERTION_GENERATE is None:
            _ORIGINAL_TAU2_NL_ASSERTION_GENERATE = nl_assertions.generate
        nl_assertions.generate = _bedrock_structured_nl_assertion_generate

    return {
        "enabled": True,
        "model": litellm_model,
        "bedrock_model_id": _strip_litellm_bedrock_prefix(model),
        "region": os.environ.get("TAU3_NL_ASSERTION_REGION") or os.environ.get("AWS_REGION") or "us-east-1",
        "max_tokens": args["max_tokens"],
        "temperature": args["temperature"],
        "structured_output": structured_output,
    }


def _json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(to_jsonable(value), ensure_ascii=False)
    except Exception:
        return str(value)


@dataclass
class Tau3GymLiveSession:
    request_id: str
    domain: str
    task_id: str
    task_split: str
    env: Any
    max_steps: int
    observation: str
    info: dict[str, Any]
    messages: list[dict[str, Any]] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    final_assistant_message: str = ""
    latest_user_message: str = ""
    latest_observation: str = ""
    turn_count: int = 0
    terminated: bool = False
    terminal_reason: str = "running"
    final_reward: float = 0.0
    reward_info_json: str = "{}"
    simulation_run_json: str = "{}"
    last_action_type: str = ""
    last_parse_error: str = ""
    nl_assertion_judge: dict[str, Any] = field(default_factory=dict)

    def _visible_tool_names(self) -> list[str]:
        tools = self.info.get("tools") or []
        names: list[str] = []
        for tool in tools:
            name = getattr(tool, "name", None)
            if name:
                names.append(str(name))
        return names

    def build_summary(self) -> dict[str, Any]:
        return {
            "runtime": "official_gym",
            "domain": self.domain,
            "task_id": self.task_id,
            "task_split": self.task_split,
            "status": "terminated" if self.terminated else "running",
            "terminal_reason": self.terminal_reason,
            "turn_count": self.turn_count,
            "executed_tools": list(self.executed_tools),
            "tool_events": list(self.tool_events),
            "available_tools": self._visible_tool_names(),
            "latest_user_message": compact_text(self.latest_user_message),
            "latest_assistant_message": compact_text(self.final_assistant_message),
            "last_observation_preview": compact_text(self.latest_observation or self.observation, limit=360),
            "last_action_type": self.last_action_type,
            "last_parse_error": self.last_parse_error,
            "nl_assertion_judge": to_jsonable(self.nl_assertion_judge),
            "final_reward": float(self.final_reward),
            "reward_info_json": self.reward_info_json,
            "simulation_run_json": self.simulation_run_json,
            "diagnostic_source": "tau_agent_gym",
        }


class Tau3GymLiveSessionManager:
    """Official tau AgentGymEnv-backed runtime with the same surface as the legacy manager.

    The existing Verl ToolAgentLoop remains in charge of token generation and
    token/logprob capture. This manager only owns tau's user/tools/environment/
    evaluator side by stepping AgentGymEnv with completed assistant actions.
    """

    _sessions: dict[str, Tau3GymLiveSession] = {}
    _lock = Lock()

    @classmethod
    def start_session(
        cls,
        *,
        request_id: str,
        domain: str,
        task_id: str,
        task_split: str = "base",
        max_steps: int = 100,
        user_model: str | None = None,
        user_region: str | None = None,
        **kwargs,
    ) -> Tau3GymLiveSession:
        del user_region  # tau Gym user LLM args handle provider details.
        try:
            from tau2.gym.gym_agent import AgentGymEnv
        except ImportError as exc:  # pragma: no cover - live-only dependency
            raise RuntimeError(
                "Could not import tau2.gym.gym_agent.AgentGymEnv. Install tau2-bench with gym extras."
            ) from exc

        resolved_user_model = user_model or os.environ.get("TAU3_LIVE_USER_MODEL")
        if not resolved_user_model:
            raise ValueError(
                "tau3 official_gym interaction requires a user simulator model. "
                "Set TAU3_LIVE_USER_MODEL before launching the live pilot."
            )

        nl_assertion_judge = _configure_tau2_nl_assertion_judge_from_env()
        user_llm_args = _parse_json_env("TAU3_LIVE_USER_ARGS_JSON", {})
        user_llm_args.update(kwargs.get("user_llm_args") or {})

        seed_raw = kwargs.get("seed", None)
        if seed_raw is None:
            seed_env = os.environ.get("TAU3_LIVE_SEED")
            seed_raw = int(seed_env) if seed_env else None

        env = AgentGymEnv(
            domain=domain,
            task_id=str(task_id),
            max_steps=max_steps,
            solo_mode=False,
            user_llm=resolved_user_model,
            user_llm_args=user_llm_args or None,
            all_messages_as_observation=os.environ.get("TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION", "1") != "0",
        )
        task_split_name = str(task_split or "base")
        if task_split_name != "base":
            split_tasks = get_domain_tasks_for_split(domain, task_split_name)
            split_task = next((task for task in split_tasks if str(task.id) == str(task_id)), None)
            if split_task is None:
                raise ValueError(f"tau3 task {task_id} not found in domain {domain} for split {task_split_name}")

            def _get_task_for_requested_split(self):
                return split_task

            env._get_task = MethodType(_get_task_for_requested_split, env)
        observation, info = env.reset(seed=seed_raw)

        session = Tau3GymLiveSession(
            request_id=request_id,
            domain=domain,
            task_id=str(task_id),
            task_split=str(task_split or "base"),
            env=env,
            max_steps=max_steps,
            observation=observation,
            latest_observation=observation,
            info=info,
            nl_assertion_judge=nl_assertion_judge,
        )
        with cls._lock:
            cls._sessions[request_id] = session
        return session

    @classmethod
    def get_session(cls, request_id: str) -> Tau3GymLiveSession:
        with cls._lock:
            if request_id not in cls._sessions:
                raise KeyError(f"tau3 official_gym session {request_id} not found")
            return cls._sessions[request_id]

    @staticmethod
    def _update_from_step(
        session: Tau3GymLiveSession,
        *,
        observation: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        action_type: str,
        assistant_text: str = "",
        tool_name: str | None = None,
        tool_arguments: dict[str, Any] | None = None,
        parse_error: str = "",
    ) -> dict[str, Any]:
        session.observation = observation
        session.latest_observation = observation
        session.info = info or {}
        session.turn_count += 1
        session.final_reward = float(reward or 0.0)
        reward_info_payload = to_jsonable(session.info.get("reward_info") or {})
        simulation_run_payload = to_jsonable(session.info.get("simulation_run") or {})
        session.reward_info_json = _json_string(reward_info_payload)
        session.simulation_run_json = _json_string(simulation_run_payload)
        session.last_action_type = action_type
        session.last_parse_error = parse_error
        if assistant_text:
            session.final_assistant_message = assistant_text
        if tool_name:
            observation_text = observation or ""
            tool_success = "Invalid action with error" not in observation_text and "error" not in observation_text.lower()
            if tool_success:
                session.executed_tools.append(tool_name)
            session.tool_events.append(
                {
                    "name": tool_name,
                    "argument_keys": sorted((tool_arguments or {}).keys()),
                    "success": tool_success,
                    "result_preview": compact_text(observation_text, limit=300),
                }
            )

        if terminated or truncated:
            session.terminated = True
            simulation_reason = ""
            if isinstance(simulation_run_payload, dict):
                simulation_reason = str(simulation_run_payload.get("termination_reason", "") or "").strip()
            if not simulation_reason and isinstance(reward_info_payload, dict):
                info_note = reward_info_payload.get("info") or {}
                if isinstance(info_note, dict):
                    note_text = str(info_note.get("note", "") or "").lower()
                    if "max_steps" in note_text:
                        simulation_reason = "max_steps"
            session.terminal_reason = simulation_reason or ("truncated" if truncated else "agent_or_user_stop")

        return session.build_summary()

    @classmethod
    def step_action(
        cls,
        request_id: str,
        action: str,
        *,
        action_type: str,
        assistant_text: str = "",
        tool_name: str | None = None,
        tool_arguments: dict[str, Any] | None = None,
        parse_error: str = "",
    ) -> tuple[bool, str, float, dict[str, Any]]:
        session = cls.get_session(request_id)
        if session.terminated:
            summary = session.build_summary()
            return True, "", float(summary["final_reward"]), {"tau3_live_result": summary, "tau3_should_terminate": True}

        observation, reward, terminated, truncated, info = session.env.step(action)
        summary = cls._update_from_step(
            session,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            action_type=action_type,
            assistant_text=assistant_text,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            parse_error=parse_error,
        )
        return bool(terminated or truncated), observation, float(reward or 0.0), {
            "tau3_live_result": summary,
            "tau3_should_terminate": bool(terminated or truncated),
        }

    @classmethod
    def execute_tool(cls, request_id: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from verl.utils.tau3_action_parser import tool_call_to_action

        action = tool_call_to_action(tool_name, arguments)
        should_terminate, observation, reward, additional = cls.step_action(
            request_id,
            action,
            action_type="tool_json",
            tool_name=tool_name,
            tool_arguments=arguments,
        )
        return {
            "result_text": observation,
            "reward": reward,
            "should_terminate": should_terminate,
            "tau3_live_result": additional["tau3_live_result"],
        }

    @classmethod
    def advance_user_turn(cls, request_id: str, messages: list[dict[str, Any]]) -> tuple[bool, str, float, dict[str, Any]]:
        from verl.utils.tau3_action_parser import extract_tool_name_and_arguments, parse_model_output_to_tau_action

        session = cls.get_session(request_id)
        session.messages = list(messages)

        assistant_message = ""
        for item in reversed(messages):
            if item.get("role") == "assistant":
                assistant_message = str(item.get("content", "") or "")
                break
        session.final_assistant_message = assistant_message

        parsed = parse_model_output_to_tau_action(assistant_message)
        tool_name, tool_arguments = extract_tool_name_and_arguments(assistant_message, parsed.action_type)
        should_terminate, observation, reward, additional = cls.step_action(
            request_id,
            parsed.action_for_env,
            action_type=parsed.action_type,
            assistant_text=assistant_message,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            parse_error=parsed.parse_error or "",
        )
        session.latest_user_message = observation
        return should_terminate, observation, reward, additional

    @classmethod
    def finalize_session(cls, request_id: str) -> None:
        with cls._lock:
            cls._sessions.pop(request_id, None)
