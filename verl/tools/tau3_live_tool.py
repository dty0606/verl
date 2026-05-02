from __future__ import annotations

import os
from typing import Any, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.tau3_live_runtime import Tau3GymLiveSessionManager, Tau3LiveSessionManager, tau3_runtime_mode


def _select_manager(runtime: str | None):
    mode = tau3_runtime_mode(runtime)
    if mode == "official_gym":
        return Tau3GymLiveSessionManager
    return Tau3LiveSessionManager


class Tau3LiveTool(BaseTool):
    """Generic tau/tau3 live tool wrapper.

    In official_gym mode, this does not execute local env methods directly.
    Instead, it forwards the parsed tool call to AgentGymEnv.step(...), allowing
    official tau Orchestrator/UserSimulator/environment/evaluator semantics.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.runtime = tau3_runtime_mode((config or {}).get("runtime") or os.environ.get("TAU3_LIVE_RUNTIME", "proxy_legacy"))

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        del instance_id
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            raise ValueError("Tau3LiveTool requires agent_data to locate the live tau3 session")

        runtime = tau3_runtime_mode(
            getattr(agent_data, "interaction_kwargs", {}).get("runtime")
            or getattr(getattr(agent_data, "interaction", None), "runtime", None)
            or self.runtime
            or os.environ.get("TAU3_LIVE_RUNTIME", "proxy_legacy")
        )
        manager = _select_manager(runtime)

        result = manager.execute_tool(
            request_id=getattr(agent_data, "interaction_instance_id", None) or agent_data.request_id,
            tool_name=self.name,
            arguments=parameters,
        )
        additional = {
            "tau3_live_result": result["tau3_live_result"],
            "tau3_should_terminate": bool(result.get("should_terminate", False)),
            "tau3_runtime": runtime,
        }
        return ToolResponse(text=result["result_text"]), float(result.get("reward", 0.0) or 0.0), additional

    async def release(self, instance_id: str, **kwargs) -> None:
        del instance_id, kwargs
        return None
