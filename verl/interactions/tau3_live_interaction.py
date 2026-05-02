from __future__ import annotations

import asyncio
import os
from typing import Any, Optional
from uuid import uuid4

from verl.utils.tau3_live_runtime import Tau3GymLiveSessionManager, Tau3LiveSessionManager, tau3_runtime_mode

try:
    from .base import BaseInteraction
except ModuleNotFoundError as exc:  # pragma: no cover - compatibility with newer Verl forks
    if exc.name not in {"verl.interactions.base", f"{__package__}.base"}:
        raise

    class BaseInteraction:
        """Minimal local fallback when the legacy interaction package is absent."""

        def __init__(self, config: dict[str, Any]):
            self.config = config
            self.name: str = config.get("name", "interaction_agent")


def _select_manager(runtime: str | None):
    mode = tau3_runtime_mode(runtime)
    if mode == "official_gym":
        return Tau3GymLiveSessionManager
    return Tau3LiveSessionManager


class Tau3LiveInteraction(BaseInteraction):
    """Live tau/tau3 bridge for SDPO multi-turn rollouts.

    runtime=official_gym uses the official tau AgentGymEnv / UserSimulator /
    Orchestrator / evaluator path while keeping Verl in charge of actor
    generation. runtime=proxy_legacy preserves the previous custom Bedrock loop.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.max_steps = int(config.get("max_steps", 12))
        self.default_user_model = config.get("user_model") or os.environ.get("TAU3_LIVE_USER_MODEL")
        self.user_region = config.get("user_region", "us-east-1")
        self.runtime = tau3_runtime_mode(config.get("runtime") or os.environ.get("TAU3_LIVE_RUNTIME", "proxy_legacy"))

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        domain: str = "airline",
        task_id: str | int = "",
        task_split: str = "base",
        user_model: Optional[str] = None,
        runtime: Optional[str] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        kwargs.pop("name", None)
        selected_runtime = tau3_runtime_mode(runtime or kwargs.pop("runtime", None) or self.runtime)
        manager = _select_manager(selected_runtime)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: manager.start_session(
                request_id=instance_id,
                domain=domain,
                task_id=str(task_id),
                task_split=task_split,
                max_steps=self.max_steps,
                user_model=user_model or self.default_user_model,
                user_region=self.user_region,
                **kwargs,
            ),
        )
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        selected_runtime = tau3_runtime_mode(kwargs.get("runtime") or self.runtime)
        manager = _select_manager(selected_runtime)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: manager.advance_user_turn(instance_id, messages))

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        selected_runtime = tau3_runtime_mode(kwargs.get("runtime") or self.runtime)
        manager = _select_manager(selected_runtime)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: manager.finalize_session(instance_id))
