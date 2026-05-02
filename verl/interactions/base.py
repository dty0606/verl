from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4


class BaseInteraction:
    """Small interaction interface used by multi-turn agent loops."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "interaction_agent")

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        del kwargs
        return str(uuid4()) if instance_id is None else instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        del instance_id, messages, kwargs
        return False, "Your current result seems acceptable.", 0.0, {}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del instance_id, kwargs
        return None
