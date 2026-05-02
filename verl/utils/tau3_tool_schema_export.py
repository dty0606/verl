from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def get_tau_environment(domain: str, solo_mode: bool = False):
    """Construct the official tau environment for a domain."""

    from tau2.registry import registry

    env_ctor = registry.get_env_constructor(domain)
    return env_ctor(solo_mode=solo_mode)


def _done_tool_schema() -> dict[str, Any]:
    """GymAgent adds a done tool internally; expose it to the model too."""

    return {
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
    }


def export_openai_tool_schemas(
    domain: str,
    solo_mode: bool = False,
    include_done: bool = True,
) -> list[dict[str, Any]]:
    """Return official tau OpenAI-style tool schemas.

    Source of truth:
      environment.get_tools() -> list[Tool]
      tool.openai_schema -> {"type": "function", "function": {...}}

    The optional done tool mirrors AgentGymEnv/GymAgent's stop tool so the
    policy can explicitly terminate when the task is complete.
    """

    env = get_tau_environment(domain=domain, solo_mode=solo_mode)
    tools = env.get_tools()

    schemas: list[dict[str, Any]] = []
    for tool in tools:
        schema = tool.openai_schema
        validate_openai_schema(schema)
        # Ensure `required` field is always present (pydantic validator requires it).
        # Tau tools with no required args may omit it; force an empty list.
        params = schema["function"]["parameters"]
        if "required" not in params:
            params["required"] = []
        schemas.append(schema)

    if include_done and "done" not in {s["function"]["name"] for s in schemas}:
        schemas.append(_done_tool_schema())

    return schemas


def validate_openai_schema(schema: dict[str, Any]) -> None:
    assert schema.get("type") == "function", schema
    assert "function" in schema, schema
    fn = schema["function"]
    assert fn.get("name"), schema
    assert "parameters" in fn, schema
    assert "properties" in fn["parameters"], schema


def convert_to_verl_tool_config(openai_schema: dict[str, Any]) -> dict[str, Any]:
    """Convert official tau schema to the existing Verl YAML shape."""

    return {
        "class_name": "verl.tools.tau3_live_tool.Tau3LiveTool",
        "config": {"type": "native"},
        "tool_schema": openai_schema,
    }


def export_verl_tool_config(
    domain: str,
    out_path: str | Path,
    solo_mode: bool = False,
    include_done: bool = True,
) -> None:
    import yaml

    schemas = export_openai_tool_schemas(domain=domain, solo_mode=solo_mode, include_done=include_done)
    payload = {
        "schema_source": "tau2.environment.Tool.openai_schema",
        "domain": domain,
        "tools": [convert_to_verl_tool_config(s) for s in schemas],
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def smoke_assert_airline_schema() -> None:
    schemas = export_openai_tool_schemas("airline")
    by_name = {s["function"]["name"]: s for s in schemas}

    assert "get_user_details" in by_name, sorted(by_name)
    props = by_name["get_user_details"]["function"]["parameters"].get("properties", {})
    assert "user_id" in props, props

    nonempty = [s for s in schemas if s["function"]["parameters"].get("properties")]
    assert len(nonempty) >= max(1, len(schemas) // 2), (
        f"Too many empty schemas: {len(nonempty)}/{len(schemas)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="airline")
    parser.add_argument(
        "--out",
        default="examples/sglang_multiturn/config/tool_config/tau3_live_airline_tool_config.yaml",
    )
    parser.add_argument("--solo-mode", action="store_true")
    parser.add_argument("--no-done-tool", action="store_true")
    args = parser.parse_args()

    if args.domain == "airline":
        smoke_assert_airline_schema()

    export_verl_tool_config(
        domain=args.domain,
        out_path=args.out,
        solo_mode=args.solo_mode,
        include_done=not args.no_done_tool,
    )
    print(f"Wrote official tau tool config to {args.out}")


if __name__ == "__main__":
    main()
