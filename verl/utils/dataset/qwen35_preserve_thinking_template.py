"""Qwen3.5 chat-template override that preserves historical thinking.

The stock Qwen3/Qwen3.5 thinking templates may hide previous assistant
reasoning when rendering multi-turn chats. That is reasonable for inference,
but it breaks full-trajectory thinking SFT where old assistant ``<think>``
blocks are part of the supervised transcript.

This module keeps a small, explicit ChatML-style template in one place so SFT
pre-tokenization and checkpoint/rollout tokenizer patching can share it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


PRESERVE_THINKING_CHAT_TEMPLATE = r"""{%- if tools %}
<|im_start|>system
You have access to the following tools. If you call a tool, emit exactly one tool call and no user-facing text in the same assistant turn.
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools><|im_end|>
{%- endif %}
{%- for message in messages %}
{%- set role = message['role'] %}
<|im_start|>{{ role }}
{%- if role == 'assistant' %}
{{ message.get('content', '') or '' }}
{%- if message.get('tool_calls') %}
{%- for tool_call in message['tool_calls'] %}
{%- set function = tool_call.get('function', tool_call) %}
{%- set arguments = function.get('arguments', {}) %}
<tool_call>
<function={{ function.get('name', '') }}>
{%- if arguments is mapping %}
{%- for parameter_name, parameter_value in arguments.items() %}
<parameter={{ parameter_name }}>
{{ parameter_value | tojson }}
</parameter>
{%- endfor %}
{%- else %}
<parameter=arguments>
{{ arguments | tojson }}
</parameter>
{%- endif %}
</function>
</tool_call>
{%- endfor %}
{%- endif %}<|im_end|>
{%- elif role == 'tool' %}
{{ message.get('content', '') or '' }}<|im_end|>
{%- else %}
{{ message.get('content', '') or '' }}<|im_end|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}"""


def preserve_thinking_template_sha256() -> str:
    return hashlib.sha256(PRESERVE_THINKING_CHAT_TEMPLATE.encode("utf-8")).hexdigest()


def install_preserve_thinking_chat_template(processing_class: Any) -> Any:
    """Mutate a tokenizer/processor to use the preserve-thinking template."""

    processing_class.chat_template = PRESERVE_THINKING_CHAT_TEMPLATE
    if hasattr(processing_class, "tokenizer") and getattr(processing_class, "tokenizer") is not None:
        processing_class.tokenizer.chat_template = PRESERVE_THINKING_CHAT_TEMPLATE
    return processing_class


def write_preserve_thinking_template(path: str | Path) -> Path:
    """Write the template to ``path`` for audit/reuse."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PRESERVE_THINKING_CHAT_TEMPLATE + "\n", encoding="utf-8")
    return path
