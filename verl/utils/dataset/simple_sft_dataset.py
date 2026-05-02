# Simple text-only SFT dataset for strict Qwen-style chat templates.
#
# This dataset intentionally avoids VERL's default per-message chat-template
# calls. It renders valid conversation prefixes instead, normalizes tool-call
# arguments before rendering, and can audit masked assistant spans.

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from verl.utils import hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def clean_none(value: Any) -> Any:
    """Recursively drop None-valued mapping keys after parquet deserialization."""
    if isinstance(value, Mapping):
        return {k: clean_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [clean_none(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_none(value.tolist())
    return value


def _coerce_arguments_to_mapping(arguments: Any) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, Mapping):
        return dict(clean_none(arguments))
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"__raw_arguments__": arguments}
        if isinstance(parsed, Mapping):
            return dict(clean_none(parsed))
        return {"__value__": parsed}
    return {"__value__": arguments}


def normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    """Normalize one assistant tool call for HF/Jinja template rendering.

    OpenAI API messages commonly store function arguments as JSON strings, but
    several Qwen/VERL Jinja templates iterate over ``arguments.items()``. For
    training-time rendering we therefore coerce arguments to a mapping.
    """
    if not isinstance(tool_call, Mapping):
        raise ValueError(f"Assistant tool_call must be a mapping, got {type(tool_call)!r}")

    normalized = dict(clean_none(copy.deepcopy(tool_call)))
    function_entry = normalized.get("function")

    if isinstance(function_entry, Mapping):
        function_entry = dict(function_entry)
        function_entry["arguments"] = _coerce_arguments_to_mapping(function_entry.get("arguments"))
        normalized["function"] = function_entry
        normalized.setdefault("type", "function")
        return normalized

    if "name" in normalized:
        normalized["arguments"] = _coerce_arguments_to_mapping(normalized.get("arguments"))
        return normalized

    return normalized


def flatten_tool_call_for_qwen(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    """Convert OpenAI nested tool call shape to the flat shape used by some Qwen templates."""
    function_entry = tool_call.get("function")
    if isinstance(function_entry, Mapping):
        return {
            "name": function_entry.get("name", tool_call.get("name", "")),
            "arguments": _coerce_arguments_to_mapping(function_entry.get("arguments")),
        }
    return dict(tool_call)


def normalize_messages_for_qwen_template(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Clean messages and normalize assistant tool calls for Qwen chat templates."""
    normalized_messages: list[dict[str, Any]] = []
    seen_non_system = False

    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            raise ValueError(f"Message must be a mapping, got {type(raw_message)!r}")

        message = dict(clean_none(copy.deepcopy(raw_message)))
        role = message.get("role")

        # Tau3 simulation traces start with an assistant greeting. Qwen chat
        # templates require user-first after the optional system message.
        if role != "system":
            if not seen_non_system and role == "assistant" and not message.get("tool_calls"):
                seen_non_system = True
                continue
            seen_non_system = True

        tool_calls = message.get("tool_calls")
        if tool_calls:
            message["tool_calls"] = [normalize_tool_call(tool_call) for tool_call in tool_calls]
        else:
            message.pop("tool_calls", None)

        normalized_messages.append(message)

    return normalized_messages


def flatten_message_tool_calls(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flat_messages: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(copy.deepcopy(message))
        if copied.get("tool_calls"):
            copied["tool_calls"] = [flatten_tool_call_for_qwen(tool_call) for tool_call in copied["tool_calls"]]
        flat_messages.append(copied)
    return flat_messages


def normalize_tools_for_qwen_template(tools: Any) -> Any:
    return clean_none(copy.deepcopy(tools))


def _as_token_list(tokenized: Any) -> list[int]:
    if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return normalize_token_ids(tokenized)


def _contiguous_spans(mask: Sequence[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


class SimpleSFTDataset(Dataset):
    """Text-only SFT dataset that renders valid Qwen chat-template prefixes."""

    def __init__(
        self,
        parquet_files,
        tokenizer,
        config: DictConfig,
        processor=None,
        max_samples: int = -1,
    ):
        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or {}
        self.max_length = self.config.get("max_length", 32768)
        self.truncation = self.config.get("truncation", "error")
        self.pad_mode = DatasetPadMode(self.config.get("pad_mode", "no_padding"))
        self.messages_key = self.config.get("messages_key", "messages")
        self.tools_key = self.config.get("tools_key", "tools")
        self.enable_thinking_key = self.config.get("enable_thinking_key", "enable_thinking")
        self.enable_thinking_default = self.config.get("enable_thinking_default", None)
        self.apply_chat_template_kwargs = self.config.get("apply_chat_template_kwargs", {})
        self.audit_samples = int(self.config.get("audit_samples", 0) or 0)
        self.audit_max_chars = int(self.config.get("audit_max_chars", 500) or 500)

        dataframes = []
        for parquet_file in parquet_files:
            local_path = copy_local_path_from_hdfs(parquet_file)
            dataframes.append(pd.read_parquet(local_path, dtype_backend="pyarrow"))
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        if max_samples > 0 and max_samples < len(self.dataframe):
            indices = np.random.default_rng(seed=42).choice(len(self.dataframe), max_samples, replace=False)
            self.dataframe = self.dataframe.iloc[indices.tolist()]

        raw_messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()
        self.messages = [normalize_messages_for_qwen_template(messages) for messages in raw_messages]

        if self.tools_key in self.dataframe.columns:
            self.tools = (
                self.dataframe[self.tools_key]
                .apply(convert_nested_value_to_list_recursive)
                .apply(normalize_tools_for_qwen_template)
                .tolist()
            )
        else:
            self.tools = None

        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

        print(f"SimpleSFTDataset: {len(self.dataframe)} samples, max_length={self.max_length}")

    def __len__(self):
        return len(self.dataframe)

    @property
    def _processor(self):
        return self.processor if self.processor is not None else self.tokenizer

    def _template_kwargs(self, enable_thinking: bool | None) -> dict[str, Any]:
        kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return kwargs

    def _apply_chat_template(self, messages, tools, template_kwargs, *, add_generation_prompt=False):
        return self._processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **template_kwargs,
        )

    def _render_token_ids(self, messages, tools, template_kwargs, *, add_generation_prompt=False) -> list[int]:
        return _as_token_list(
            self._apply_chat_template(
                messages,
                tools,
                template_kwargs,
                add_generation_prompt=add_generation_prompt,
            )
        )

    def _choose_template_messages(self, messages, tools, template_kwargs) -> tuple[list[dict[str, Any]], str]:
        """Use OpenAI nested tool calls first, then flat Qwen-style calls as fallback."""
        try:
            self._render_token_ids(messages, tools, template_kwargs, add_generation_prompt=False)
            return messages, "nested"
        except Exception as nested_exc:
            flat_messages = flatten_message_tool_calls(messages)
            try:
                self._render_token_ids(flat_messages, tools, template_kwargs, add_generation_prompt=False)
                logger.warning(
                    "SimpleSFTDataset fell back to flat tool_call shape after nested tool_call rendering failed: %s",
                    nested_exc,
                )
                return flat_messages, "flat"
            except Exception as flat_exc:
                raise RuntimeError(
                    "Qwen chat-template rendering failed for both nested and flat tool_call shapes. "
                    f"nested_error={type(nested_exc).__name__}: {nested_exc}; "
                    f"flat_error={type(flat_exc).__name__}: {flat_exc}"
                ) from flat_exc

    def _assistant_loss_mask(self, messages, tools, template_kwargs, full_len: int) -> list[int]:
        loss_mask = [0] * full_len
        assistant_indices = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]

        for assistant_index in assistant_indices:
            prefix_messages = messages[:assistant_index]
            if prefix_messages:
                prefix_ids = self._render_token_ids(
                    prefix_messages,
                    tools,
                    template_kwargs,
                    add_generation_prompt=True,
                )
                prefix_len = len(prefix_ids)
            else:
                prefix_len = 0

            suffix_ids = self._render_token_ids(
                messages[: assistant_index + 1],
                tools,
                template_kwargs,
                add_generation_prompt=False,
            )
            suffix_len = len(suffix_ids)

            for token_index in range(prefix_len, min(suffix_len, full_len)):
                loss_mask[token_index] = 1

        return loss_mask

    def _audit_report(self, item, messages, input_ids, loss_mask) -> dict[str, Any]:
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        spans = _contiguous_spans(loss_mask)
        decoded_spans = [
            self.tokenizer.decode(input_ids[start:end], skip_special_tokens=False)[: self.audit_max_chars]
            for start, end in spans
        ]

        failures: list[str] = []
        if len(spans) != len(assistant_messages):
            failures.append(f"assistant_span_count_mismatch: spans={len(spans)} assistants={len(assistant_messages)}")

        for index, message in enumerate(assistant_messages[: len(decoded_spans)]):
            decoded = decoded_spans[index]
            content = str(message.get("content") or "")
            if "<think" in content and "<think" not in decoded:
                failures.append(f"missing_think_in_span_{index}")
            for tool_call in message.get("tool_calls") or []:
                function_entry = tool_call.get("function") if isinstance(tool_call, Mapping) else None
                tool_name = (
                    function_entry.get("name")
                    if isinstance(function_entry, Mapping)
                    else tool_call.get("name")
                    if isinstance(tool_call, Mapping)
                    else None
                )
                if tool_name and tool_name not in decoded:
                    failures.append(f"missing_tool_name_in_span_{index}: {tool_name}")

        return {
            "item": int(item),
            "assistant_count": len(assistant_messages),
            "masked_span_count": len(spans),
            "labeled_tokens": int(sum(loss_mask)),
            "spans": spans,
            "decoded_spans": decoded_spans,
            "failures": failures,
        }

    def audit_item(self, item: int) -> dict[str, Any]:
        _, report = self._build_item(item, audit=True)
        return report

    def _build_item(self, item: int, *, audit: bool = False):
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        template_kwargs = self._template_kwargs(enable_thinking)
        messages, template_shape = self._choose_template_messages(messages, tools, template_kwargs)
        full_ids = self._render_token_ids(messages, tools, template_kwargs, add_generation_prompt=False)
        loss_mask_values = self._assistant_loss_mask(messages, tools, template_kwargs, len(full_ids))

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask_values, dtype=torch.long)
        position_ids = torch.arange(len(full_ids), dtype=torch.long)

        sequence_length = len(input_ids)
        if sequence_length > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"Sequence length {sequence_length} > max_length {self.max_length}")
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
                position_ids = position_ids[-self.max_length :]
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        report = self._audit_report(item, messages, input_ids.tolist(), loss_mask.tolist()) if audit else {}
        if report:
            report["template_shape"] = template_shape
        if audit and report["failures"]:
            raise AssertionError(f"SimpleSFTDataset audit failed for item {item}: {report['failures']}")

        if self.audit_samples and item < self.audit_samples:
            audit_report = self._audit_report(item, messages, input_ids.tolist(), loss_mask.tolist())
            logger.warning("SimpleSFTDataset audit item %s: %s", item, json.dumps(audit_report, ensure_ascii=False))
            if audit_report["failures"]:
                raise AssertionError(f"SimpleSFTDataset audit failed for item {item}: {audit_report['failures']}")

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            result = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            return result, report

        if self.pad_mode == DatasetPadMode.RIGHT:
            if len(input_ids) < self.max_length:
                pad_len = self.max_length - len(input_ids)
                pad_id = self.tokenizer.pad_token_id or 0
                attention_mask = torch.cat(
                    [torch.ones(len(input_ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]
                )
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=torch.long)])
                position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])
            else:
                attention_mask = torch.ones_like(input_ids)

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            return result, report

        raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def __getitem__(self, item):
        result, _ = self._build_item(item, audit=False)
        return result
