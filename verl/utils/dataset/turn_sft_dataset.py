# Turn-level SFT dataset for Qwen-style reasoning/tool chat templates.
#
# Rows are AReaL tau2-style:
#   messages = prior conversation history
#   answer = current assistant target only
#
# This avoids supervising historical assistant <think> blocks that Qwen3.5
# intentionally strips when rendering full multi-turn conversations.

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from verl.utils import hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.simple_sft_dataset import (
    clean_none,
    flatten_message_tool_calls,
    normalize_messages_for_qwen_template,
    normalize_tool_call,
    normalize_tools_for_qwen_template,
)
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _as_token_list(tokenized: Any) -> list[int]:
    if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return normalize_token_ids(tokenized)


def _contiguous_spans(mask: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _answer_to_message(answer: dict[str, Any]) -> dict[str, Any]:
    answer = dict(clean_none(copy.deepcopy(answer)))
    content = str(answer.get("content") or "")
    thinking = str(answer.get("thinking") or answer.get("reasoning") or "").strip()
    if thinking:
        content = f"<think>\n{thinking}\n</think>\n{content}"

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if answer.get("tool_calls"):
        message["tool_calls"] = [normalize_tool_call(tool_call) for tool_call in answer["tool_calls"]]
    return message


def _extract_thinking_from_rendered_content(content: str) -> str:
    if "</think>" not in content:
        return ""
    return content.split("</think>", 1)[0].split("<think>", 1)[-1].strip()


class TurnSFTDataset(Dataset):
    """SFT dataset that labels exactly one current assistant answer per row."""

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
        self.answer_key = self.config.get("answer_key", "answer")
        self.tools_key = self.config.get("tools_key", "tools")
        self.enable_thinking_key = self.config.get("enable_thinking_key", "enable_thinking")
        self.enable_thinking_default = self.config.get("enable_thinking_default", None)
        self.apply_chat_template_kwargs = self.config.get("apply_chat_template_kwargs", {})
        self.audit_samples = int(self.config.get("audit_samples", 0) or 0)
        self.audit_max_chars = int(self.config.get("audit_max_chars", 800) or 800)

        dataframes = []
        for parquet_file in parquet_files:
            local_path = copy_local_path_from_hdfs(parquet_file)
            dataframes.append(pd.read_parquet(local_path, dtype_backend="pyarrow"))
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        if max_samples > 0 and max_samples < len(self.dataframe):
            indices = np.random.default_rng(seed=42).choice(len(self.dataframe), max_samples, replace=False)
            self.dataframe = self.dataframe.iloc[indices.tolist()]

        raw_messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()
        raw_answers = self.dataframe[self.answer_key].apply(convert_nested_value_to_list_recursive).tolist()
        self.messages = [normalize_messages_for_qwen_template(messages) for messages in raw_messages]
        self.answers = [_answer_to_message(answer) for answer in raw_answers]

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

        print(f"TurnSFTDataset: {len(self.dataframe)} samples, max_length={self.max_length}")

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

    def _render_token_ids(self, messages, tools, template_kwargs, *, add_generation_prompt=False) -> list[int]:
        return _as_token_list(
            self._processor.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                **template_kwargs,
            )
        )

    def _choose_template_messages(self, messages, tools, template_kwargs) -> tuple[list[dict[str, Any]], str]:
        try:
            self._render_token_ids(messages, tools, template_kwargs, add_generation_prompt=False)
            return messages, "nested"
        except Exception as nested_exc:
            flat_messages = flatten_message_tool_calls(messages)
            try:
                self._render_token_ids(flat_messages, tools, template_kwargs, add_generation_prompt=False)
                logger.warning(
                    "TurnSFTDataset fell back to flat tool_call shape after nested rendering failed: %s",
                    nested_exc,
                )
                return flat_messages, "flat"
            except Exception as flat_exc:
                raise RuntimeError(
                    "Qwen chat-template rendering failed for both nested and flat tool_call shapes. "
                    f"nested_error={type(nested_exc).__name__}: {nested_exc}; "
                    f"flat_error={type(flat_exc).__name__}: {flat_exc}"
                ) from flat_exc

    def _audit_report(self, item, messages, answer, input_ids, loss_mask, template_shape) -> dict[str, Any]:
        spans = _contiguous_spans(loss_mask)
        decoded_target = "".join(
            self.tokenizer.decode(input_ids[start:end], skip_special_tokens=False) for start, end in spans
        )
        decoded_preview = decoded_target[: self.audit_max_chars]
        failures: list[str] = []
        if len(spans) != 1:
            failures.append(f"target_span_count_mismatch: spans={len(spans)}")

        content = str(answer.get("content") or "")
        thinking = _extract_thinking_from_rendered_content(content)
        if thinking:
            thinking_probe = thinking[: min(32, len(thinking))]
            if thinking_probe and thinking_probe not in decoded_target:
                failures.append("missing_thinking_text_in_target")
            if "</think>" not in decoded_target:
                failures.append("missing_think_close_in_target")

        for tool_call in answer.get("tool_calls") or []:
            function_entry = tool_call.get("function") if isinstance(tool_call, dict) else None
            tool_name = (
                function_entry.get("name")
                if isinstance(function_entry, dict)
                else tool_call.get("name")
                if isinstance(tool_call, dict)
                else None
            )
            if tool_name and tool_name not in decoded_target:
                failures.append(f"missing_tool_name_in_target: {tool_name}")

        if not sum(loss_mask):
            failures.append("empty_loss_mask")

        return {
            "item": int(item),
            "history_messages": len(messages),
            "target_labeled_tokens": int(sum(loss_mask)),
            "spans": spans,
            "decoded_target": decoded_preview,
            "template_shape": template_shape,
            "failures": failures,
        }

    def audit_item(self, item: int) -> dict[str, Any]:
        _, report = self._build_item(item, audit=True)
        return report

    def _build_item(self, item: int, *, audit: bool = False):
        messages = self.messages[item]
        answer = self.answers[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        template_kwargs = self._template_kwargs(enable_thinking)
        full_messages, template_shape = self._choose_template_messages([*messages, answer], tools, template_kwargs)
        prompt_messages = full_messages[:-1]

        prompt_ids = self._render_token_ids(
            prompt_messages,
            tools,
            template_kwargs,
            add_generation_prompt=True,
        )
        full_ids = self._render_token_ids(
            full_messages,
            tools,
            template_kwargs,
            add_generation_prompt=False,
        )
        prompt_len = len(prompt_ids)
        loss_mask_values = [0] * min(prompt_len, len(full_ids)) + [1] * max(0, len(full_ids) - prompt_len)
        loss_mask_values = loss_mask_values[: len(full_ids)]

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

        report = (
            self._audit_report(item, messages, answer, input_ids.tolist(), loss_mask.tolist(), template_shape)
            if audit
            else {}
        )
        if audit and report["failures"]:
            raise AssertionError(f"TurnSFTDataset audit failed for item {item}: {report['failures']}")

        if self.audit_samples and item < self.audit_samples:
            audit_report = self._audit_report(
                item, messages, answer, input_ids.tolist(), loss_mask.tolist(), template_shape
            )
            logger.warning("TurnSFTDataset audit item %s: %s", item, json.dumps(audit_report, ensure_ascii=False))
            if audit_report["failures"]:
                raise AssertionError(f"TurnSFTDataset audit failed for item {item}: {audit_report['failures']}")

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}, report

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

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }, report

        raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def __getitem__(self, item):
        result, _ = self._build_item(item, audit=False)
        return result
