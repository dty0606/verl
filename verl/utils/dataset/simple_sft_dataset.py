# Simple SFT dataset that tokenizes the full conversation at once.
#
# Unlike MultiTurnSFTDataset which calls apply_chat_template per turn,
# this class renders the full conversation to tokens in one call, then
# builds the loss mask by comparing the full-conversation tokens with
# a no-generation-prompt version to identify assistant response regions.
#
# This avoids Qwen3.5 chat template errors from passing single tool/assistant
# messages without the required conversation context.

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from verl.utils import hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SimpleSFTDataset(Dataset):
    """SFT dataset that tokenizes full conversations in one shot.

    Loss is computed on assistant tokens only. The assistant regions are
    identified by tokenizing the conversation twice:
    1. Full conversation (all messages)
    2. Conversation with the last assistant message removed

    The difference gives us the last assistant response tokens. We repeat
    this for each assistant turn by building prefix masks.

    For simplicity, this implementation uses a single full-conversation
    tokenization and marks ALL assistant content tokens for loss, using
    a role-based heuristic after tokenization.
    """

    def __init__(
        self,
        parquet_files,
        tokenizer,
        config: DictConfig,
        processor=None,
        max_samples: int = -1,
    ):
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_length = config.get("max_length", 32768)
        self.truncation = config.get("truncation", "error")
        self.pad_mode = DatasetPadMode(config.get("pad_mode", "no_padding"))
        self.messages_key = config.get("messages_key", "messages")
        self.tools_key = config.get("tools_key", "tools")
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")
        self.enable_thinking_default = config.get("enable_thinking_default", None)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # Load data
        dataframes = []
        for f in parquet_files:
            local_path = copy_local_path_from_hdfs(f)
            dataframes.append(pd.read_parquet(local_path))
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        if max_samples > 0 and max_samples < len(self.dataframe):
            indices = np.random.default_rng(seed=42).choice(len(self.dataframe), max_samples, replace=False)
            self.dataframe = self.dataframe.iloc[indices.tolist()]

        # Extract and clean messages
        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()
        for i, conversation in enumerate(self.messages):
            self.messages[i] = [{k: v for k, v in msg.items() if v is not None} for msg in conversation]

        # Extract tools
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None

        # Extract enable_thinking
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

        print(f"SimpleSFTDataset: {len(self.dataframe)} samples, max_length={self.max_length}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        processor = self.processor if self.processor is not None else self.tokenizer
        template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking

        # Tokenize full conversation at once — no per-turn calls.
        full_ids = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            **template_kwargs,
        )
        if isinstance(full_ids, list) and full_ids and isinstance(full_ids[0], list):
            full_ids = full_ids[0]
        full_ids = normalize_token_ids(full_ids)

        # Build loss mask: 1 for assistant tokens, 0 for everything else.
        # Strategy: tokenize prefixes ending at each assistant boundary to
        # find where assistant content starts and ends.
        loss_mask = [0] * len(full_ids)

        # Find assistant message indices
        assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]

        for asst_idx in assistant_indices:
            # Tokens up to (not including) this assistant message
            prefix_msgs = messages[:asst_idx]
            if not prefix_msgs:
                # First message is assistant (shouldn't happen after greeting strip, but handle it)
                prefix_len = 0
            else:
                prefix_ids = processor.apply_chat_template(
                    prefix_msgs,
                    tools=tools,
                    add_generation_prompt=True,  # includes the generation prompt before assistant
                    tokenize=True,
                    **template_kwargs,
                )
                if isinstance(prefix_ids, list) and prefix_ids and isinstance(prefix_ids[0], list):
                    prefix_ids = prefix_ids[0]
                prefix_ids = normalize_token_ids(prefix_ids)
                prefix_len = len(prefix_ids)

            # Tokens up to and including this assistant message
            suffix_msgs = messages[: asst_idx + 1]
            suffix_ids = processor.apply_chat_template(
                suffix_msgs,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                **template_kwargs,
            )
            if isinstance(suffix_ids, list) and suffix_ids and isinstance(suffix_ids[0], list):
                suffix_ids = suffix_ids[0]
            suffix_ids = normalize_token_ids(suffix_ids)
            suffix_len = len(suffix_ids)

            # Mark assistant tokens for loss
            for j in range(prefix_len, min(suffix_len, len(full_ids))):
                loss_mask[j] = 1

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        position_ids = torch.arange(len(full_ids), dtype=torch.long)

        # Handle length
        seq_len = len(input_ids)
        if seq_len > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"Sequence length {seq_len} > max_length {self.max_length}")
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
                position_ids = position_ids[-self.max_length :]

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        elif self.pad_mode == DatasetPadMode.RIGHT:
            if seq_len < self.max_length:
                pad_len = self.max_length - seq_len
                pad_id = self.tokenizer.pad_token_id or 0
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=torch.long)])
                attention_mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ])
                position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])
            else:
                attention_mask = torch.ones_like(input_ids)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")
