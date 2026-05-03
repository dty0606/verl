"""Pre-tokenized SFT dataset — reads input_ids and loss_mask directly from parquet.

No chat template calls during training. Use scripts/tau3/pretokenize_turn_sft.py
to create the pre-tokenized parquet first.
"""
import logging
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _as_flat_int_list(value, *, name: str, item: int) -> list[int]:
    value = convert_nested_value_to_list_recursive(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"{name} for item {item} must be a list, got {type(value).__name__}")
    if any(isinstance(v, (list, tuple, dict)) for v in value):
        raise ValueError(f"{name} for item {item} must be a flat 1-D list")
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} for item {item} contains non-integer values") from exc


class PretokenizedSFTDataset(Dataset):
    """SFT dataset that reads pre-tokenized input_ids and loss_mask from parquet."""

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

        self.tokenizer = tokenizer
        self.config = config or {}
        self.max_length = self.config.get("max_length", 32768)
        self.truncation = self.config.get("truncation", "right")
        self.pad_mode = DatasetPadMode(self.config.get("pad_mode", "no_padding"))

        dataframes = []
        for f in parquet_files:
            local_path = copy_local_path_from_hdfs(f)
            dataframes.append(pd.read_parquet(local_path))
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        if max_samples > 0 and max_samples < len(self.dataframe):
            indices = np.random.default_rng(seed=42).choice(len(self.dataframe), max_samples, replace=False)
            self.dataframe = self.dataframe.iloc[indices.tolist()]

        self.input_ids_col = self.dataframe["input_ids"].apply(convert_nested_value_to_list_recursive).tolist()
        self.loss_mask_col = self.dataframe["loss_mask"].apply(convert_nested_value_to_list_recursive).tolist()

        print(f"PretokenizedSFTDataset: {len(self.dataframe)} samples, max_length={self.max_length}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        input_ids = _as_flat_int_list(self.input_ids_col[item], name="input_ids", item=item)
        loss_mask = _as_flat_int_list(self.loss_mask_col[item], name="loss_mask", item=item)
        if len(input_ids) != len(loss_mask):
            raise ValueError(
                f"input_ids/loss_mask length mismatch for item {item}: {len(input_ids)} vs {len(loss_mask)}"
            )
        if any(mask_value not in (0, 1) for mask_value in loss_mask):
            raise ValueError(f"loss_mask for item {item} must be binary 0/1 values")

        seq_len = len(input_ids)
        if seq_len > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == "error":
                raise ValueError(f"Sequence length {seq_len} > max_length {self.max_length}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        if loss_mask and loss_mask[0]:
            # sft_loss shifts a flattened jagged mask. Never label the first
            # token of a sample because there is no in-sample previous token.
            loss_mask[0] = 0
        if sum(loss_mask) <= 0:
            raise ValueError(f"Empty loss mask for item {item} after truncation={self.truncation}")

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        position_ids = torch.arange(len(input_ids), dtype=torch.long)

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}

        if self.pad_mode == DatasetPadMode.RIGHT:
            cur_len = len(input_ids)
            if cur_len < self.max_length:
                pad_len = self.max_length - cur_len
                pad_id = self.tokenizer.pad_token_id or 0
                attention_mask = torch.cat([
                    torch.ones(cur_len, dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ])
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
            }

        raise ValueError(f"Unknown pad mode {self.pad_mode}")
