# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compose XPU short convolution and attention residual mixing."""

import torch

from .attn_res import attn_res
from .sconv import short_conv


def conv_mix(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    conv_out: torch.Tensor,
    residual: torch.Tensor | None,
    addend: torch.Tensor | None,
    stack: torch.Tensor,
    last: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply short conv then mix the BF16 rows, preserving residual rounding."""
    short_conv(
        x,
        conv_weight,
        cache,
        positions,
        block_table,
        seq_idx,
        slot_mapping,
        query_start,
        stream_offset,
        stream_width,
        addend,
        residual,
        conv_out,
    )
    return attn_res(
        stack,
        conv_out if last is None else last,
        None if last is None else conv_out,
        queries,
        norm_weight,
        norm_eps,
        True,
        eps,
    )
