# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from ._codec import PackedProjection
from .moe_decode import decode_experts
from .moe_packed import sorted_experts


def validate_config(
    hidden: int, intermediate: int, topk: int, experts: int, quant_format: str
) -> None:
    """Check the model dimensions against the XPU kernels' tiles."""
    if quant_format not in ("nvfp4", "mxfp8"):
        raise NotImplementedError("Inkling2 XPU requires NVFP4 or MXFP8 weights")
    # 128 aligns both the largest GEMM N tile and the four K groups of GEMV.
    if min(hidden, intermediate) <= 0 or hidden % 128 or intermediate % 128:
        raise ValueError(
            "Inkling2 XPU needs hidden and intermediate sizes divisible by 128"
        )
    if not 0 < topk <= experts:
        raise ValueError("Inkling2 XPU requires 0 < top-k <= number of experts")


def packed_experts(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    is_padding: torch.Tensor | None,
    w13: PackedProjection,
    w2: PackedProjection,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """Run the packed expert projections and return the reduced BF16 output."""
    tokens = x.shape[0]
    if tokens == 0:
        return torch.empty_like(x)
    if is_padding is not None:
        if (
            is_padding.shape != (tokens,)
            or is_padding.dtype != torch.bool
            or is_padding.device != x.device
        ):
            raise ValueError("Inkling2 XPU MoE expects a bool [tokens] padding mask")
        ids = ids.masked_fill(is_padding[:, None], -1)
    if tokens <= 4:
        return decode_experts(x, ids, weights, w13, w2, gate_cap, up_cap)
    return sorted_experts(x, ids, weights, is_padding, w13, w2, gate_cap, up_cap)
