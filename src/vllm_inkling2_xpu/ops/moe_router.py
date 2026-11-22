# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F


def select_experts(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the routed expert IDs and scaled routing weights."""
    if x.dtype != torch.bfloat16:
        raise ValueError("Inkling2 XPU MoE requires BF16 activations")
    if x.shape[0] == 0:
        return (
            x.new_empty((0, topk), dtype=torch.int32),
            x.new_empty((0, topk), dtype=torch.float32),
        )
    logits = torch.mm(x, weight.t(), out_dtype=torch.float32)
    scores = logits.sigmoid() + bias
    # Bias is only for selection; ties prefer lower expert IDs.
    ids = scores.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    weights = F.logsigmoid(logits.gather(-1, ids)).softmax(dim=-1) * route_scale
    return ids.to(torch.int32).contiguous(), weights
