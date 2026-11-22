# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed W4/W8A16 expert GEMVs for small batches on Intel XPU."""

import torch

from vllm.triton_utils import tl, triton

from ._codec import (
    PackedProjection,
    f16_e2m1,
    f16_e4m3,
    mxfp8_scale,
    nvfp4_scale,
    swiglu,
)


@triton.jit
def _row_dot(
    x,
    W,
    S,
    Amax,
    Denominator,
    expert,
    rows,
    K: tl.constexpr,
    NVFP4: tl.constexpr,
    CH: tl.constexpr,
):
    j16 = tl.arange(0, 16)
    acc = tl.zeros(rows.shape, tl.float32)
    if NVFP4:
        wrow = W + rows.to(tl.int64)[:, None, None] * (K // 2)
        srow = S + rows.to(tl.int64)[:, None] * (K // 16)
        # Accumulate each half group before multiplying by its scale.
        for c0 in range(0, K // 32, CH):
            cc = c0 + tl.arange(0, CH)
            b = tl.load(wrow + (cc * 16)[None, :, None] + j16[None, None, :])
            b = b.to(tl.int32)
            xa = tl.load(x + (cc * 32)[:, None] + j16[None, :]).to(tl.float32)
            xb = tl.load(x + (cc * 32 + 16)[:, None] + j16[None, :]).to(tl.float32)
            pa = tl.sum(f16_e2m1(b & 15) * xa[None, :, :], 2)
            pb = tl.sum(f16_e2m1(b >> 4) * xb[None, :, :], 2)
            sa = f16_e4m3(tl.load(srow + 2 * cc[None, :]).to(tl.int32))
            sb = f16_e4m3(tl.load(srow + 2 * cc[None, :] + 1).to(tl.int32))
            acc += tl.sum(pa * sa + pb * sb, 1)
        acc *= nvfp4_scale(Amax, Denominator, expert)
    else:
        j32 = tl.arange(0, 32)
        wrow = W + rows.to(tl.int64)[:, None, None] * K
        srow = S + rows.to(tl.int64)[:, None] * (K // 32)
        for c0 in range(0, K // 32, CH):
            cc = c0 + tl.arange(0, CH)
            b = tl.load(wrow + (cc * 32)[None, :, None] + j32[None, None, :])
            xk = tl.load(x + (cc * 32)[:, None] + j32[None, :]).to(tl.float32)
            p = tl.sum(f16_e4m3(b.to(tl.int32)) * xk[None, :, :], 2)
            scale = mxfp8_scale(tl.load(srow + cc[None, :]).to(tl.int32))
            acc += tl.sum(p * scale, 1)
    return acc


@triton.jit
def _gate_up(
    X,
    Ids,
    W,
    S,
    Amax,
    Denominator,
    Out,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    NVFP4: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    BN: tl.constexpr,
    CH: tl.constexpr,
):
    route = tl.program_id(0)
    expert = tl.load(Ids + route)
    if expert >= 0:
        token = route // TOPK
        rows = expert * N + tl.program_id(1) * BN + tl.arange(0, BN)
        acc = _row_dot(
            X + token.to(tl.int64) * K,
            W,
            S,
            Amax,
            Denominator,
            expert,
            rows,
            K,
            NVFP4,
            CH,
        )
        pairs = tl.reshape(acc.to(tl.bfloat16).to(tl.float32), (BN // 2, 2))
        gate, up = tl.split(pairs)
        col = tl.program_id(1) * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(
            Out + route.to(tl.int64) * (N // 2) + col,
            swiglu(gate, up, GATE_CAP, UP_CAP),
        )


@triton.jit
def _down_combine(
    Act,
    Ids,
    Weights,
    W,
    S,
    Amax,
    Denominator,
    Out,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    NVFP4: tl.constexpr,
    BN: tl.constexpr,
    CH: tl.constexpr,
):
    token = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    out = tl.zeros((BN,), tl.float32)
    for slot in range(TOPK):
        route = token * TOPK + slot
        expert = tl.load(Ids + route)
        if expert >= 0:
            d = _row_dot(
                Act + route.to(tl.int64) * K,
                W,
                S,
                Amax,
                Denominator,
                expert,
                expert * N + n,
                K,
                NVFP4,
                CH,
            )
            out += d.to(tl.bfloat16).to(tl.float32) * tl.load(Weights + route)
    tl.store(Out + token.to(tl.int64) * N + n, out)


def decode_experts(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    w13: PackedProjection,
    w2: PackedProjection,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """Run route-wise gate/up GEMVs and fuse down GEMVs with the reduction."""
    tokens, hidden = x.shape
    routes, topk = ids.numel(), ids.shape[1]
    nvfp4 = w13[2] is not None
    bn, warps = (16, 4) if nvfp4 else (32, 8)
    chunks = 4
    n13 = w13[0].shape[1]
    intermediate = n13 // 2
    assert hidden % (32 * chunks) == intermediate % (32 * chunks) == 0
    activated = x.new_empty((routes, intermediate))
    _gate_up[(routes, n13 // bn)](
        x,
        ids,
        w13[0].view(torch.uint8),
        w13[1].view(torch.uint8),
        *w13[2:],
        activated,
        n13,
        hidden,
        topk,
        nvfp4,
        gate_cap,
        up_cap,
        bn,
        chunks,
        num_warps=warps,
    )
    out = torch.empty_like(x)
    _down_combine[(tokens, hidden // bn)](
        activated,
        ids,
        weights,
        w2[0].view(torch.uint8),
        w2[1].view(torch.uint8),
        *w2[2:],
        out,
        hidden,
        intermediate,
        topk,
        nvfp4,
        bn,
        chunks,
        num_warps=warps,
    )
    return out
