# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-sorted W4/W8A16 MoE with 2D block loads on Intel XPU."""

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton

from ._codec import (
    PackedProjection,
    f16_e2m1,
    f16_e4m3,
    mxfp8_scale,
    nvfp4_scale,
    swiglu,
)


@torch.no_grad()
def repack_nvfp4_split_half(weight: torch.Tensor) -> None:
    """Repack NVFP4 weights in place from adjacent nibbles to split halves.

    vllm-xpu-kernels has no NVFP4 grouped GEMM. Pairing codes 16 elements
    apart within each 32-element group lets the XPU kernels dequantize both
    halves in registers. Repack one expert at a time to bound temporary memory.
    """
    _, n, half_k = weight.shape
    for expert_weight in weight:
        codes = torch.stack((expert_weight & 15, expert_weight >> 4), dim=-1)
        codes = codes.reshape(n, half_k // 16, 2, 16)
        expert_weight.copy_((codes[:, :, 0] | (codes[:, :, 1] << 4)).reshape(n, half_k))


@triton.jit
def _gather_rows(
    X,
    SortedRoutes,
    NumPadded,
    SortedX,
    routes,
    tokens,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row < tl.load(NumPadded):
        route = tl.load(SortedRoutes + row)
        token = tl.minimum(route // TOPK, tokens - 1).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        for k0 in tl.static_range(0, K, BLOCK):
            c = k0 + cols
            v = tl.load(X + token * K + c, c < K, 0)
            tl.store(SortedX + row * K + c, tl.where(route < routes, v, 0), c < K)


@triton.jit
def _packed_gemm(
    X,
    W,
    Scales,
    Amax,
    Denominator,
    SortedRoutes,
    TileExpert,
    NumPadded,
    Out,
    routes,
    padded,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    NVFP4: tl.constexpr,
    ACTIVATE: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pn = tl.program_id(0)
    tile_m = tl.program_id(1)
    first = tile_m * BM
    if first < tl.load(NumPadded):
        expert = tl.load(TileExpert + tile_m)
        n0 = pn * BN
        wrow = expert * N + n0
        nn = tl.arange(0, BN)
        xd = tl.make_tensor_descriptor(X, [padded, K], [K, 1], [BM, 16])
        WK: tl.constexpr = K // 2 if NVFP4 else K
        wd = tl.make_tensor_descriptor(W, [E * N, WK], [WK, 1], [BN, 16])
        srow = Scales + (wrow + nn).to(tl.int64) * (K // 16 if NVFP4 else K // 32)
        acc = tl.zeros((BM, BN), tl.float32)
        for c in range(K // 32):
            k0 = c * 32
            xa = xd.load([first, k0])
            xb = xd.load([first, k0 + 16])
            if NVFP4:
                packed = tl.trans(wd.load([wrow, c * 16])).to(tl.int32)
                sa = f16_e4m3(tl.load(srow + 2 * c).to(tl.int32))
                sb = f16_e4m3(tl.load(srow + 2 * c + 1).to(tl.int32))
                wa = (f16_e2m1(packed & 15) * sa[None, :]).to(tl.bfloat16)
                wb = (f16_e2m1(packed >> 4) * sb[None, :]).to(tl.bfloat16)
            else:
                scale = mxfp8_scale(tl.load(srow + c).to(tl.int32))
                ca = tl.trans(wd.load([wrow, k0])).to(tl.int32)
                cb = tl.trans(wd.load([wrow, k0 + 16])).to(tl.int32)
                wa = (f16_e4m3(ca) * scale[None, :]).to(tl.bfloat16)
                wb = (f16_e4m3(cb) * scale[None, :]).to(tl.bfloat16)
            acc = tl.dot(xa, wa, acc)
            acc = tl.dot(xb, wb, acc)
        if NVFP4:
            acc *= nvfp4_scale(Amax, Denominator, expert)
        rows = first + tl.arange(0, BM)
        if ACTIVATE:
            # Round the gate/up pairs once before activating, in sorted order.
            pairs = tl.reshape(acc.to(tl.bfloat16).to(tl.float32), (BM, BN // 2, 2))
            gate, up = tl.split(pairs)
            col = pn * (BN // 2) + tl.arange(0, BN // 2)
            tl.store(
                Out + rows.to(tl.int64)[:, None] * (N // 2) + col[None, :],
                swiglu(gate, up, GATE_CAP, UP_CAP),
            )
        else:
            route = tl.load(SortedRoutes + rows).to(tl.int64)
            tl.store(
                Out + route[:, None] * N + (n0 + nn)[None, :],
                acc,
                (route < routes)[:, None],
            )


@triton.jit
def _weighted_combine(
    Down,
    Weights,
    IsPadding,
    Out,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = True
    if IsPadding is not None:
        live = not tl.load(IsPadding + token)
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(TOPK):
        route = token * TOPK + slot
        value = tl.load(Down + route * H + col, live & (col < H), 0).to(tl.float32)
        weight = tl.load(Weights + route, live, 0)
        acc += value * weight
    tl.store(Out + token * H + col, acc, col < H)


def _project(x, projection, out, layout, routes, bm, bn, launch, caps=None):
    values, scales, amax, denominator = projection
    experts, n = values.shape[:2]
    gate_cap, up_cap = (0.0, 0.0) if caps is None else caps
    assert n % bn == 0
    _packed_gemm[(n // bn, layout[1].numel())](
        x,
        values.view(torch.uint8),
        scales.view(torch.uint8),
        amax,
        denominator,
        *layout,
        out,
        routes,
        x.shape[0],
        n,
        x.shape[1],
        experts,
        amax is not None,
        caps is not None,
        gate_cap,
        up_cap,
        bm,
        bn,
        **launch,
    )


def sorted_experts(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    is_padding: torch.Tensor | None,
    w13: PackedProjection,
    w2: PackedProjection,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """Gather expert-sorted activations once, then scatter down-projection rows."""
    tokens, hidden = x.shape
    routes, topk = ids.numel(), ids.shape[1]
    experts = w13[0].shape[0]
    launch: dict[str, int | str]
    if routes <= 16 * experts:
        bm, bn, launch = 16, 64, {"num_warps": 4}
    else:
        bm, bn, launch = 64, 128, {"num_warps": 8, "grf_mode": "256"}
    layout = moe_align_block_size(ids, bm, experts, pad_sorted_ids=True)
    sorted_ids, _, num_padded = layout
    padded = sorted_ids.numel()
    sorted_x = x.new_empty((padded, hidden))
    _gather_rows[(padded,)](
        x,
        sorted_ids,
        num_padded,
        sorted_x,
        routes,
        tokens,
        hidden,
        topk,
        256,
        num_warps=4,
    )
    activated = x.new_empty((padded, w13[0].shape[1] // 2))
    _project(
        sorted_x, w13, activated, layout, routes, bm, bn, launch, (gate_cap, up_cap)
    )
    down = x.new_empty((routes, hidden))
    _project(activated, w2, down, layout, routes, bm, bn, launch)
    out = torch.empty_like(x)
    block = 256
    _weighted_combine[(tokens, triton.cdiv(hidden, block))](
        down,
        weights,
        is_padding,
        out,
        hidden,
        topk,
        block,
        num_warps=4,
    )
    return out
