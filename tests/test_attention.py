# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Inkling2 XPU attention ops."""

import pytest
import torch

from vllm.platforms import current_platform

requires_xpu = pytest.mark.skipif(
    not current_platform.is_xpu() or not torch.xpu.is_available(),
    reason="requires Intel XPU",
)

HEAD_DIM = 128
REL_EXTENT = 1024


def shear_rel_logits(rel_logits, seqs, tile, is_local):
    """Torch reference of the sheared layout written by attention_prologue."""
    tokens, heads, extent = rel_logits.shape
    out = rel_logits.new_full((tokens + 128, heads, extent + 256), float("nan"))
    cols = torch.arange(extent + 256, device=rel_logits.device)
    start = 0
    for q_len, kv_len in seqs:
        idx = torch.arange(q_len, device=rel_logits.device)
        pos = kv_len - q_len + idx
        tile_end = ((idx // tile + 1) * tile).clamp(max=q_len)
        n_max = (kv_len - q_len + tile_end + 127) // 128
        dist = pos[:, None] - (cols[None] + (n_max[:, None] - 10) * 128)
        valid = (dist >= 0) & (dist < extent)
        rel = rel_logits[start : start + q_len]
        vals = rel.gather(2, dist.clamp(0, extent - 1)[:, None].expand(-1, heads, -1))
        pad = torch.where((dist < 0) | is_local, float("-inf"), 0.0).to(
            rel_logits.dtype
        )
        out[start : start + q_len] = torch.where(valid[:, None], vals, pad[:, None])
        start += q_len
    return out


def _ref_attention(
    q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
):
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    start = 0
    for i, (q_len, kv_len) in enumerate(seqs):
        blocks = block_table[i].long()
        k, v = (
            cache[blocks]
            .flatten(0, 1)[:kv_len]
            .float()
            .repeat_interleave(num_heads // num_kv_heads, dim=1)
            for cache in (key_cache, value_cache)
        )
        qpos = torch.arange(kv_len - q_len, kv_len, device=q.device)
        dist = qpos[:, None] - torch.arange(kv_len, device=q.device)[None]
        rel = rel_logits[start : start + q_len].float().transpose(0, 1)
        idx = dist.clamp(0, REL_EXTENT - 1).expand(num_heads, -1, -1)
        bias = rel.gather(2, idx).masked_fill(~((dist >= 0) & (dist < REL_EXTENT)), 0)
        qk = torch.einsum("qhd,khd->hqk", q[start : start + q_len].float(), k)
        scores = (qk + bias / scale) * scale
        mask = dist < 0
        if window_left is not None:
            mask |= dist > window_left
        scores.masked_fill_(mask, float("-inf"))
        lse[start : start + q_len] = scores.logsumexp(-1).T
        out[start : start + q_len] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        start += q_len
    return out, lse


def _xpu_attention_inputs(seqs, layout, is_local):
    rng = torch.Generator().manual_seed(128)
    heads, kv_heads, page = 24, 8, 64
    tokens = sum(q_len for q_len, _ in seqs)
    max_pages = (max(kv_len for _, kv_len in seqs) + page - 1) // page
    blocks = len(seqs) * max_pages
    kv = torch.randn(blocks + 1, page, kv_heads, 2 * HEAD_DIM, generator=rng)
    kv[..., :HEAD_DIM] = torch.nn.functional.normalize(kv[..., :HEAD_DIM], dim=-1) * 2
    kv = kv.bfloat16()
    if layout == "HND":
        kv = kv.transpose(1, 2).contiguous().transpose(1, 2)
    k, v = kv.split(HEAD_DIM, dim=-1)
    bt = (torch.randperm(blocks, generator=rng).view(len(seqs), max_pages) + 1).int()
    q = torch.nn.functional.normalize(
        torch.randn(tokens, heads, HEAD_DIM, generator=rng), dim=-1
    )
    q = (q * 3).bfloat16()
    bias = torch.randn(tokens, heads, REL_EXTENT, generator=rng).bfloat16()
    rel = shear_rel_logits(bias, seqs, 128, is_local)
    scale = HEAD_DIM**-0.5
    ref = _ref_attention(
        q, k, v, bias, seqs, bt, scale, REL_EXTENT - 1 if is_local else None
    )
    kv = kv.xpu()
    k, v = kv.split(HEAD_DIM, dim=-1)
    launch = dict(
        block_table=bt.xpu(),
        cache_seqlens=torch.tensor(
            [kv_len for _, kv_len in seqs], dtype=torch.int32, device="xpu"
        ),
        cu_seqlens_q=torch.tensor([0, *(q_len for q_len, _ in seqs)])
        .cumsum(0)
        .int()
        .xpu(),
        max_query_len=max(q_len for q_len, _ in seqs),
        softmax_scale=scale,
    )
    return q.xpu(), k, v, rel.xpu(), launch, ref


@requires_xpu
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("is_local", [False, True])
@torch.inference_mode()
def test_xpu_rel_attention_prefill(layout, is_local):
    from vllm_inkling2_xpu.ops.attention_prefill import _kv_layout, rel_attention
    from vllm_inkling2_xpu.ops.attention_triton import MODE_OUT_LSE

    q, k, v, rel, launch, (ref_out, ref_lse) = _xpu_attention_inputs(
        [(193, 1709), (37, 37)], layout, is_local
    )
    assert _kv_layout(k, v) == (
        (64 * 8, 64, 0, 2 * HEAD_DIM)
        if layout == "HND"
        else (64, 0, 2 * HEAD_DIM, 16 * HEAD_DIM)
    )

    def run():
        out = torch.full_like(q, torch.nan, dtype=torch.float32)
        lse = torch.full((q.shape[1], q.shape[0]), torch.nan, device=q.device)
        rel_attention(
            q,
            k,
            v,
            out,
            lse,
            rel,
            **launch,
            window=REL_EXTENT if is_local else 0,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )
        return out, lse.T

    out, lse = run()
    torch.testing.assert_close(out.cpu(), ref_out, atol=3e-3, rtol=1e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, atol=1e-4, rtol=1e-5)
    for actual, repeat in zip((out, lse), run()):
        assert torch.equal(actual.view(torch.int32), repeat.view(torch.int32))


@requires_xpu
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize(
    "q_len,kv_len,num_splits", [(1, 33, 1), (1, 6144, None), (8, 6144, None)]
)
@torch.inference_mode()
def test_xpu_rel_attention_decode(is_local, q_len, kv_len, num_splits):
    from vllm_inkling2_xpu.ops.attention import (
        plan_decode_attention,
        rel_decode_attention,
    )

    q, k, v, rel, launch, (ref_out, ref_lse) = _xpu_attention_inputs(
        [(q_len, kv_len)], "NHD", is_local
    )
    if num_splits is None:
        num_splits = plan_decode_attention(
            is_local, 1, q_len, q.shape[1], k.shape[2], REL_EXTENT
        )
    rng = torch.Generator().manual_seed(3)
    width = q.shape[1] * HEAD_DIM
    gate = torch.randn(q_len, width, generator=rng).bfloat16()
    sinks = torch.randn(q_len, q.shape[1], generator=rng).bfloat16()
    ref = ref_out.flatten(1) * gate.float().sigmoid()
    ref *= (ref_lse - sinks.float()).sigmoid().repeat_interleave(HEAD_DIM, dim=1)
    q = torch.cat((q, q[:1]))
    gate = torch.cat((gate, gate[:1])).xpu()
    sinks = torch.cat((sinks, sinks[:1])).xpu()

    def run():
        out = torch.full_like(gate, torch.nan)
        rel_decode_attention(
            q,
            k,
            v,
            rel,
            gate,
            sinks,
            out,
            **launch,
            num_actual_tokens=q_len,
            is_local=is_local,
            num_splits=num_splits,
        )
        return out

    out = run()
    torch.testing.assert_close(out[:q_len].float().cpu(), ref, atol=3e-3, rtol=1e-2)
    assert (out[q_len:] == 0).all()
    assert torch.equal(out.view(torch.int16), run().view(torch.int16))
