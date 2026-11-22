# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-runnable unit tests for FP8 E4M3 KV cache helpers and LSE chunk merge."""

import torch

from vllm_inkling2_xpu.attention import Inkling2Attention


def test_split_kv_cache_views_uint8_as_float8_e4m3fn():
    dummy = object.__new__(Inkling2Attention)
    dummy.fp8_kv = True
    dummy.head_dim = 128

    raw = torch.zeros((4, 8, 64, 256), dtype=torch.uint8)
    k_cache, v_cache = dummy._split_kv_cache(raw)
    assert k_cache.dtype == torch.float8_e4m3fn
    assert v_cache.dtype == torch.float8_e4m3fn
    assert k_cache.shape == (4, 64, 8, 128)
    assert v_cache.shape == (4, 64, 8, 128)


def test_online_flash_lse_chunk_merge_exact_parity():
    torch.manual_seed(7)
    nt, h, d, n_keys = 16, 4, 32, 128
    q = torch.randn(nt, h, d)
    k = torch.randn(n_keys, h, d)
    v = torch.randn(n_keys, h, d)
    scale = d**-0.5

    # Unchunked reference
    scores = torch.einsum("thd,khd->htk", q, k) * scale
    lse_ref = torch.logsumexp(scores, dim=-1)  # [h, nt]
    probs = torch.softmax(scores, dim=-1)
    out_ref = torch.einsum("htk,khd->thd", probs, v)

    # 2-chunk exact LSE merge (mirrors _prefill_attention)
    s0, s1 = scores[..., :64], scores[..., 64:]
    l0, l1 = torch.logsumexp(s0, dim=-1), torch.logsumexp(s1, dim=-1)
    o0 = torch.einsum("htk,khd->thd", torch.softmax(s0, dim=-1), v[:64])
    o1 = torch.einsum("htk,khd->thd", torch.softmax(s1, dim=-1), v[64:])

    m = torch.maximum(l0, l1)
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w0 = torch.exp(l0 - m_safe)
    w1 = torch.exp(l1 - m_safe)
    ws = w0 + w1
    ws_safe = torch.where(ws == 0, torch.ones_like(ws), ws)
    lse_merged = torch.where(
        ws == 0, torch.full_like(m, float("-inf")), m_safe + torch.log(ws_safe)
    )
    out_merged = o0 * (w0 / ws_safe).t().unsqueeze(-1) + o1 * (
        w1 / ws_safe
    ).t().unsqueeze(-1)

    assert torch.allclose(lse_merged, lse_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_merged, out_ref, atol=1e-5, rtol=1e-5)


def test_bounded_prefill_fails_loudly_when_fast_path_guard_fails():
    from vllm_inkling2_xpu.ops import attention_prefill, attention_triton

    # page=12 is not divisible by BLOCK_N=16 -> fast-path guard fails
    q = torch.zeros((4, 24, 128), dtype=torch.bfloat16)
    kv = torch.zeros((2, 8, 12, 256), dtype=torch.bfloat16)
    k, v = kv.transpose(1, 2).split(128, dim=-1)
    rel = torch.zeros((132, 24, 1280), dtype=torch.bfloat16)
    bt = torch.zeros((1, 2), dtype=torch.int32)
    seqlens = torch.tensor([12], dtype=torch.int32)
    cu_q = torch.tensor([0, 4], dtype=torch.int32)
    out = torch.zeros((4, 24, 128), dtype=torch.bfloat16)
    lse = torch.zeros((24, 4), dtype=torch.float32)

    try:
        attention_prefill.rel_attention(
            q,
            k,
            v,
            out,
            lse,
            rel,
            block_table=bt,
            cache_seqlens=seqlens,
            cu_seqlens_q=cu_q,
            max_query_len=4,
            softmax_scale=128**-0.5,
            window=0,
            num_splits=1,
            mode=attention_triton.MODE_OUT_LSE,
            k_lo_tok=0,
            k_hi_tok=12,
        )
    except ValueError as exc:
        assert "Bounded prefill attention" in str(exc)
    else:
        raise AssertionError("Expected ValueError for bounded non-fast-path call")


def test_scratch_layout_resolves_for_single_page_and_single_kv_head():
    from vllm_inkling2_xpu.ops.attention_prefill import _kv_layout

    for cnt, kvh, page, d in [(1, 8, 64, 128), (1, 2, 16, 128), (4, 1, 64, 128)]:
        buf = torch.empty((cnt, kvh, page, 2 * d), dtype=torch.bfloat16)
        kc, vc = buf.transpose(1, 2).split(d, dim=-1)
        layout = _kv_layout(kc, vc)
        assert layout is not None, f"Expected HND layout for {(cnt, kvh, page, d)}"


def test_unaligned_k_lo_tok_raises_value_error():
    from vllm_inkling2_xpu.ops import attention_prefill, attention_triton

    q = torch.zeros((4, 24, 128), dtype=torch.bfloat16)
    kv = torch.zeros((2, 8, 64, 256), dtype=torch.bfloat16)
    k, v = kv.transpose(1, 2).split(128, dim=-1)
    rel = torch.zeros((132, 24, 1280), dtype=torch.bfloat16)
    bt = torch.zeros((1, 2), dtype=torch.int32)
    seqlens = torch.tensor([64], dtype=torch.int32)
    cu_q = torch.tensor([0, 4], dtype=torch.int32)
    out = torch.zeros((4, 24, 128), dtype=torch.bfloat16)
    lse = torch.zeros((24, 4), dtype=torch.float32)

    for bad_lo in (-16, 8):
        try:
            attention_prefill.rel_attention(
                q,
                k,
                v,
                out,
                lse,
                rel,
                block_table=bt,
                cache_seqlens=seqlens,
                cu_seqlens_q=cu_q,
                max_query_len=4,
                softmax_scale=128**-0.5,
                window=0,
                num_splits=1,
                mode=attention_triton.MODE_OUT_LSE,
                k_lo_tok=bad_lo,
                k_hi_tok=64,
            )
        except ValueError as exc:
            assert "k_lo_tok" in str(exc)
        else:
            raise AssertionError(f"Expected ValueError for k_lo_tok={bad_lo}")
