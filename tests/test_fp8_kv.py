# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-runnable unit tests for FP8 E4M3 KV cache helpers and LSE chunk merge."""

import dataclasses
from types import SimpleNamespace

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


def test_fp8_prefill_scratch_shares_one_buffer_per_page_shape(monkeypatch):
    from vllm_inkling2_xpu import attention

    # Shrink the scratch budget so the reservation stays small on CPU.
    monkeypatch.setattr(attention, "_FP8_PREFILL_SCRATCH_MAX_BYTES", 1 << 20)

    cpu = torch.device("cpu")
    buf = attention._fp8_prefill_scratch(cpu, 8, 64, 256)
    assert buf is not None
    assert buf.dtype == torch.bfloat16
    # 1 MiB budget / (8 * 64 * 256 * 2 B) pages = 4 pages.
    assert buf.shape == (4, 8, 64, 256)
    # Every layer with the same page shape shares a single buffer.
    assert attention._fp8_prefill_scratch(cpu, 8, 64, 256) is buf
    # A different page shape gets its own scratch.
    other = attention._fp8_prefill_scratch(cpu, 4, 64, 256)
    assert other is not None and other is not buf
    # Pages not divisible by 16 can never run the fast path: reserve nothing.
    assert attention._fp8_prefill_scratch(cpu, 8, 12, 256) is None
    # A single page larger than the budget reserves nothing.
    assert attention._fp8_prefill_scratch(cpu, 8, 16, 1 << 21) is None


def test_single_prefill_seq_len_reads_cpu_metadata():
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from vllm_inkling2_xpu.attention import _single_prefill_seq_len

    common = CommonAttentionMetadata(
        query_start_loc=torch.zeros(3, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 5], dtype=torch.int32),
        seq_lens=torch.tensor([12, 40], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=4,
        max_seq_len=40,
        block_table_tensor=torch.zeros((2, 4), dtype=torch.int32),
        slot_mapping=torch.zeros(5, dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.tensor([12, 40], dtype=torch.int32),
    )
    # One decode + one prefill: the prefill row's length is exact on CPU.
    assert _single_prefill_seq_len(common, 1, 1) == 40
    # Not exactly one prefill: no single prefill length to describe.
    assert _single_prefill_seq_len(common, 0, 2) is None
    assert _single_prefill_seq_len(common, 1, 0) is None
    # CPU seq lens unavailable: callers must fall back to the device tensor.
    no_cpu = dataclasses.replace(common, seq_lens_cpu_upper_bound=None)
    assert _single_prefill_seq_len(no_cpu, 1, 1) is None


def test_prefill_attention_reads_seq_len_from_batch_metadata(monkeypatch):
    from vllm_inkling2_xpu import attention

    monkeypatch.setattr(attention, "_FP8_PREFILL_SCRATCH_MAX_BYTES", 1 << 20)

    layer = object.__new__(Inkling2Attention)
    layer.fp8_kv = True
    layer.is_local = False
    layer.sliding_window = 0
    layer.head_dim = 128
    layer.scaling = 128**-0.5

    calls = []

    def fake_rel_attention(q, k, v, out, lse, rel, **kwargs):
        calls.append(kwargs.get("k_hi_tok", -1))

    monkeypatch.setattr(attention, "rel_attention", fake_rel_attention)

    nt, heads = 4, 24
    kv_cache = torch.zeros((8, 8, 64, 256), dtype=torch.uint8)
    md = SimpleNamespace(
        num_actual_tokens=nt,
        num_decode_tokens=0,
        max_query_len=nt,
        query_start_loc=torch.tensor([0, nt], dtype=torch.int32),
        # seq_lens disagrees with single_prefill_seq_len on purpose: the
        # CPU-side metadata value must win with no device-tensor read.
        seq_lens=torch.tensor([999], dtype=torch.int32),
        block_table=torch.arange(8, dtype=torch.int32).unsqueeze(0),
        single_prefill_seq_len=40,
    )
    q = torch.zeros((nt, heads, 128), dtype=torch.bfloat16)
    rel = torch.zeros((nt + 128, heads, 1280), dtype=torch.bfloat16)
    out = torch.zeros((nt, heads, 128), dtype=torch.float32)
    lse = torch.zeros((heads, nt), dtype=torch.float32)

    # seq_len=40 -> end_page=1 -> bounded chunk ends at page 1 * 64 tokens.
    layer._prefill_attention(md, kv_cache, q, rel, out, lse, 0)
    assert calls == [64]

    # Without CPU metadata the device seq len drives the same bound.
    calls.clear()
    md.single_prefill_seq_len = None
    md.seq_lens = torch.tensor([40], dtype=torch.int32)
    layer._prefill_attention(md, kv_cache, q, rel, out, lse, 0)
    assert calls == [64]

    # A seq len overflowing the block table takes the unbounded path.
    calls.clear()
    md.seq_lens = torch.tensor([999], dtype=torch.int32)
    layer._prefill_attention(md, kv_cache, q, rel, out, lse, 0)
    assert calls == [-1]


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
