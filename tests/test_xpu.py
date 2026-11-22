# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness and graph replay tests for Inkling2 on Intel XPU."""

import json

import pytest
import torch

from inkling2_test_configs import TINY_LOOP_CONFIG
from vllm import SamplingParams
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu() or not torch.xpu.is_available(),
    reason="requires Intel XPU",
)


@pytest.mark.parametrize("tokens", [1, 32, 130, 300, 2048])
def test_xpu_k_group_linear(tokens):
    from vllm_inkling2_xpu.ops.linear import k_group_partials, linear_k_groups

    rng = torch.Generator().manual_seed(12)
    x = torch.randn(tokens, 2304, generator=rng).bfloat16()
    w = torch.randn(192, x.shape[1], generator=rng).bfloat16()
    ref = torch.stack(
        [a.float() @ b.float().T for a, b in zip(x.split(1024, 1), w.split(1024, 1))]
    )
    x, w = x.xpu(), w.xpu()
    partials = k_group_partials(x, w)
    out = linear_k_groups(x, w)
    torch.testing.assert_close(partials.cpu(), ref, rtol=2e-5, atol=5e-5)
    torch.testing.assert_close(
        out.cpu(), ref[0] + ref[1] + ref[2], rtol=2e-5, atol=1e-4
    )
    assert torch.equal(
        partials.view(torch.int32), k_group_partials(x, w).view(torch.int32)
    )
    assert torch.equal(out.view(torch.int32), linear_k_groups(x, w).view(torch.int32))


@pytest.mark.parametrize("width", [128, 1024, 2048, 2304, 4608])
def test_xpu_rms_norm(width):
    from vllm_inkling2_xpu.ops.norm import rms_norm

    rng = torch.Generator().manual_seed(98)
    x = torch.randn(37, width, generator=rng).bfloat16()
    w = torch.randn(width, generator=rng).bfloat16()
    ref = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6)
    ref = (ref * w).bfloat16()
    x, w = x.xpu(), w.xpu()
    out = rms_norm(x, w, 1e-6)
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=2**-7)
    assert torch.equal(out.view(torch.int16), rms_norm(x, w, 1e-6).view(torch.int16))


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("fused_add", [False, True])
@torch.inference_mode()
def test_xpu_dspark_rmsnorm(rows, fused_add):
    from vllm_inkling2_xpu.dspark import _RMSNorm

    rng = torch.Generator().manual_seed(852)
    width, eps = 2304, 1e-6
    x = torch.randn(rows, width, generator=rng).bfloat16()
    residual = torch.randn(rows, width, generator=rng).bfloat16() if fused_add else None
    weight = (0.5 + torch.rand(width, generator=rng)).bfloat16()
    norm = _RMSNorm(width, eps).to("xpu", torch.bfloat16)
    norm.weight.copy_(weight)
    ref_sum = x.double()
    if residual is not None:
        ref_sum = ref_sum + residual.double()
    ref = ref_sum * torch.rsqrt(ref_sum.square().mean(-1, keepdim=True) + eps) * weight

    def run():
        actual_x = x.xpu()
        actual_residual = residual.xpu() if residual is not None else None
        out = norm(actual_x, actual_residual)
        if actual_residual is not None:
            assert out is actual_x
            torch.testing.assert_close(
                actual_residual.cpu(), ref_sum.bfloat16(), rtol=0, atol=0
            )
        return out

    out = run()
    torch.testing.assert_close(out.cpu(), ref.bfloat16(), rtol=2**-7, atol=0)
    assert torch.equal(out.view(torch.int16), run().view(torch.int16))


def test_xpu_full_decode_graph_matches_eager(tmp_path, vllm_runner):
    """Check graph replay of the looped Mini architecture with packed experts."""
    from vllm.config import CUDAGraphMode

    (tmp_path / "config.json").write_text(json.dumps(TINY_LOOP_CONFIG))
    outputs = []
    for enforce_eager in (False, True):
        with vllm_runner(
            str(tmp_path),
            load_format="dummy",
            skip_tokenizer_init=True,
            tokenizer_mode="hf",
            dtype="bfloat16",
            language_model_only=True,
            block_size=64,
            max_num_seqs=1,
            max_num_batched_tokens=128,
            enable_chunked_prefill=True,
            max_model_len=256,
            gpu_memory_utilization=0.1,
            enable_prefix_caching=False,
            compilation_config={
                "mode": "none",
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1],
            },
            enforce_eager=enforce_eager,
        ) as model:
            graph_mode = (
                model.llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
            )
            assert graph_mode == (
                CUDAGraphMode.NONE if enforce_eager else CUDAGraphMode.FULL_DECODE_ONLY
            )
            result = model.llm.generate(
                [{"prompt_token_ids": list(range(2, 100))}],
                SamplingParams(
                    temperature=0, max_tokens=12, detokenize=False, ignore_eos=True
                ),
            )
            ids = result[0].outputs[0].token_ids
            assert len(ids) == 12
            outputs.append(ids)
    assert outputs[0] == outputs[1]
