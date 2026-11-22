# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 Mini's packed XPU experts against a dense FP64 reference."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu() or not torch.xpu.is_available(),
    reason="requires Intel XPU",
)

# Inkling 2 Mini checkpoint dimensions and activation parameters.
EXPERTS, HIDDEN, INTERMEDIATE, TOPK = 128, 2304, 1152, 8
GATE_CAP, UP_CAP = 4.0, 25.0


@torch.inference_mode()
def test_xpu_router_bias_and_ties():
    from vllm_inkling2_xpu.ops.moe_router import select_experts

    x = torch.tensor([[1.0], [-1.0]], dtype=torch.bfloat16)
    w = torch.tensor(
        [[3.0], [-1.0], [2.0], [-1.0], [0.5], [-1.0]], dtype=torch.bfloat16
    )
    bias = torch.tensor([0.0, 0.75, -0.8, 0.75, -0.3, 0.75])
    args = (x.xpu(), w.xpu(), bias.xpu())
    for k in (2, 4):
        ids, weights = select_experts(*args, k, 8.0)
        scores = (x.double() @ w.double().T).sigmoid()
        ref_ids = (scores + bias).argsort(descending=True, stable=True)[:, :k]
        ref_weights = scores.gather(1, ref_ids)
        ref_weights = ref_weights / ref_weights.sum(-1, keepdim=True) * 8
        torch.testing.assert_close(ids.cpu(), ref_ids.int())
        torch.testing.assert_close(weights.cpu(), ref_weights.float())
        ids_again, weights_again = select_experts(*args, k, 8.0)
        assert torch.equal(ids, ids_again)
        assert torch.equal(weights.view(torch.int32), weights_again.view(torch.int32))


def _projection(n, k, active_ids, fmt, rng):
    """Build packed checkpoint fields and independently dequantize the live experts."""
    experts = len(active_ids)
    if fmt == "nvfp4":
        packed = torch.randint(
            256, (experts, n, k // 2), dtype=torch.uint8, generator=rng
        )
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float64,
        )
        dense = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1)
        dense = dense.view(experts, n, k // 16, 16)
        codes = torch.tensor([0x01, 0x28, 0x34, 0x38, 0x3C, 0x44], dtype=torch.uint8)
        scales = codes[torch.randint(len(codes), (experts, n, k // 16), generator=rng)]
        scales = scales.view(torch.float8_e4m3fn)
        dense *= scales.double()[..., None]
        amax = 0.5 + torch.rand(EXPERTS, generator=rng)
        denominator = torch.tensor([k**0.5 * 2])
        dense *= (amax[active_ids].double() / denominator.double())[:, None, None, None]
        outer = (amax.xpu(), denominator.xpu())
    else:
        codes = torch.tensor(
            [0x00, 0x02, 0x84, 0x21, 0xA8, 0x38, 0xB9, 0x4B, 0xCA, 0x58, 0xD8],
            dtype=torch.uint8,
        )
        packed = codes[torch.randint(len(codes), (experts, n, k), generator=rng)]
        scales = torch.randint(
            117, 122, (experts, n, k // 32), dtype=torch.uint8, generator=rng
        )
        dense = packed.view(torch.float8_e4m3fn).double().view(experts, n, k // 32, 32)
        dense *= (2.0 ** (scales.double() - 127))[..., None]
        outer = (None, None)

    def to_xpu(t):
        out = torch.zeros((EXPERTS, *t.shape[1:]), dtype=torch.uint8, device="xpu")
        out.index_copy_(0, active_ids.xpu(), t.view(torch.uint8).xpu())
        return out.view(t.dtype)

    return (to_xpu(packed), to_xpu(scales), *outer), dense.view(experts, n, k)


@pytest.fixture(scope="module", params=["nvfp4", "mxfp8"])
def projections(request):
    from vllm_inkling2_xpu.ops.moe_packed import repack_nvfp4_split_half

    rng = torch.Generator().manual_seed(291)
    active_ids = torch.tensor([0, 1, 7, 11, 13, 23, 39, 57, 73, 88, 104, 127])
    w13, ref13 = _projection(2 * INTERMEDIATE, HIDDEN, active_ids, request.param, rng)
    w2, ref2 = _projection(HIDDEN, INTERMEDIATE, active_ids, request.param, rng)
    if request.param == "nvfp4":
        repack_nvfp4_split_half(w13[0])
        repack_nvfp4_split_half(w2[0])
    return active_ids, w13, w2, ref13, ref2


def _dense_reference(x, ids, weights, is_padding, w13, w2):
    x = x.double()
    down = torch.zeros((*ids.shape, HIDDEN), dtype=torch.float64)
    for i, (gate_up, proj) in enumerate(zip(w13, w2)):
        tokens, slots = ((ids == i) & ~is_padding[:, None]).nonzero(as_tuple=True)
        if tokens.numel() == 0:
            continue
        rounded = (x[tokens] @ gate_up.T).bfloat16().double()
        gate, up = rounded[:, ::2], rounded[:, 1::2]
        y = (GATE_CAP * (gate / GATE_CAP).tanh()) * gate.sigmoid()
        y *= UP_CAP * (up / UP_CAP).tanh()
        down[tokens, slots] = (y.bfloat16().double() @ proj.T).bfloat16().double()
    contributions = down * weights.double()[..., None]
    return contributions.sum(1).bfloat16(), contributions.abs().sum(1)


@pytest.mark.parametrize("tokens", [1, 2, 4, 5, 8, 64, 300])
@torch.inference_mode()
def test_xpu_packed_experts(projections, tokens):
    from vllm_inkling2_xpu.ops.moe import packed_experts

    active_ids, w13, w2, ref13, ref2 = projections
    rng = torch.Generator().manual_seed(1801 + tokens)
    x = torch.randn(tokens, HIDDEN, generator=rng).bfloat16()
    ids = torch.rand((tokens, active_ids.numel()), generator=rng).argsort()[:, :TOPK]
    weights = torch.randn(tokens, TOPK, generator=rng).softmax(-1) * 8
    padding = torch.arange(tokens) % 7 == 3
    x[padding] = torch.nan
    expected, magnitude = _dense_reference(x, ids, weights, padding, ref13, ref2)
    args = (x.xpu(), active_ids[ids].int().xpu(), weights.xpu())
    # 300 tokens uses the large GEMM tile; 5, 8 and 64 use the small tile.
    mask = padding.xpu() if tokens > 1 else None
    out = packed_experts(*args, mask, w13, w2, GATE_CAP, UP_CAP)
    # Rounding differences in the BF16 intermediates scale with the summed
    # expert contributions, not with the (possibly cancelled) output, so bound
    # each element by eps times sum_j |w_j * down_j|. The kernel stays within
    # 1.1x and an fp32-GEMM reference within 1.3x of that on these inputs.
    error = (out.cpu().double() - expected.double()).abs()
    eps = torch.finfo(torch.bfloat16).eps
    assert (error <= 2 * eps * magnitude).all()
    assert (out.cpu()[padding] == 0).all()
    again = packed_experts(*args, mask, w13, w2, GATE_CAP, UP_CAP)
    assert torch.equal(out.view(torch.int16), again.view(torch.int16))
    if tokens in (1, 5):
        # A wholly padded batch must not read uninitialized projection scratch.
        mask = torch.ones(tokens, dtype=torch.bool, device=out.device)
        padded = packed_experts(*args, mask, w13, w2, GATE_CAP, UP_CAP)
        assert torch.equal(padded, torch.zeros_like(out))


@torch.inference_mode()
def test_xpu_nvfp4_repack_skipped_for_processed_weights(dist_init):
    """Kernel-format reloads restore split-half weights; repacking again would
    corrupt them because the split-half shuffle is not an involution."""
    from vllm.model_executor.utils import weights_already_processed
    from vllm.models.inkling2.configs import Inkling2ModelConfig
    from vllm_inkling2_xpu.moe import Inkling2MoE

    config = Inkling2ModelConfig(
        hidden_size=512,
        intermediate_size=256,
        n_routed_experts=16,
        n_shared_experts=0,
        num_experts_per_tok=8,
        route_scale=8.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        use_gate_bias=True,
    )
    with torch.device("xpu"):
        experts = Inkling2MoE(config, "nvfp4").experts
    rng = torch.Generator().manual_seed(0)
    for weight in (experts.w13_weight, experts.w2_weight):
        raw = torch.randint(256, weight.shape, dtype=torch.uint8, generator=rng)
        weight.view(torch.uint8).copy_(raw)
    originals = [w.clone() for w in (experts.w13_weight, experts.w2_weight)]
    experts.process_weights_after_loading()
    repacked = [w.clone() for w in (experts.w13_weight, experts.w2_weight)]
    assert all(
        not torch.equal(before, after) for before, after in zip(originals, repacked)
    )
    with weights_already_processed():
        experts.process_weights_after_loading()
    assert torch.equal(experts.w13_weight, repacked[0])
    assert torch.equal(experts.w2_weight, repacked[1])
