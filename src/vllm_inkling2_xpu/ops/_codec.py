# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, tldevice, triton

# Packed weights, block scales, and (NVFP4 only) outer scale operands.
PackedProjection = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
]


@triton.jit
def f16_e4m3(code):
    # Rebias into FP16: the exact value is scaled by 2**-8.
    bits = ((code & 0x7F) << 7) | ((code & 0x80) << 8)
    return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def f16_e2m1(code):
    # Rebias into FP16: the exact value is scaled by 2**-14.
    bits = ((code & 7) << 9) | ((code & 8) << 12)
    return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def mxfp8_scale(code):
    # 2**(code - 127) with the inverse of the E4M3 rebias applied.
    return ((code + 8) << 23).to(tl.float32, bitcast=True)


@triton.jit
def nvfp4_scale(Amax, Denominator, expert):
    return tl.div_rn(tl.load(Amax + expert), tl.load(Denominator)) * 4194304.0


@triton.jit
def _softcap(x, cap: tl.constexpr):
    if cap > 0:
        return cap * tldevice.tanh(x / cap)
    return x


@triton.jit
def swiglu(gate, up, gate_cap: tl.constexpr, up_cap: tl.constexpr):
    return (_softcap(gate, gate_cap) * tl.sigmoid(gate)) * _softcap(up, up_cap)
