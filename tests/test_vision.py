# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-runnable unit tests for the portable Inkling2 XPU vision encoder."""

import torch

from vllm_inkling2_xpu.vision import Inkling2VisionEncoder

_SMALL_VISION_CFG = {
    "model_type": "inkling2_vision",
    "text_hidden_size": 32,
    "temporal_patch_size": 2,
    "native_patch_size": 4,
    "adapter_merge_size": 2,
    "projection_hidden_dim": 64,
    "projector_bias": False,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "mlp_ratio": 2.0,
    "posemb_grid_size": 4,
    "layer_norm_eps": 1e-5,
}


def _init_random_weights(enc: Inkling2VisionEncoder) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        for p in enc.parameters():
            p.normal_(0.0, 0.02)


def test_vision_encoder_load_weights_splits_and_fuses_qkv():
    enc = Inkling2VisionEncoder(_SMALL_VISION_CFG, text_hidden_size=32)
    _init_random_weights(enc)

    # Synthesize checkpoint tensors with separate q_proj/k_proj/v_proj
    ckpt: list[tuple[str, torch.Tensor]] = []
    for name, param in enc.named_parameters():
        if ".self_attn.qkv_proj." in name:
            chunks = param.detach().clone().chunk(3, dim=0)
            for proj, chunk in zip(("q_proj", "k_proj", "v_proj"), chunks):
                ckpt.append((name.replace("qkv_proj", proj), chunk))
        else:
            ckpt.append((name, param.detach().clone()))

    reloaded = Inkling2VisionEncoder(_SMALL_VISION_CFG, text_hidden_size=32)
    loaded = reloaded.load_weights(ckpt)
    assert loaded == set(dict(reloaded.named_parameters()).keys())
    for name, param in enc.named_parameters():
        assert torch.equal(param, dict(reloaded.named_parameters())[name])


def test_vision_encoder_single_and_multi_image_forward():
    enc = Inkling2VisionEncoder(_SMALL_VISION_CFG, text_hidden_size=32).eval()
    _init_random_weights(enc)

    side = (
        _SMALL_VISION_CFG["native_patch_size"] * _SMALL_VISION_CFG["adapter_merge_size"]
    )
    torch.manual_seed(42)
    # Single image (2x3 merged tokens = 6 patches)
    px1 = torch.randint(0, 256, (6, side, side, 3), dtype=torch.bfloat16)
    g1 = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    # Second image (2x2 merged tokens = 4 patches)
    px2 = torch.randint(0, 256, (4, side, side, 3), dtype=torch.bfloat16)
    g2 = torch.tensor([[1, 2, 2]], dtype=torch.int64)

    with torch.no_grad():
        out1 = enc(px1, g1)
        out2 = enc(px2, g2)
        out_both = enc(torch.cat([px1, px2], dim=0), torch.cat([g1, g2], dim=0))

    assert out1.shape == (6, 32) and out1.dtype == torch.bfloat16
    assert out2.shape == (4, 32) and out2.dtype == torch.bfloat16
    assert out_both.shape == (10, 32)
    assert torch.isfinite(out_both).all()
    assert torch.allclose(out_both[:6], out1, atol=1e-2, rtol=1e-2)
    assert torch.allclose(out_both[6:], out2, atol=1e-2, rtol=1e-2)
