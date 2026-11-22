# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entry-point registration tests for the out-of-tree XPU plugin."""

from importlib import import_module, metadata

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu() or not torch.xpu.is_available(),
    reason="requires Intel XPU",
)


def test_installed_entrypoint_registers_xpu_models_and_is_reentrant(monkeypatch):
    entrypoints = metadata.entry_points(group="vllm.general_plugins")
    plugin = [
        entrypoint for entrypoint in entrypoints if entrypoint.name == "inkling2_xpu"
    ]
    assert len(plugin) == 1
    register = plugin[0].load()

    from vllm import ModelRegistry
    from vllm.platforms.xpu import XPUPlatform

    expected = {
        "Inkling2ForConditionalGeneration": (
            "vllm_inkling2_xpu.model",
            "Inkling2ForConditionalGeneration",
        ),
        "Inkling2LoopForConditionalGeneration": (
            "vllm_inkling2_xpu.loop",
            "Inkling2LoopForConditionalGeneration",
        ),
        "Inkling2DSparkModel": (
            "vllm_inkling2_xpu.dspark",
            "Inkling2DSparkModel",
        ),
    }

    register()
    assert "inkling2" in XPUPlatform.supported_quantization
    for architecture, (module, class_name) in expected.items():
        assert ModelRegistry._try_load_model_cls(architecture) is getattr(
            import_module(module), class_name
        )

    register()
    assert XPUPlatform.supported_quantization.count("inkling2") == 1
    registered_models = {
        architecture: ModelRegistry.models[architecture] for architecture in expected
    }
    supported_quantization = list(XPUPlatform.supported_quantization)

    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    register()

    assert XPUPlatform.supported_quantization == supported_quantization
    assert all(
        ModelRegistry.models[architecture] is model
        for architecture, model in registered_models.items()
    )
