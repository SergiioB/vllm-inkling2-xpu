# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU implementation of Inkling2."""

from vllm.platforms import current_platform


def register() -> None:
    """Register the Inkling 2 XPU model implementations with vLLM."""
    if not current_platform.is_xpu():
        return

    from vllm import ModelRegistry
    from vllm.platforms.xpu import XPUPlatform

    if "inkling2" not in XPUPlatform.supported_quantization:
        XPUPlatform.supported_quantization.append("inkling2")

    for architecture, model in (
        (
            "Inkling2ForConditionalGeneration",
            "vllm_inkling2_xpu.model:Inkling2ForConditionalGeneration",
        ),
        (
            "Inkling2LoopForConditionalGeneration",
            "vllm_inkling2_xpu.loop:Inkling2LoopForConditionalGeneration",
        ),
        (
            "Inkling2DSparkModel",
            "vllm_inkling2_xpu.dspark:Inkling2DSparkModel",
        ),
    ):
        ModelRegistry.register_model(architecture, model)
