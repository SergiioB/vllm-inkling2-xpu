# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 packed NVFP4 and MXFP8 experts for Intel XPU."""

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import nn

import vllm.envs as envs
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import is_weights_pre_processed, set_weight_attrs

from vllm.models.inkling2.configs import Inkling2ModelConfig
from .ops._codec import PackedProjection
from .ops.moe import packed_experts, validate_config
from .ops.moe_packed import repack_nvfp4_split_half
from .ops.moe_router import select_experts

QuantFormat = Literal["nvfp4", "mxfp8"]


class Inkling2Gate(nn.Module):
    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        num_experts, hidden = config.n_routed_experts, config.hidden_size
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        for param in (self.weight, self.bias):
            set_weight_attrs(param, {"weight_loader": default_weight_loader})


class Inkling2Experts(nn.Module):
    def __init__(self, config: Inkling2ModelConfig, quant_format: QuantFormat) -> None:
        super().__init__()
        hidden, inter = config.hidden_size, config.intermediate_size
        validate_config(
            hidden,
            inter,
            config.num_experts_per_tok,
            config.n_routed_experts,
            quant_format,
        )
        self.num_experts = config.n_routed_experts
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.nvfp4 = quant_format == "nvfp4"

        pack, group = (2, 16) if self.nvfp4 else (1, 32)
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        shapes = ((2 * inter, hidden), (hidden, inter))
        for name, (n, k) in zip(("w13", "w2"), shapes):
            self._register(f"{name}_weight", (n, k // pack), weight_dtype)
            self._register(f"{name}_weight_scale", (n, k // group), scale_dtype)
            if self.nvfp4:
                self._register(f"{name}_weight_amax", (), torch.float32)
                self._register(
                    f"{name}_weight_outer_scale_denominator",
                    (1,),
                    torch.float32,
                    per_expert=False,
                )

    def _register(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        per_expert: bool = True,
    ) -> None:
        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        set_weight_attrs(param, {"weight_loader": self._load_full})
        self.register_parameter(name, param)

    @staticmethod
    def _load_full(param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded)

    def process_weights_after_loading(self) -> None:
        # Kernel-format reloads already hold split-half weights.
        if self.nvfp4 and not is_weights_pre_processed():
            repack_nvfp4_split_half(self.w13_weight)
            repack_nvfp4_split_half(self.w2_weight)

    def scales(self, name: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the NVFP4 outer scale amax and denominator for a projection."""
        if not self.nvfp4:
            return None, None
        return (
            getattr(self, f"{name}_weight_amax"),
            getattr(self, f"{name}_weight_outer_scale_denominator"),
        )

    def _packed_weights(self, name: str) -> PackedProjection:
        return (
            getattr(self, f"{name}_weight"),
            getattr(self, f"{name}_weight_scale"),
            *self.scales(name),
        )

    def forward(
        self,
        x: torch.Tensor,
        routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        """Run packed projections and combine the selected experts' outputs."""
        ids, weights, is_padding = routing
        return packed_experts(
            x,
            ids,
            weights,
            is_padding,
            self._packed_weights("w13"),
            self._packed_weights("w2"),
            self.gate_cap,
            self.up_cap,
        )


class Inkling2RoutingCapture(nn.Module):
    """Expose a capture callback for the expert IDs of one layer visit."""

    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.capture_fn: Callable[[torch.Tensor], None] | None = None


class Inkling2MoE(nn.Module):
    """Route to packed experts on a single XPU."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_format: QuantFormat,
        visit_ids: Sequence[int] = (0,),
    ) -> None:
        super().__init__()
        if config.n_shared_experts != 0 or not config.use_gate_bias:
            raise NotImplementedError(
                "Inkling2 XPU requires routed experts with gate bias"
            )
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config, quant_format)
        self.routing_captures = nn.ModuleList(
            Inkling2RoutingCapture(visit) for visit in visit_ids
        )

    def _select_experts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate
        return select_experts(x, gate.weight, gate.bias, gate.topk, gate.route_scale)

    def routed_expert_ids(self, x: torch.Tensor) -> torch.Tensor:
        return self._select_experts(x)[0]

    def forward(self, x: torch.Tensor, pass_idx: int = 0) -> torch.Tensor:
        if torch.compiler.is_compiling():
            raise NotImplementedError("Inkling2 XPU MoE does not support torch.compile")
        x = x.contiguous()
        ids, weights = self._select_experts(x)
        capture_fn = self.routing_captures[pass_idx].capture_fn
        if capture_fn is not None:
            capture_fn(ids)
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
        return self.experts(x, (ids, weights, is_padding))
