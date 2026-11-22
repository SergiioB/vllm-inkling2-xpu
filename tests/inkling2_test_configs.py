# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny Inkling2 checkpoint config used by the XPU graph test."""

from typing import Any

TINY_CONFIG: dict[str, Any] = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "eos_token_id": 1,
    "text_config": {
        "torch_dtype": "bfloat16",
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "vocab_size": 1024,
        "unpadded_vocab_size": 1000,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "n_routed_experts": 16,
        "num_experts_per_tok": 8,
        "intermediate_size": 256,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
    },
    "mtp_config": {
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e6,
        "max_position_embeddings": 4096,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "block_size": 8,
        "markov_rank": 32,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [1, 2],
    },
}

TINY_LOOP_CONFIG: dict[str, Any] = {
    **TINY_CONFIG,
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "text_config": {
        **TINY_CONFIG["text_config"],
        "hidden_size": 2304,
        "num_hidden_layers": 4,
        "local_layer_ids": [0, 1, 3],
        "n_layers_per_attn_res_block": 1,
        "loop_config": {
            "core_start": 1,
            "core_end": 3,
            "num_passes": 3,
            "gate_rank": 16,
            "is_varloop": True,
        },
    },
    "mtp_config": {
        **TINY_CONFIG["mtp_config"],
        "hidden_size": 2304,
        "target_layer_ids": [3, 8],
    },
}
