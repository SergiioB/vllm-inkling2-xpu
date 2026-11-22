# vLLM Inkling 2 XPU plugin

Out-of-tree support for running Inkling 2 Mini on Intel XPU with vLLM. The plugin
registers the XPU model implementations and enables the model's `inkling2`
quantization on XPU; it does not patch or vendor vLLM itself.

## Compatibility

Requires an unmodified vLLM main checkout at or after `c32251b5d1` (Inkling 2
Mini support, #69028); no released vLLM version contains the shared model
implementation. Requires a PyTorch `2.15+xpu` environment. Tested on one Intel
Arc Pro B70.

The plugin uses vLLM's documented `vllm.general_plugins` entry point and
`ModelRegistry.register_model` API. vLLM's internal
`vllm.plugins.load_general_plugins` is called before model quantization
validation in the API server, engine core, and worker. Compatibility-sensitive
internals include `vllm.platforms.xpu.XPUPlatform.supported_quantization` and the
shared `vllm.models.inkling2.{configs,processing,quant_config,speculator_config}`
modules, `vllm.models.inkling.nvidia.sconv_swa_attn`, and interfaces in
`vllm.config`, `vllm.forward_context`, `vllm.model_executor.*`, `vllm.multimodal`,
`vllm.triton_utils`, `vllm.v1.*`, `vllm.distributed`, `vllm.envs`, `vllm.sequence`,
and `vllm.utils`. The registration test uses the private
`ModelRegistry._try_load_model_cls` helper only to verify lazy registration
targets. Changes to these internals may require a plugin update.

## Install

Install into an XPU-enabled vLLM environment, such as the
`vllm/vllm-openai-xpu` image or a source build:

```bash
python -m pip install --no-deps /path/to/vllm-inkling2-xpu
```


## Tests

Run from a vLLM source checkout after installing the plugin:

```bash
PYTHONPATH="$PWD:/path/to/plugin/tests" pytest -q -p tests.conftest /path/to/plugin/tests
```

```bash
vllm serve thinkingmachines/Inkling-2-Mini \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 1 \
  --block-size 64 \
  --no-enable-prefix-caching \
  -cc.mode=none \
  -cc.cudagraph_mode=FULL_DECODE_ONLY
```

Optionally enable DSpark with seven speculative tokens:

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":7}'
```

## Scope and performance

Single-device, text-only inference. Activations are BF16; expert weights use
NVFP4 or MXFP8, following vLLM's Hopper path. Tensor/expert parallelism, LoRA,
and `torch.compile` are not supported.

Development-tier measurements on one Arc Pro B70 with vLLM PR #69186: 57.7
tok/s decode; 75.8 tok/s with DSpark k=7 and mean acceptance length 2.99; about
3,250 tok/s prefill for a 2,000-token prompt. Without #69186, decode is about
7.5 ms/token slower than these headline numbers.

## TODO — Numerical parity

Numerical-parity results are being produced separately.

This project is not affiliated with vLLM, Intel, or Thinking Machines.
